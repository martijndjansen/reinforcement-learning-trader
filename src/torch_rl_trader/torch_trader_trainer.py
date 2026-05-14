import numpy as np
import plotly.graph_objects as go
import torch
import torch._dynamo
from aim import Figure, Run
from tensordict import TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictSequential,
)
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import ValueOperator
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import ExplorationType, SACLoss
from tqdm import tqdm

from torch_rl_trader.torch_trader_environment import TorchTraderEnvironment
from trader.config import Config

torch.set_float32_matmul_precision("high")

torch._dynamo.config.recompile_limit = 64

torch.manual_seed(0)


class TraderTrainer:
    def __init__(
        self,
        cfg: Config,
        train_env: TorchTraderEnvironment,
        val_env: TorchTraderEnvironment,
        policy: ProbabilisticTensorDictSequential,
        q_module: ValueOperator,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.model.device)
        # implement env = TransformedEnv(base_env)

        self.create_env_fn = lambda: train_env
        self.val_env = val_env
        self.action_spec = train_env.action_spec
        self.utd_ratio = self.cfg.model.utd_ratio

        self.warmup = 2000  # TODO: make configurable
        self.total_frames = cfg.trading.num_episodes * cfg.env.episode_length

        self.policy = policy.to(self.device)
        self.q_module = q_module.to(self.device)
        self.sac_loss = self._init_loss()
        # self.optimizer = torch.optim.Adam(self.sac_loss.parameters(), lr=1e-4)
        self.target_updater = SoftUpdate(self.sac_loss, eps=0.999)
        # self.sac_loss = torch.compile(self.sac_loss)
        self.collector = self._init_collector()
        self.replay_buffer = self._init_replay_buffer()
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.q_module.parameters(), lr=3e-4)

        self.alpha_optimizer = torch.optim.Adam([self.sac_loss.log_alpha], lr=3e-4)

    def _init_loss(self) -> SACLoss:
        sac_loss = SACLoss(
            actor_network=self.policy,
            qvalue_network=self.q_module,
            num_qvalue_nets=2,
            delay_actor=False,
            delay_qvalue=True,
            action_spec=self.action_spec,
            delay_value=True,
            target_entropy="auto",
        )
        sac_loss.make_value_estimator(gamma=0.99)
        return sac_loss

    def _init_collector(self) -> Collector:
        return Collector(
            create_env_fn=self.create_env_fn,  # type: ignore
            policy=self.policy,
            frames_per_batch=self.cfg.env.episode_length // 4,
            total_frames=self.total_frames,
            device=self.device,
            storing_device=self.device,
        )

    def _init_replay_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(storage=LazyTensorStorage(100_000, device=self.device))

    def _log_plotly_weights(
        self, weights: np.ndarray, portfolio_values: np.ndarray, aim_run: Run
    ) -> None:
        fig_1 = go.Figure()

        fig_1.add_scatter(
            y=list(portfolio_values),
            mode="lines+markers",
            name="validation episode",
        )
        fig_weights_over_time = go.Figure()

        if len(weights) > 0:
            for i, ticker in enumerate([*self.cfg.trading.tickers, "CASH"]):
                fig_weights_over_time.add_scatter(
                    y=list(weights[:, i]),
                    mode="lines",
                    hoverinfo="x+y",
                    name=ticker,
                    stackgroup="one",
                )
            aim_run.track(
                Figure(fig_weights_over_time),
                name="Val weights over time",
                # step=episode,
                context={"type": "plotly"},
            )

        fig_1.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="red",
        )

        aim_run.track(
            Figure(fig_1),
            name="Final Value",
            # step=episode,
            context={"type": "plotly"},
        )

    def run_validation(self, aim_run: Run, current_frames: int) -> None:

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            eval_td = self.val_env.rollout(
                max_steps=self.cfg.env.episode_length,
                policy=self.policy,
                auto_cast_to_device=True,
            )
            # In run_validation, add:
            print("obs mean:", eval_td["observation"].mean().item())
            print("obs std:", eval_td["observation"].std().item())
            print(
                "obs min/max:",
                eval_td["observation"].min().item(),
                eval_td["observation"].max().item(),
            )

        weights = eval_td["current_weights"].cpu().numpy()
        portfolio_values = eval_td["next", "portfolio_value"].squeeze().cpu().numpy()

        self._log_plotly_weights(weights, portfolio_values, aim_run)

        portfolio_value = eval_td["next", "portfolio_value"][-1].item()

        total_reward = eval_td["next", "reward"].sum().item()
        aim_run.track({"val/total_reward": total_reward}, step=current_frames)
        aim_run.track({"val/portfolio_value": portfolio_value}, step=current_frames)

    def _log_metrics(
        self, loss_vals: TensorDict, aim_run: Run, current_frames: int, pbar: tqdm
    ) -> None:
        loss_vals_dict = {k: v.item() for k, v in loss_vals.items()}
        aim_run.track(
            loss_vals_dict,
            step=current_frames,
        )

        pbar.set_postfix({k: f"{v:.2f}" for k, v in loss_vals_dict.items()})

    def _optimize_batch(self, current_batch_frames: int) -> TensorDict:
        loss_vals = TensorDict()
        num_updates = int(current_batch_frames * self.utd_ratio)

        opt_pbar = tqdm(
            total=num_updates, desc="  └─ OPT", leave=False, unit="grad_step"
        )

        for _ in range(num_updates):
            sample = self.replay_buffer.sample(256)
            sample = (
                sample.to_tensordict().contiguous().to(self.device, non_blocking=True)
            )

            loss_vals = self.sac_loss(sample)

            self.critic_optimizer.zero_grad(set_to_none=True)
            loss_vals["loss_qvalue"].backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.q_module.parameters(), 1.0)
            self.critic_optimizer.step()

            # Actor
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss_vals["loss_actor"].backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.actor_optimizer.step()

            # Alpha
            self.alpha_optimizer.zero_grad(set_to_none=True)
            loss_vals["loss_alpha"].backward()
            self.alpha_optimizer.step()

            opt_pbar.update(1)

        opt_pbar.close()
        self.target_updater.step()
        return loss_vals

    def train(self, aim_run: Run) -> None:
        current_frames = 0
        pbar = tqdm(total=self.total_frames, desc="TRN")
        val_interval = 252 * 1  # TODO: make configurable

        for batch in self.collector:
            current_batch_frames = batch.numel()
            current_frames += current_batch_frames
            pbar.update(current_batch_frames)

            self.replay_buffer.extend(batch.reshape(-1))

            if len(self.replay_buffer) < self.warmup:
                pbar.set_postfix(
                    {"status": f"Warmup {len(self.replay_buffer)}/{self.warmup}"}
                )
                continue

            if (current_frames // val_interval) > (
                (current_frames - current_batch_frames) // val_interval
            ):
                pbar.set_description("VAL")
                self.run_validation(aim_run, current_frames)
                pbar.set_description("TRN")

            loss_vals = self._optimize_batch(current_batch_frames)
            self._log_metrics(loss_vals, aim_run, current_frames, pbar)

        pbar.close()
