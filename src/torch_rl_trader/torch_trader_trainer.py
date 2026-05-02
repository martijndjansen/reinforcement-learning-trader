import torch
from aim import Run
from tensordict.nn import (
    ProbabilisticTensorDictSequential,
)
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import ValueOperator
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from tqdm import tqdm

from torch_rl_trader.torch_trader_environment import TorchTraderEnvironment
from trader.config import Config

torch.manual_seed(0)


class TraderTrainer:
    def __init__(
        self,
        cfg: Config,
        train_env: TorchTraderEnvironment,
        policy: ProbabilisticTensorDictSequential,
        q_module: ValueOperator,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.model.device)
        # implement env = TransformedEnv(base_env)

        self.create_env_fn = lambda: train_env
        self.action_spec = train_env.action_spec
        self.utd_ratio = self.cfg.model.utd_ratio

        self.warmup = 100  # TODO: make configurable
        self.total_frames = cfg.trading.num_episodes * cfg.env.episode_length

        self.policy = policy.to(self.device)
        self.q_module = q_module.to(self.device)
        self.sac_loss = self._init_loss()
        self.collector = self._init_collector()
        self.replay_buffer = self._init_replay_buffer()
        self.optimizer = torch.optim.Adam(self.sac_loss.parameters(), lr=3e-4)
        self.target_updater = SoftUpdate(self.sac_loss, eps=0.995)

    def _init_loss(self) -> SACLoss:
        sac_loss = SACLoss(
            actor_network=self.policy,
            qvalue_network=self.q_module,
            num_qvalue_nets=2,
            delay_actor=True,
            delay_qvalue=True,
            action_spec=self.action_spec,
        )
        sac_loss.make_value_estimator(gamma=0.9)
        return sac_loss

    def _init_collector(self) -> Collector:
        return Collector(
            create_env_fn=self.create_env_fn,  # type: ignore
            policy=self.policy,
            frames_per_batch=self.cfg.env.episode_length,
            total_frames=self.total_frames,
            device=self.device,
            storing_device=self.device,
        )

    def _init_replay_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(storage=LazyTensorStorage(100_000, device=self.device))

    def _log_metrics(
        self, loss_vals: dict, aim_run: Run, current_frames: int, pbar: tqdm
    ) -> None:

        aim_run.track(
            {k: v.item() for k, v in loss_vals.items()},
            step=current_frames,
        )

        pbar.set_postfix(
            {
                "Actor": f"{loss_vals.get('loss_actor', 0):.3f}",
                "Q": f"{loss_vals.get('loss_qvalue', 0):.3f}",
                "Alpha": f"{loss_vals.get('loss_alpha', 0):.4f}",
                "Buf": len(self.replay_buffer),
            }
        )

    def _optimize_batch(self) -> dict:
        loss_vals = {}

        for _ in range(self.utd_ratio):
            sample = self.replay_buffer.sample(2048)
            loss_vals = self.sac_loss(sample)

            loss_total = (
                loss_vals["loss_actor"]
                + loss_vals["loss_qvalue"]
                + loss_vals["loss_alpha"]
            )

            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            self.target_updater.step()
        return loss_vals

    def train(self, aim_run: Run) -> None:
        current_frames = 0
        pbar = tqdm(total=self.total_frames, desc="Training Trader")

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

            loss_vals = self._optimize_batch()
            self._log_metrics(loss_vals, aim_run, current_frames, pbar)

        pbar.close()
