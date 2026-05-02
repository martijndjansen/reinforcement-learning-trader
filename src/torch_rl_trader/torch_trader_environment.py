from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from trader.config import EnvConfig
from trader.data_classes import EnvData


class TorchTraderEnvironment(EnvBase):
    def __init__(
        self,
        env_data: EnvData,
        config: EnvConfig,
        device: str,
        batch_size: int = None,
    ) -> None:
        if batch_size is None:
            batch_size = torch.Size([])
        else:
            batch_size = torch.Size([batch_size])
        super().__init__(device=device, batch_size=torch.Size(batch_size))

        self.data = torch.as_tensor(env_data.data, device=device, dtype=torch.float32)
        self.prices = torch.as_tensor(
            env_data.prices, device=device, dtype=torch.float32
        )

        returns = self.prices[1:] / (self.prices[:-1] + 1e-8)
        self.step_returns = torch.cat(
            [returns, torch.ones(returns.shape[0], 1, device=device)], dim=-1
        )

        self.num_stocks = env_data.num_stocks
        self.num_assets = env_data.num_stocks + 1
        self.history_length = config.history_length
        self.episode_length = config.episode_length
        self.fee_rate = config.fee_rate
        self.alpha = config.alpha

        self.observation_spec = Composite(
            observation=Unbounded(shape=self._get_obs_size(env_data)),
            current_weights=Bounded(low=0, high=1, shape=self.num_assets),
            current_step=Unbounded(shape=torch.Size([]), dtype=torch.int32),
            portfolio_value=Unbounded(shape=1),
            elapsed_steps=Unbounded(shape=torch.Size([]), dtype=torch.int32),
        ).to(device)

        self.action_spec = Bounded(low=-10.0, high=10.0, shape=self.num_assets).to(
            device
        )
        self.reward_spec = Unbounded(shape=1).to(device)
        self.done_spec = Composite(
            done=Bounded(low=0, high=1, shape=1, dtype=torch.bool),
            terminated=Bounded(low=0, high=1, shape=1, dtype=torch.bool),
            truncated=Bounded(low=0, high=1, shape=1, dtype=torch.bool),
        ).to(device)

    def _reset(self, tensordict: TensorDict | None = None, **_: Any) -> TensorDict:

        device = tensordict.device if tensordict is not None else self.device

        if tensordict is not None and "current_step" in tensordict.keys():
            start_step = tensordict["current_step"]
        else:
            max_start = len(self.prices) - self.episode_length - 1
            start_step = torch.randint(
                self.history_length, max_start, (), device=device
            ).to(torch.int32)

        if tensordict is not None and "current_weights" in tensordict.keys():
            weights = tensordict["current_weights"]
        else:
            raw_w = torch.rand(self.num_assets, device=device) + 0.1
            weights = raw_w / raw_w.sum()

        out = TensorDict(
            {
                "current_step": start_step,
                "current_weights": weights,
                "portfolio_value": torch.tensor([1.0], device=device),
                "elapsed_steps": torch.tensor(0, device=device, dtype=torch.int32),
            },
            batch_size=self.batch_size,
            device=device,
        )

        out["observation"] = self._get_observation(out)

        return out

    def _step(self, tensordict: TensorDict) -> TensorDict:
        step = tensordict["current_step"]
        weights = tensordict["current_weights"]
        val = tensordict["portfolio_value"]
        elapsed = tensordict["elapsed_steps"]

        target_weights = torch.softmax(tensordict["action"], dim=-1)

        cost = torch.abs(target_weights - weights).sum() * self.fee_rate
        r = self.step_returns[step]
        growth = (target_weights * r).sum()

        new_val = val * (1.0 - cost) * growth
        net_return = (new_val - val) / (val + 1e-8)
        reward = (1 - torch.exp(-self.alpha * net_return)) / self.alpha

        new_weights = (target_weights * r) / (growth + 1e-8)
        new_step = step + 1
        new_elapsed = elapsed + 1

        terminated = new_step >= len(self.data) - 1
        truncated = new_elapsed >= self.episode_length
        done = terminated | truncated

        out = TensorDict(
            {
                "current_step": new_step,
                "current_weights": new_weights,
                "portfolio_value": new_val,
                "elapsed_steps": new_elapsed,
                "reward": reward,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        out["observation"] = self._get_observation(out)
        return out

    def _get_observation(self, td: TensorDict) -> Tensor:
        step = td["current_step"]
        window = self.data[step - self.history_length : step]
        return torch.cat([window.reshape(-1), td["current_weights"]])

    def _get_obs_size(self, env_data: EnvData) -> int:
        return (
            env_data.num_features * self.num_stocks + env_data.global_features
        ) * self.history_length + self.num_assets

    def _set_seed(self, seed: int | None) -> None:
        return super()._set_seed(seed)
