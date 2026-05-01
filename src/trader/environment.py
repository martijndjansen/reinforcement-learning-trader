from typing import Any

import gymnasium as gym
import numpy as np

from .config import EnvConfig
from .data_classes import EnvData


class TraderEnvironment(gym.Env):
    def __init__(self, env_data: EnvData, config: EnvConfig) -> None:

        self.num_stocks = env_data.num_stocks
        self.data = env_data.data
        self.prices = env_data.prices

        self.step_returns = np.pad(
            self.prices[1:] / (self.prices[:-1] + 1e-8),
            ((0, 0), (0, 1)),
            constant_values=1.0,
        )

        self.num_current_weights = env_data.num_stocks + 1
        self.budget = config.budget
        self.fee_rate = config.fee_rate
        self.episode_length = config.episode_length
        self.history_length = config.history_length
        self.distribution = config.distribution
        self.alpha = config.alpha

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(
                (env_data.num_features * env_data.num_stocks + env_data.global_features)
                * self.history_length
                + env_data.num_stocks
                + 1,
            ),
            dtype=np.float32,
        )

        limits = (-np.inf, np.inf)
        if self.distribution == "dirichlet":
            limits = (0, 1)
        elif self.distribution == "gaussian":
            limits = (-10, 10)
        else:
            raise ValueError(f"Distribution `{self.distribution}` not implemented")

        self.action_space = gym.spaces.Box(
            low=limits[0],
            high=limits[1],
            shape=(self.num_current_weights,),
            dtype=np.float32,
        )
        self.reset()

    def _get_state(self) -> np.ndarray:

        window_data = self.data[
            self.current_step - self.history_length : self.current_step
        ]
        state = np.concatenate([window_data.flatten(), self.current_weights])
        return state.astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        raw_weights = self.np_random.uniform(0.1, 1, self.num_current_weights)
        self.current_weights = raw_weights / np.sum(raw_weights)

        max_start = len(self.prices) - self.episode_length - 1
        self.current_step = self.np_random.integers(self.history_length, max_start)

        self.portfolio_value = self.budget
        self._elapsed_steps = 0

        return self._get_state(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.distribution == "gaussian":
            shift_actions = action - np.max(action)
            action = np.exp(shift_actions) / np.sum(np.exp(shift_actions))
        cost = np.sum(np.abs(action - self.current_weights)) * self.fee_rate
        r = self.step_returns[self.current_step]

        growth_factor = action @ r
        new_portfolio_value = self.portfolio_value * (1.0 - cost) * growth_factor

        net_return = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        reward = (1 - np.exp(-self.alpha * net_return)) / self.alpha

        self.current_weights = (action * r) / (growth_factor + 1e-8)
        self.portfolio_value = new_portfolio_value
        self.current_step += 1
        self._elapsed_steps += 1

        info = {}

        observation = self._get_state()
        terminated = bool(self.current_step >= len(self.data) - 1)
        truncated = self._elapsed_steps >= self.episode_length
        if terminated or truncated:
            info["final_info"] = {
                "portfolio_value": self.portfolio_value,
                "steps": self._elapsed_steps,
            }

        return observation, reward, terminated, truncated, info
