from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from .data_classes import EnvData


class TraderEnvironment(gym.Env):
    def __init__(
        self,
        env_data: EnvData,
        budget: np.float32,
        episode_length: int,
        history_length: int,
        fee_rate: np.float32,
        lmb: np.float32,
    ) -> None:

        self.num_stocks = env_data.num_stocks
        self.data = env_data.data
        self.prices = env_data.prices

        self.step_returns = np.pad(
            self.prices[1:] / (self.prices[:-1] + 1e-8),
            ((0, 0), (0, 1)),
            constant_values=1.0,
        )

        self.num_current_weights = env_data.num_stocks + 1
        self.budget = budget
        self.fee_rate = fee_rate
        self.episode_length = episode_length
        self.history_length = history_length
        self.lmb = lmb

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

        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.num_current_weights,), dtype=np.float32
        )

        self.reset()

    def _get_state(self) -> np.ndarray:

        window_data = self.data[
            self.current_step - self.history_length + 1 : self.current_step + 1
        ]
        state = np.concatenate([window_data.flatten(), self.current_weights])
        return state.astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, dict]:
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
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        cost = np.sum(np.abs(action - self.current_weights)) * self.fee_rate

        growth_factor = action @ self.step_returns[self.current_step]
        new_portfolio_value = self.portfolio_value * (1.0 - cost) * growth_factor
        reward = np.log(new_portfolio_value / self.portfolio_value) - cost * self.lmb

        self.portfolio_value = new_portfolio_value
        self.current_weights = action
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
