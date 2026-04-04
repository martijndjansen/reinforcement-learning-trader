from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from .data_classes import EnvData


class TraderEnvironment(gym.Env):
    def __init__(
        self,
        env_data: EnvData,
        budget: np.float32,
    ) -> None:

        self.num_stocks = env_data.num_stocks
        self.data = env_data.data

        self.prices = env_data.prices

        self.num_current_weights = env_data.num_stocks + 1

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(
                env_data.num_features * env_data.num_stocks + self.num_current_weights,
            ),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.num_current_weights,), dtype=np.float32
        )
        self.budget = budget

        self.max_step_size = 0.2

        self.reset()

    def _get_state(self) -> np.ndarray:
        current_features = self.data[self.current_step].flatten()
        state = np.concatenate([current_features, self.current_weights])
        return state.astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.current_weights = np.zeros(self.num_current_weights, dtype=np.float32)

        self.current_weights[-1] = 1.0

        self.current_shares = np.zeros(self.num_stocks, dtype=np.float32)

        self.current_budget = self.budget

        self.portfolio_value = self.budget

        return self._get_state(), {}

    def step(
        self, action: np.float32
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        current_prices: np.float32 = self.prices[self.current_step]

        self.current_weights = np.clip(
            self.current_weights + self.max_step_size * action, 0, 1
        )
        total_weight = np.sum(self.current_weights) + 1e-8
        self.current_weights = self.current_weights / total_weight

        stock_weights = self.current_weights[:-1]
        cash_weight = self.current_weights[-1]

        self.current_step += 1

        next_prices = self.prices[self.current_step]

        price_returns = next_prices / (current_prices + 1e-8)

        portfolio_return = stock_weights @ price_returns + cash_weight

        new_portfolio_value = portfolio_return * self.portfolio_value

        reward = new_portfolio_value - self.portfolio_value

        self.portfolio_value = new_portfolio_value

        observation = self._get_state()
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info
