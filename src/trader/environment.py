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

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(env_data.num_features * env_data.num_stocks + env_data.num_stocks,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(env_data.num_stocks,), dtype=np.float32
        )
        self.budget = budget
        self.current_budget = budget

        self.reset()

    def _get_state(self) -> np.ndarray:
        current_features = self.data[self.current_step].flatten()
        state = np.concatenate([current_features, self.current_shares])
        return state.astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.current_shares = np.zeros(self.num_stocks, dtype=np.float32)
        return self._get_state(), {}

    def step(
        self, action: np.float32
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_prices: np.float32 = self.prices[self.current_step]
        portfolio_value = self.current_budget + self.current_shares @ current_prices

        action = action / (action.sum() + 1e-8)
        investment_per_stock = action * portfolio_value
        stocks = np.floor(investment_per_stock / current_prices).astype(int)

        # calculate transaction cost
        shares_traded = np.abs(stocks - self.current_shares)
        cost_per_trade = 0.001
        transaction_cost = shares_traded @ current_prices * cost_per_trade

        new_budget = portfolio_value - stocks @ current_prices - transaction_cost

        self.current_step += 1
        self.current_budget = new_budget
        self.current_shares = stocks

        next_prices = self.prices[self.current_step]

        new_portfolio_value = new_budget + stocks @ next_prices

        reward = new_portfolio_value - portfolio_value - transaction_cost

        observation = self._get_state()
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info
