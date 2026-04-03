from typing import Any

import numpy as np
import gymnasium as gym


class TraderEnvironment(gym.Env):
    def __init__(
        self,
        num_features: int,
        num_stocks: int,
        data: np.ndarray,
        prices: np.ndarray,
        budget: int,
    ):

        self.num_stocks = num_stocks
        self.data = data

        self.prices = prices
        print(num_features)
        print(num_stocks)
        print(num_features * num_stocks + num_stocks)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features * num_stocks + num_stocks,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(num_stocks,), dtype=np.float32
        )
        self.budget = budget
        self.current_budget = budget

        self.reset()

    def _get_state(self):
        current_features = self.data[self.current_step].flatten()
        state = np.concatenate([current_features, self.current_weights])
        return state.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_weights = np.zeros(self.num_stocks, dtype=np.float32)
        return self._get_state(), {}

    def step(self, action):
        current_prices = self.prices[self.current_step]
        portfolio_value = self.current_budget + self.current_weights @ current_prices

        action = action / (action.sum() + 1e-8)
        investment_per_stock = action * portfolio_value
        stocks = np.floor(investment_per_stock / current_prices).astype(int)

        # calculate transaction cost
        shares_traded = np.abs(stocks - self.current_weights)
        cost_per_trade = 0.001
        transaction_cost = shares_traded @ current_prices * cost_per_trade

        new_budget = portfolio_value - stocks @ current_prices - transaction_cost

        self.current_step += 1
        self.current_budget = new_budget
        self.current_weights = stocks

        next_prices = self.prices[self.current_step]

        new_portfolio_value = new_budget + stocks @ next_prices

        reward = new_portfolio_value - portfolio_value

        observation = self._get_state()
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info
