import numpy as np
import gymnasium as gym
import pandas as pd


class GridWorldEnv(gym.Env):
    def __init__(
        self, num_features: int, num_stocks: int, data: pd.DataFrame, budget: int
    ):

        self.num_stocks = num_stocks
        self.data = data
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features + 1, num_stocks)
        )

        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(num_stocks,), dtype=np.float32
        )
        self.budget = budget
        self.current_budget = budget

    def _get_state(self):
        current_features = self.data.iloc[self.current_step].values.T
        weights_row = self.current_weights.reshape(1, -1)
        return np.vstack([current_features, weights_row]).astype(np.float32)

    def reset(self):
        self.current_step = 0
        self.current_weights = np.zeros(self.num_stocks)
        return self._get_state()

    def step(self, action):
        current_prices = self.data.loc[self.current_step, "Close"].values
        full_budget = self.current_budget + np.sum(
            self.current_weights * current_prices
        )

        investment_per_stock = action * full_budget

        stocks = np.floor(investment_per_stock / current_prices).astype(int)

        new_budget = full_budget - np.sum(stocks * current_prices)
        reward = new_budget + np.sum(stocks * current_prices) - full_budget

        self.current_step += 1
        self.current_budget = new_budget
        self.current_weights = stocks
        observation = self._get_state()
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info
