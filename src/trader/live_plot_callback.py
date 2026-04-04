import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from stable_baselines3.common.callbacks import BaseCallback

from .environment import TraderEnvironment


class LivePlotCallback(BaseCallback):
    def __init__(
        self,
        trader_environment: TraderEnvironment,
        tickers: list[str],
        total_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.trader_environment = trader_environment
        self.rewards = []
        self.steps = []
        self.total_timesteps = total_timesteps
        self.tickers = tickers

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.rewards.append(np.mean(rewards))
            self.steps.append(self.num_timesteps)

        if len(self.rewards) % 50 == 0:
            clear_output(wait=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            ax1.plot(self.steps, self.rewards)
            ax1.set_xlabel("Training Step")
            ax1.set_ylabel("Mean Reward")
            ax1.set_title(f"Progress ({self.num_timesteps}/{self.total_timesteps})")

            weights = self.trader_environment.current_weights
            tickers = [*self.tickers, "cash"]
            ax2.bar(tickers, weights)
            ax2.set_title("Current Portfolio Weights")

            plt.tight_layout()
            plt.show()

        return True
