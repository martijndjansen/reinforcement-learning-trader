from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class LivePlotCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.steps = []
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.rewards.append(np.mean(rewards))
            self.steps.append(self.num_timesteps)

        if len(self.rewards) % 50 == 0:
            clear_output(wait=True)
            plt.figure(figsize=(8, 4))
            plt.plot(self.steps, self.rewards)
            plt.xlabel("Training Step")
            plt.ylabel("Mean Reward")
            plt.title(
                f"Training Progress ({self.num_timesteps}/{self.total_timesteps})"
            )
            plt.show()

        return True
