import time

import numpy as np
import plotly.graph_objects as go
from aim import Figure, Run
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from .environment import TraderEnvironment


class LivePlotCallback(BaseCallback):
    def __init__(
        self,
        train_env: TraderEnvironment,
        val_env: TraderEnvironment,
        tickers: list[str],
        total_timesteps: int,
        episode_length: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.train_env = train_env
        self.val_env = val_env
        self.tickers = [*tickers, "CASH"]
        self.total_timesteps = total_timesteps

        self.aim_run = Run(repo=".", experiment="rl_trading")

        self.episode_final_values = []
        self.val_history = []

        self.val_current_weights = []
        self.val_current_values = []
        self.episode_length = episode_length
        self.progress_bar = tqdm(
            total=int(total_timesteps / episode_length), desc="Episodes", position=0
        )
        self._reset()

    def _reset(self) -> None:
        self.current_episode_value = []
        self.current_episode_weights = []
        self.episode_start_time = time.time()
        self.step_progress_bar = tqdm(
            total=int(self.episode_length), desc="Steps", position=1, leave=False
        )

    def _run_validation_episode(self) -> None:

        self.val_current_weights = []
        self.val_current_values = []

        obs, _ = self.val_env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.val_env.step(action)

            self.val_current_values.append(self.val_env.portfolio_value)
            self.val_current_weights.append(self.val_env.current_weights)

            done = terminated or truncated

        val_final_value = self.val_env.portfolio_value
        self.val_history.append(val_final_value)
        self.aim_run.track(
            {"val/episode_portfolio_value": val_final_value},
            step=self.num_timesteps,
            context={"env": "train", "chart": "valuation"},
        )

    def _on_step(self) -> bool:
        for key, value in self.logger.name_to_value.items():
            if isinstance(value, (int, float, np.number)):
                self.aim_run.track(
                    value, name=key, context={"env": "train", "chart": "algorithm"}
                )

        infos = self.locals["infos"][0]
        done = self.locals["dones"][0]

        self.aim_run.track(
            {"train/portfolio_value": self.train_env.portfolio_value},
            step=self.num_timesteps,
        )

        self.step_progress_bar.update(1)

        if not done:
            self.current_episode_value.append(self.train_env.portfolio_value)
            self.current_episode_weights.append(self.train_env.current_weights)
        else:
            if "final_info" in infos:
                self.progress_bar.update(1)
                final_val = infos["final_info"]["portfolio_value"]
                self.episode_final_values.append(final_val)
                self.current_episode_value.append(final_val)
                self.aim_run.track(
                    {
                        "train/episode_portfolio_value": final_val,
                        "train/episode_time_sec": time.time() - self.episode_start_time,
                    },
                    step=self.num_timesteps,
                    context={"env": "train", "chart": "valuation"},
                )
            if self.num_timesteps % (self.episode_length * 3) == 0:
                self._run_validation_episode()
                self._plot_dashboard()
            self._reset()

        return True

    def _plot_dashboard(self) -> None:
        fig_1 = go.Figure()

        episode_num = int(self.num_timesteps / self.episode_length)

        for name, y, weights_hist in [
            ("Train", self.episode_final_values, self.current_episode_weights),
            ("Val", self.val_history, self.val_current_weights),
        ]:
            fig_1.add_scatter(
                y=y,
                mode="lines+markers",
                name=f"{name} End Val",
            )
            fig_weights_over_time = go.Figure()

            if len(weights_hist) > 0:
                w_array = np.array(weights_hist)
                for i, ticker in enumerate(self.tickers):
                    fig_weights_over_time.add_scatter(
                        y=list(w_array[:, i]),
                        mode="lines",
                        hoverinfo="x+y",
                        name=ticker,
                        stackgroup="one",
                    )
                self.aim_run.track(
                    Figure(fig_weights_over_time),
                    name=f"{name} - weights over time",
                    step=episode_num,
                    context={"type": "plotly"},
                )

        fig_1.add_hline(
            y=self.train_env.budget,
            line_dash="dash",
            line_color="red",
        )

        self.aim_run.track(
            Figure(fig_1),
            name="Final Value",
            step=episode_num,
            context={"type": "plotly"},
        )

    def _on_training_end(self) -> None:
        self.progress_bar.close()
