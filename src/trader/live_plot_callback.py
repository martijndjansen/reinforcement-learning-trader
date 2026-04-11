from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from IPython.display import clear_output
from plotly.subplots import make_subplots
from stable_baselines3.common.callbacks import BaseCallback

from .environment import TraderEnvironment


class LivePlotCallback(BaseCallback):
    def __init__(
        self,
        train_env: TraderEnvironment,
        val_env: TraderEnvironment,
        tickers: list[str],
        total_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.train_env = train_env
        self.val_env = val_env
        self.tickers = [*tickers, "CASH"]
        self.total_timesteps = total_timesteps

        self.episode_final_values = []
        self.val_history = []

        self.val_current_weights = []
        self.val_current_values = []
        self._reset()

    def _reset(self) -> None:
        self.current_episode_value = []
        self.current_episode_weights = []

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
        self.val_history.append((self.num_timesteps, val_final_value))

    def _on_step(self) -> bool:

        infos = self.locals["infos"][0]
        done = self.locals["dones"][0]

        if not done:
            self.current_episode_value.append(self.train_env.portfolio_value)
            self.current_episode_weights.append(self.train_env.current_weights)
        else:
            if "final_info" in infos:
                final_val = infos["final_info"]["portfolio_value"]
                self.episode_final_values.append(final_val)
                self.current_episode_value.append(final_val)

            self._run_validation_episode()
            self._plot_dashboard()
            self._reset()

        return True

    def _plot_dashboard(self, terminated: bool = False) -> None:
        clear_output(wait=True)

        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=(
                "Train: Final Episode Values",
                "Train: Current Weights",
                "Train: Weights Over Time",
                "Train: Current Equity",
                "Val: Learning Curve (Final Vals)",
                "Val: Current Weights",
                "Val: Weights Over Time",
                "Val: Current Equity",
            ),
            vertical_spacing=0.08,
        )

        def add_env_plots(
            row_start: int,
            final_vals: list[float],
            weights_hist: list[float],
            equity_hist: list[float],
            prefix: str,
        ) -> None:
            # Figure 1: final episode summed stock value
            y_vals = (
                final_vals if prefix == "Train" else [v[1] for v in self.val_history]
            )
            x_vals = None if prefix == "Train" else [v[0] for v in self.val_history]

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=f"{prefix} End Val",
                    line_color="green",
                ),
                row=row_start,
                col=1,
            )
            fig.add_hline(
                y=self.train_env.budget,
                line_dash="dash",
                line_color="red",
                row=row_start,  # type:ignore
                col=1,  # type:ignore
            )

            # Figure 2: weights per stock
            if len(weights_hist) > 0:
                fig.add_trace(
                    go.Bar(
                        x=self.tickers,
                        y=weights_hist[-1],
                        name=f"{prefix} Weights",
                        marker_color="royalblue",
                    ),
                    row=row_start,
                    col=2,
                )
                fig.update_yaxes(range=[0, 1], row=row_start, col=2)

            # Figure 3: weights per stock over time
            if len(weights_hist) > 0:
                w_array = np.array(weights_hist)
                for i, ticker in enumerate(self.tickers):
                    fig.add_trace(
                        go.Scatter(
                            y=w_array[:, i],
                            mode="lines",
                            stackgroup="one",
                            name=ticker,
                            showlegend=(prefix == "Train"),
                        ),
                        row=row_start + 1,
                        col=1,
                    )
                fig.update_yaxes(range=[0, 1], row=row_start + 1, col=1)

            # Figure 4: episode: summed stock value
            fig.add_trace(
                go.Scatter(
                    y=equity_hist,
                    mode="lines",
                    name=f"{prefix} Equity",
                    line_color="purple",
                ),
                row=row_start + 1,
                col=2,
            )

        add_env_plots(
            1,
            self.episode_final_values,
            self.current_episode_weights,
            self.current_episode_value,
            "Train",
        )
        add_env_plots(
            3,
            self.val_history,
            self.val_current_weights,
            self.val_current_values,
            "Val",
        )

        fig.update_layout(
            height=1200,
            width=1200,
            template="plotly_dark",
            title_text=f"Reinforcement Learning Trader - Timestep {self.num_timesteps}",
            title_x=0.5,
        )
        fig.show()

        if terminated:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

            val = f"{self.train_env.portfolio_value / 1000:.1f}k"

            fig.write_image(f"runs/{timestamp}_step-{self.num_timesteps}_val-{val}.svg")

    def _on_training_end(self) -> None:

        self._plot_dashboard(terminated=True)
