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
        trader_environment: TraderEnvironment,
        tickers: list[str],
        total_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.trader_env = trader_environment
        self.tickers = [*tickers, "CASH"]
        self.total_timesteps = total_timesteps

        self.episode_final_values = []
        self._reset()

    def _reset(self) -> None:
        self.current_episode_value = []
        self.current_episode_weights = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"][0]
        done = self.locals["dones"][0]

        if not done:
            self.current_episode_value.append(self.trader_env.portfolio_value)
            self.current_episode_weights.append(self.trader_env.current_weights)
        else:
            if "final_info" in infos:
                final_val = infos["final_info"]["portfolio_value"]
                self.episode_final_values.append(final_val)
                self.current_episode_value.append(final_val)

            self._reset()

        return True

    def _plot_dashboard(self, terminated: bool = False) -> None:
        clear_output(wait=True)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Final episode summed stock value",
                "Weights per stock",
                "Episode: weight allocation over time",
                "Episode: summed stock value",
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # Figure 1: final episode summed stock value
        fig.add_trace(
            go.Scatter(
                y=self.episode_final_values,
                mode="lines+markers",
                name="Summed stock value",
                line={"color": "green"},
            ),
            row=1,
            col=1,
        )

        fig.add_hline(
            y=self.trader_env.budget,
            line_dash="dash",
            line_color="red",
            row=1,  # type: ignore
            col=1,  # type: ignore
        )

        # Figure 2: weights per stock
        weights_history = np.array(self.current_episode_weights)
        fig.add_trace(
            go.Bar(
                x=self.tickers,
                y=weights_history[-1],
                marker_color="royalblue",
                name="Weights",
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(range=[0, 1], row=1, col=2)

        # Figure 3: weights per stock over time
        steps = list(range(len(weights_history)))

        for i, ticker in enumerate(self.tickers):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=weights_history[:, i],
                    mode="lines",
                    stackgroup="one",
                    name=ticker,
                    hoverinfo="x+y+name",
                ),
                row=2,
                col=1,
            )
        fig.update_yaxes(range=[0, 1], row=2, col=1)

        # Figure 4: episode: summed stock value
        fig.add_trace(
            go.Scatter(
                y=self.current_episode_value,
                mode="lines",
                name="Equity",
                line={"color": "purple"},
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800,
            width=1200,
            template="plotly_dark",
            showlegend=True,
            title_text=f"Reinforcement Learning Trader - Timestep {self.num_timesteps}",
            title_x=0.5,
        )

        fig.show()

        if terminated:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

            val = f"{self.trader_env.portfolio_value / 1000:.1f}k"

            fig.write_image(f"runs/{timestamp}_step-{self.num_timesteps}_val-{val}.svg")

    def _on_training_end(self) -> None:

        self._plot_dashboard(terminated=True)
