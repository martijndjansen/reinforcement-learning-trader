import os
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_classes import EnvData


class StockDataLoader:
    REQUIRED_MACRO_TICKERS = ["^VIX", "^TNX", "GLD", "XLK", "XLP"]

    def __init__(self, tickers: list[str], period: str) -> None:
        self.tickers = tickers
        self.macro_tickers = self.REQUIRED_MACRO_TICKERS
        self.period = period
        self.data = None
        self.data_rl = None
        self.prices = None
        self.num_stocks = None
        self.filepath = "data/stock_data.csv"
        self.pricepath = "data/stock_prices.npy"

    @staticmethod
    def cyclical(f: Callable, index: int, max: int) -> np.float32:
        return f(2 * np.pi * index / max)

    def _add_stock_features(self) -> None:
        stocks: list[str] = self.tickers

        if self.data is None:
            raise ValueError("Failed to download data.")

        # stock features
        for stock in stocks + self.macro_tickers:
            self.data["Return", stock] = np.log(
                self.data["Close", stock] / self.data["Close", stock].shift(1)
            ).fillna(0)  # type: ignore

        market_mean = self.data["Return"][stocks].astype(np.float32).mean(axis=1)  # type: ignore
        for stock in stocks:
            self.data["Relative", stock] = self.data["Return", stock] - market_mean
            self.data["Volatility", stock] = (
                self.data["Return", stock].rolling(window=20).std()
            )
            self.data["Momentum", stock] = (
                self.data["Close", stock]
                / (self.data["Close", stock].rolling(window=20).mean() + 1e-8)
            ) - 1
            raw_vol = self.data["Volume", stock]
            self.data["LogVolume", stock] = np.log(
                (raw_vol / (raw_vol.shift(1) + 1e-8)).clip(lower=1e-8)
            )

    def _add_global_features(self) -> None:
        if self.data is None:
            raise ValueError("Failed to download data.")

        r = self.data["Return"]

        global_features = {
            ("Global", "Stock_Volatility"): self.data["Volatility"]
            .mean(axis=1)  # type: ignore
            .to_numpy(),
            ("Global", "Fear_VIX"): r["^VIX"].to_numpy(),
            ("Global", "Rates_TNX"): r["^TNX"].to_numpy(),
            ("Global", "Safety_GLD"): r["GLD"].to_numpy(),
            ("Global", "Regime_XLK_XLP"): (r["XLK"] - r["XLP"]).to_numpy(),
        }

        self.data = pd.concat(
            [self.data, pd.DataFrame(global_features, index=self.data.index)], axis=1
        )

    def _flatten_data(self) -> None:
        if self.data is None:
            raise ValueError("Failed to download data.")

        self.data = self.data.fillna(0)
        self.data = self.data.sort_index(axis=1)
        self.data = self.data.drop(
            labels=list(set(self.macro_tickers) - set(self.tickers)), level=1, axis=1
        )
        self.data = self.data.drop(["Open", "Close", "High", "Low", "Volume"], axis=1)
        self.data.columns = [f"{feat}_{stock}" for feat, stock in self.data.columns]

        self.data = self.data.reset_index()

    def _add_time_features(self) -> None:
        if self.data is None:
            raise ValueError("Failed to download data.")
        self.data["Month"] = self.data["Date"].dt.month
        self.data["DayOfWeek"] = self.data["Date"].dt.dayofweek

        for type, max in [("Month", 12), ("DayOfWeek", 5)]:
            for name, function in [("Sin", np.sin), ("Cos", np.cos)]:
                self.data[f"{type}_{name}"] = self.cyclical(
                    function, cast(int, self.data[type]), max
                )

    def _prepare_data(self) -> tuple[EnvData, EnvData]:
        if self.data is None or self.prices is None:
            raise ValueError("Failed to download data.")

        if not (os.path.exists(self.filepath) and os.path.exists(self.pricepath)):
            self.data.to_csv(self.filepath)
            np.save(self.pricepath, self.prices)

        stock_cols = [
            c
            for c in self.data.columns
            if any(s in c for s in self.tickers) and not c.startswith("Global")
        ]
        global_cols = [c for c in self.data.columns if c.startswith("Global")]
        time_cols = [c for c in self.data.columns if "Sin" in c or "Cos" in c]

        # scale data
        split_idx = int(len(self.data) * 0.8)
        cols_to_scale = stock_cols + global_cols
        scaler = StandardScaler()
        scaler.fit(self.data.loc[: split_idx - 1, cols_to_scale])
        self.data[cols_to_scale] = scaler.transform(self.data[cols_to_scale]).astype(
            np.float32
        )

        ordered_cols = ["Date"] + stock_cols + global_cols + time_cols
        self.data = self.data[ordered_cols]

        self.data_rl = self.data.drop(["Date"], axis=1).to_numpy().astype(np.float32)
        num_features_per_stock = len(stock_cols) // len(self.tickers)

        env_data = [
            EnvData(
                data=d,
                prices=p,
                num_features=num_features_per_stock,
                num_stocks=len(self.tickers),
                global_features=len(global_cols) + len(time_cols),
            )
            for d, p in [
                (self.data_rl[:split_idx], self.prices[:split_idx]),
                (self.data_rl[split_idx:], self.prices[split_idx:]),
            ]
        ]

        return env_data[0], env_data[1]

    def _download_data(self) -> pd.DataFrame | None:
        import yfinance as yf

        return yf.download(
            self.tickers + self.macro_tickers, period=self.period, auto_adjust=True
        )

    def load(self) -> tuple[EnvData, EnvData]:
        if os.path.exists(self.filepath) and os.path.exists(self.pricepath):
            self.data = pd.read_csv(self.filepath, index_col=0)
            self.prices = np.load(self.pricepath)

            return self._prepare_data()

        self.data = self._download_data()

        if self.data is None:
            raise ValueError("Failed to download data.")

        self._add_stock_features()
        self._add_global_features()

        self.prices = self.data["Close"][self.tickers].to_numpy(dtype=np.float32)

        self._flatten_data()
        self._add_time_features()

        return self._prepare_data()
