from typing import Callable, Tuple, cast

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from .data_classes import EnvData


class StockDataLoader:
    def __init__(
        self, tickers: list[str], macro_tickers: list[str], period: str
    ) -> None:
        self.tickers = tickers
        self.macro_tickers = macro_tickers
        self.period = period
        self.data = None
        self.data_rl = None
        self.prices = None
        self.num_stocks = None

    @staticmethod
    def cyclical(f: Callable, index: int, max: int) -> np.float32:
        return f(2 * np.pi * index / max)

    def load(self) -> Tuple[pd.DataFrame, EnvData, EnvData]:
        self.data = yf.download(
            self.tickers + self.macro_tickers, period=self.period, auto_adjust=True
        )

        if self.data is None:
            raise ValueError("Failed to download data.")

        stocks: list[str] = self.tickers

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

        # global features
        self.data[("Global", "Stock_Volatility")] = self.data["Volatility"].mean(axis=1)  # type: ignore
        self.data[("Global", "Fear_VIX")] = self.data["Return", "^VIX"]
        self.data[("Global", "Rates_TNX")] = self.data["Return", "^TNX"]
        self.data[("Global", "Safety_GLD")] = self.data["Return", "GLD"]
        self.data[("Global", "Regime_XLK_XLP")] = (
            self.data["Return", "XLK"] - self.data["Return", "XLP"]
        )

        self.data = self.data.iloc[20:].copy()
        self.data = self.data.sort_index(axis=1, level=1)

        self.num_stocks = len(self.tickers)
        self.prices = (
            self.data.xs("Close", level=0, axis=1)[stocks].to_numpy().astype(np.float32)
        )

        self.data = self.data.drop(
            labels=list(set(self.macro_tickers) - set(self.tickers)), level=1, axis=1
        )
        self.data = self.data.drop(["Open", "Close", "High", "Low", "Volume"], axis=1)
        self.data.columns = [f"{feat}_{stock}" for feat, stock in self.data.columns]

        self.data = self.data.reset_index()

        self.data["Month"] = self.data["Date"].dt.month
        self.data["DayOfWeek"] = self.data["Date"].dt.dayofweek

        for type, max in [("Month", 12), ("DayOfWeek", 5)]:
            for name, function in [("Sin", np.sin), ("Cos", np.cos)]:
                self.data[f"{type}_{name}"] = self.cyclical(
                    function, cast(int, self.data[type]), max
                )

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

        return (
            self.data,
            EnvData(
                data=self.data_rl[:split_idx],
                prices=self.prices[:split_idx],
                num_features=num_features_per_stock,
                num_stocks=self.num_stocks,
                global_features=len(global_cols) + len(time_cols),
            ),
            EnvData(
                data=self.data_rl[split_idx:],
                prices=self.prices[split_idx:],
                num_features=num_features_per_stock,
                num_stocks=self.num_stocks,
                global_features=len(global_cols) + len(time_cols),
            ),
        )
