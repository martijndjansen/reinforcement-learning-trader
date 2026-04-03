import yfinance as yf
import numpy as np


class StockDataLoader:
    def __init__(self, tickers, period):
        self.tickers = tickers
        self.period = period
        self.data = None
        self.data_rl = None
        self.prices = None
        self.num_features = None
        self.num_stocks = None

    def load(self):
        self.data = yf.download(self.tickers, period=self.period, auto_adjust=True)

        if self.data is None:
            raise ValueError("Failed to download data.")

        stocks: list[str] = self.data["Close"].columns.tolist()

        for stock in stocks:
            self.data["Return", stock] = np.log(
                self.data["Close", stock] / self.data["Close", stock].shift(1)
            ).fillna(0)  # type: ignore
        self.data = self.data.sort_index(axis=1, level=1)

        self.num_features = len(self.data.columns.get_level_values(0).unique())
        self.num_stocks = len(self.tickers)
        self.prices = (
            self.data.xs("Close", level=0, axis=1).to_numpy().astype(np.float32)
        )

        self.data.columns = [f"{feat}_{stock}" for feat, stock in self.data.columns]

        self.data_rl = self.data.to_numpy().astype(np.float32)

        return self.data, self.data_rl, self.num_features, self.num_stocks, self.prices
