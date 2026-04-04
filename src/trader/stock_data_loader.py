import numpy as np
import yfinance as yf

from .data_classes import EnvData


class StockDataLoader:
    def __init__(self, tickers: list[str], period: str) -> None:
        self.tickers = tickers
        self.period = period
        self.data = None
        self.data_rl = None
        self.prices = None
        self.num_features = None
        self.num_stocks = None

    def load(self) -> EnvData:
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

        return EnvData(
            data=self.data_rl,
            prices=self.prices,
            num_features=self.num_features,
            num_stocks=self.num_stocks,
        )
