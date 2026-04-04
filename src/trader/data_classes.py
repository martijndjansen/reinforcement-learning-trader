from dataclasses import dataclass

import numpy as np


@dataclass
class EnvData:
    data: np.ndarray
    prices: np.ndarray
    num_features: int
    num_stocks: int

    def __post_init__(self) -> None:
        self.data = np.array(self.data, dtype=np.float32)
        self.prices = np.array(self.prices, dtype=np.float32)
