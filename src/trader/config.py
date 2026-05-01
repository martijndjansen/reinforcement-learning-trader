from dataclasses import dataclass
from typing import Literal


@dataclass
class EnvConfig:
    """Hyperparameters for the Trading Environment."""

    budget: float
    episode_length: int
    history_length: int
    fee_rate: float
    alpha: float
    distribution: Literal["gaussian", "dirichlet"]


@dataclass
class ModelConfig:
    """Parameters for the TQC Algorithm."""

    buffer_size: int
    top_quantiles_to_drop_per_net: int
    log_interval: int
    device: str = "cuda"


@dataclass
class TradingConfig:
    """Configuration for Data Loading and Assets."""

    tickers: list[str]
    period: str
    num_episodes: int


@dataclass
class Config:
    """The Root Configuration Object."""

    trading: TradingConfig
    env: EnvConfig
    model: ModelConfig
