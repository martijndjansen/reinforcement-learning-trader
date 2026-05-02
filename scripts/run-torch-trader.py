import hydra
import torch
from aim import Run
from omegaconf import OmegaConf

from torch_rl_trader.models import build_mlp_sac_modules
from torch_rl_trader.torch_trader_environment import TorchTraderEnvironment
from torch_rl_trader.torch_trader_trainer import TraderTrainer
from trader.config import Config
from trader.stock_data_loader import StockDataLoader

torch.manual_seed(0)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Config) -> None:
    aim_run = Run(repo=".", experiment="rl_trading")
    aim_run["hparams"] = OmegaConf.to_container(cfg, resolve=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    train_env_data, _ = StockDataLoader(
        tickers=cfg.trading.tickers,
        period=cfg.trading.period,
    ).load()

    base_env = TorchTraderEnvironment(
        env_data=train_env_data, config=cfg.env, device=device
    )

    policy, q_module = build_mlp_sac_modules(
        obs_dim=base_env.observation_spec["observation"].shape[-1],
        act_dim=base_env.action_spec.shape[-1],
        hidden_sizes=[256, 256],
    )

    trader_trainer = TraderTrainer(
        cfg=cfg, train_env=base_env, policy=policy, q_module=q_module
    )

    trader_trainer.train(aim_run=aim_run)


if __name__ == "__main__":
    main()
