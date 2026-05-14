import hydra
import torch
from aim import Run
from omegaconf import OmegaConf
from torchrl.envs import DoubleToFloat, ObservationNorm, TransformedEnv

from torch_rl_trader.models import build_mlp_sac_modules
from torch_rl_trader.torch_trader_environment import TorchTraderEnvironment
from torch_rl_trader.torch_trader_trainer import TraderTrainer
from trader.config import Config
from trader.stock_data_loader import StockDataLoader

torch.manual_seed(0)


def setup_stable_env(raw_env: TorchTraderEnvironment) -> TransformedEnv:
    env = TransformedEnv(raw_env)

    env.append_transform(DoubleToFloat())

    # Create the norm transform
    obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
    env.append_transform(obs_norm)
    # env.append_transform(RewardScaling(loc=0, scale=0.1))

    obs_norm.init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    return env


FRAMEWORK = "torchrl"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Config) -> None:
    aim_run = Run(repo=".", experiment="rl_trading")

    aim_run.name = (
        f"{FRAMEWORK}_{cfg.model.type}_{cfg.trading.portfolio_name}",
        f"_ep{cfg.trading.num_episodes}",
    )
    aim_run.add_tag(FRAMEWORK)
    aim_run.add_tag(cfg.model.type)
    aim_run.add_tag(cfg.trading.portfolio_name)
    aim_run["hparams"] = OmegaConf.to_container(cfg, resolve=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    train_env_data, val_env_data = StockDataLoader(
        tickers=cfg.trading.tickers,
        period=cfg.trading.period,
    ).load()

    base_env = TorchTraderEnvironment(
        env_data=train_env_data, config=cfg.env, device=device
    )

    val_env_base = TorchTraderEnvironment(
        env_data=val_env_data, config=cfg.env, device=device
    )
    transformed_env = setup_stable_env(base_env)
    val_env = TransformedEnv(val_env_base)
    val_env.append_transform(
        ObservationNorm(
            in_keys=["observation"],
            standard_normal=True,
            loc=transformed_env.transform[1].loc,
            scale=transformed_env.transform[1].scale,
        )
    )

    policy, q_module = build_mlp_sac_modules(
        obs_dim=base_env.observation_spec["observation"].shape[-1],
        act_dim=base_env.action_spec.shape[-1],
        hidden_sizes=[256, 256],
    )
    policy = policy.to(device)
    q_module = q_module.to(device)

    trader_trainer = TraderTrainer(
        cfg=cfg,
        train_env=transformed_env,
        val_env=val_env,
        policy=policy,
        q_module=q_module,
    )

    trader_trainer.train(aim_run=aim_run)


if __name__ == "__main__":
    main()
