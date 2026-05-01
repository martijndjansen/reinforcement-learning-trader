import hydra
from sb3_contrib import TQC

from trader import LivePlotCallback, StockDataLoader, TraderEnvironment
from trader.config import Config


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Config) -> None:
    train_env_data, val_env_data = StockDataLoader(
        tickers=cfg.trading.tickers, period=cfg.trading.period
    ).load()

    train_env = TraderEnvironment(env_data=train_env_data, config=cfg.env)
    val_env = TraderEnvironment(env_data=val_env_data, config=cfg.env)

    total_timesteps = cfg.trading.num_episodes * cfg.env.episode_length

    callback = LivePlotCallback(
        train_env=train_env,
        val_env=val_env,
        tickers=cfg.trading.tickers,
        total_timesteps=total_timesteps,
        episode_length=cfg.env.episode_length,
    )

    model = TQC(
        "MlpPolicy",
        train_env,
        top_quantiles_to_drop_per_net=cfg.model.top_quantiles_to_drop_per_net,
        device=cfg.model.device,
        buffer_size=cfg.model.buffer_size,
    )

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=cfg.model.log_interval,
        callback=callback,
    )


if __name__ == "__main__":
    main()
