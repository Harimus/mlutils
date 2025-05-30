import logging

import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

try:
    import wandb

    def wandb_from_config(config: DictConfig):
        """Initializes a Weights & Biases run from a configuration object.
        Args:
            config (DictConfig): Configuration object containing Weights & Biases
            settings.
            Expects the following structure:
            config.wandb:
                enabled: bool, whether to enable Weights & Biases logging
                project: str, name of the Weights & Biases project
                entity: str, Weights & Biases entity (username or team name)
        Returns:
            wandb.run: A Weights & Biases run object if logging is enabled, otherwise
            None.
        """
        if not config.wandb.enabled:
            logger.warning("Weights & Biases logging is disabled in the config.")
            return None
        run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        )
        return run

    def wandb_log_reward(run, name, reward: float, step: int = None):
        """Logs a single reward value to Weights & Biases."""
        run.log({f"{name}/reward": reward}, step=step)
        logger.info(f"Logged {name}/reward: {reward} to Weights & Biases.")

    def wandb_log_video(run, name, video_path: str):
        """Logs a video to Weights & Biases."""
        run.log({f"{name}/video": wandb.Video(video_path, caption=name)})
        logger.info(f"Logged {name}/video: {video_path} to Weights & Biases.")

    def wandb_log_rollout(run, name, mean: np.ndarray, std: np.ndarray = None):
        """Logs a rollout to Weights & Biases."""
        table = wandb.Table(
            data=[[i, mean[i], std[i]] for i in range(len(mean))],
            columns=["Step", "Mean Reward", "Std Reward"],
        )
        run.log(
            {
                f"{name}/rollout": table,
                f"{name}/mean_reward": wandb.Histogram(mean),
                f"{name}/mean_plot": wandb.plot.line(
                    table, "Step", "Mean Reward", title=f"{name} Mean Reward"
                ),
                f"{name}/std_reward": wandb.Histogram(std) if std is not None else None,
            }
        )

except ImportError:
    logger.warning("wandb is not installed. Using dummy wandb_ functions.")

    def wandb_from_config(config: DictConfig):
        """Dummy function if wandb is not installed."""
        logger.info("wandb is not installed. Skipping Weights & Biases initialization.")
        return None

    def wandb_log_reward(run, name, reward: float):
        """Dummy function if wandb is not installed."""
        logger.info(f"dummy wandb_log_reward called with name={name}, reward={reward}")

    def wandb_log_video(run, name, video_path: str):
        """Dummy function if wandb is not installed."""
        logger.info(
            f" dummy wandb_log_video called with name={name}, video_path={video_path}"
        )

    def wandb_log_rollout(run, name, mean: np.ndarray, std: np.ndarray = None):
        """Dummy function if wandb is not installed."""
        logger.info(
            f"wandb_log_rollout called with name={name}, mean={mean}, std={std}"
        )
