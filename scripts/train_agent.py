import hydra
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig
from loguru import logger as console_logger
from ev_charging.metrics import Metrics


@hydra.main(version_base=None, config_path="../configs/", config_name="train_agent")
def main(cfg: DictConfig):

    console_logger.info(f"Python : {sys.version}")
    console_logger.info(f"PYTHONPATH : {sys.path}")
    console_logger.info(f"PyTorch : {torch.__version__}")
    console_logger.info(f"PyTorch CUDA : {torch.version.cuda}")

    train_seed = cfg["training"]["seed"]
    console_logger.info(f"Training Seed : {train_seed}")

    env = instantiate(cfg["env"])(seed=train_seed)
    state_size = env.observation_space["state"].shape[0]
    console_logger.info(f"State size: {state_size}")
    action_size = env.action_space.n
    console_logger.info(f"Nb actions: {action_size}")
    env_device = cfg["env"]["device"]
    console_logger.info(f"Env Device: {env_device}")

    agent_name = cfg["algo"]["name"]
    agent = instantiate(cfg["algo"]["agent"])(
        state_size=state_size, action_size=action_size, seed=train_seed
    )
    console_logger.info(f"Agent: {agent_name}")

    nb_env_steps = cfg["training"]["nb_env_steps"]
    train_batch_size = cfg["training"]["batch_size"]

    train_episode_metrics = Metrics()
    train_step_metrics = Metrics()

    pbar = tqdm(range(1, nb_env_steps + 1), desc="Env steps")
    pbar.set_postfix(
        train_ep_return="n/a",
    )

    state_dict, info = env.reset()
    state = state_dict["state"]
    done = False

    if cfg["logging"]["use_cometml"]:
        from ev_charging.logger import CometmlLogger

        metrics_logger = CometmlLogger()
    else:
        from ev_charging.logger import StdoutLogger

        metrics_logger = StdoutLogger()

    verbose_log_ep_interval = cfg["logging"]["verbose_log_ep_interval"]
    step_logging_throttle_steps = cfg["logging"]["step_logging_throttle_steps"]
    save_model_step_freq = cfg["logging"]["save_model_step_freq"]
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]
    output_weights_folder = Path(output_dir) / Path("weights")
    output_weights_folder.mkdir(parents=True, exist_ok=True)
    episode_count = 0
    for env_step in pbar:
        action = agent.action(state)
        next_state_dict, reward, terminated, truncated, info = env.step(action)
        next_state = next_state_dict["state"]
        done = terminated or truncated

        agent.add_to_replay_buffer(state, action, reward, next_state, done)

        state = next_state

        train_logs = agent.update(batch_size=train_batch_size, env_step=env_step)
        if train_logs.get("train_loss", None) is not None:
            train_loss = train_logs["train_loss"]
            train_step_metrics.accumulate_metric(
                metric_name="train_step_loss",
                metric_value=train_loss,
                env_step=env_step,
            )

        train_step_metrics.accumulate_metric(
            metric_name="train_reward",
            metric_value=reward.item(),
            env_step=env_step,
        )
        train_step_metrics.accumulate_metric(
            metric_name="train_reward_normalized",
            metric_value=agent.reward_rms.normalize(reward.item()),
            env_step=env_step,
        )
        train_step_metrics.accumulate_metric(
            metric_name="train_action",
            metric_value=action.item(),
            env_step=env_step,
        )

        train_episode_metrics.accumulate_metric(
            metric_name="train_ep_return",
            metric_value=reward.item(),
            env_step=env_step,
            agg_fn=np.sum,
        )
        train_episode_metrics.accumulate_metric(
            metric_name="train_ep_return_std",
            metric_value=reward.item(),
            env_step=env_step,
            agg_fn=np.std,
        )
        train_episode_metrics.accumulate_metric(
            metric_name="train_ep_return_min",
            metric_value=reward.item(),
            env_step=env_step,
            agg_fn=np.min,
        )
        train_episode_metrics.accumulate_metric(
            metric_name="train_ep_return_max",
            metric_value=reward.item(),
            env_step=env_step,
            agg_fn=np.max,
        )
        train_episode_metrics.accumulate_metric(
            metric_name="train_ep_action_median",
            metric_value=action.item(),
            env_step=env_step,
            agg_fn=np.median,
        )
        train_episode_metrics.accumulate_metric(
            metric_name="train_ep_action_min",
            metric_value=action.item(),
            env_step=env_step,
            agg_fn=np.min,
        )
        train_episode_metrics.accumulate_metric(
            metric_name="train_ep_action_max",
            metric_value=action.item(),
            env_step=env_step,
            agg_fn=np.max,
        )
        train_episode_metrics.accumulate_metric(
            metric_name="train_ep_len",
            metric_value=1,
            env_step=env_step,
            agg_fn=np.sum,
        )

        agent.on_train_step_done(env_step)

        if env_step % save_model_step_freq == 0:
            torch.save(
                agent.model.state_dict(),
                output_weights_folder / Path(f"{agent_name}_{env_step}.pt"),
            )

        if env_step % step_logging_throttle_steps == 0:
            step_metrics = train_step_metrics.data

            for metric_name in step_metrics.keys():
                metric_values = step_metrics[metric_name]["values"]
                steps = step_metrics[metric_name]["steps"]
                for metric_value, step in zip(metric_values, steps):
                    metrics_logger.log_metric(
                        name=metric_name, value=metric_value, step=step, silent=True
                    )

            train_step_metrics.data = {}

        if done:
            state_dict, info = env.reset()
            state = state_dict["state"]
            done = False

            episode_count += 1
            aggregated_train_ep_metrics = (
                train_episode_metrics.compute_aggregated_metrics()
            )

            for metric_name in aggregated_train_ep_metrics.keys():
                metric_value = aggregated_train_ep_metrics[metric_name]["value"]
                step = aggregated_train_ep_metrics[metric_name]["step"]
                silent = episode_count % verbose_log_ep_interval != 0
                metrics_logger.log_metric(
                    name=metric_name, value=metric_value, step=step, silent=silent
                )
                if metric_name == "train_ep_return":
                    pbar.set_postfix(
                        train_ep_return=f"{metric_value:.4f}",
                    )


if __name__ == "__main__":
    main()
