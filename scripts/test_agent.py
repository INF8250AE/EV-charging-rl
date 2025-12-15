import hydra
import sys
import json
import numpy as np
import torch
import random
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from loguru import logger as console_logger
from hydra.utils import instantiate
from ev_charging.metrics import Metrics
from ev_charging.visual_recorder import EvVizRecorder


def eval_policy(
    env,
    agent,
    n_eval_episodes: int,
    logging_prefix: str,
    seed: int,
    num_video_rollouts: int,
    video_output_dir,
    video_fps: int,
    agent_id: int,
) -> dict:
    """Runs deterministic evaluation episodes and returns per-episode returns."""
    rollouts_metrics = Metrics()

    for episode in tqdm(
        range(1, n_eval_episodes + 1), desc="Test rollouts", leave=False
    ):
        record_video = episode <= num_video_rollouts
        video_path = (
            video_output_dir / f"{logging_prefix}_agent_{agent_id}_ep_{episode}.mp4"
        )
        recorder = EvVizRecorder(
            env, output_path=str(video_path), fps=video_fps, snapshot_every=1
        )

        state_dict, info = env.reset(seed=seed + episode)
        state = state_dict["state"]
        done = False
        step = 1

        rollout_metrics = Metrics()

        while not done:
            action = agent.test_action(state).to(env._device)

            state_dict, reward, terminated, truncated, info = env.step(action)
            state = state_dict["state"]
            done = terminated or truncated

            if record_video:
                recorder.record_step(action, reward, done, info)

            for k, v in info.items():
                rollout_metrics.accumulate_metric(
                    metric_name=f"{logging_prefix}_ep_{k}_mean",
                    metric_value=v,
                    env_step=step,
                    agg_fn=np.mean,
                )
                rollout_metrics.accumulate_metric(
                    metric_name=f"{logging_prefix}_ep_{k}_std",
                    metric_value=v,
                    env_step=step,
                    agg_fn=np.std,
                )
                rollout_metrics.accumulate_metric(
                    metric_name=f"{logging_prefix}_ep_{k}_min",
                    metric_value=v,
                    env_step=step,
                    agg_fn=np.min,
                )
                rollout_metrics.accumulate_metric(
                    metric_name=f"{logging_prefix}_ep_{k}_max",
                    metric_value=v,
                    env_step=step,
                    agg_fn=np.max,
                )
                rollout_metrics.accumulate_metric(
                    metric_name=f"{logging_prefix}_ep_{k}_sum",
                    metric_value=v,
                    env_step=step,
                    agg_fn=np.sum,
                )

            rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}_ep_return",
                metric_value=reward.item(),
                env_step=step,
                agg_fn=np.sum,
            )
            rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}_ep_reward_std",
                metric_value=reward.item(),
                env_step=step,
                agg_fn=np.std,
            )
            rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}_ep_reward_mean",
                metric_value=reward.item(),
                env_step=step,
                agg_fn=np.mean,
            )
            rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}_ep_reward_min",
                metric_value=reward.item(),
                env_step=step,
                agg_fn=np.min,
            )
            rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}_ep_reward_max",
                metric_value=reward.item(),
                env_step=step,
                agg_fn=np.max,
            )
            rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}_ep_action_median",
                metric_value=action.item(),
                env_step=step,
                agg_fn=np.median,
            )
            rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}_ep_action_std",
                metric_value=action.item(),
                env_step=step,
                agg_fn=np.std,
            )
            rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}_ep_action_min",
                metric_value=action.item(),
                env_step=step,
                agg_fn=np.min,
            )
            rollout_metrics.accumulate_metric(
                metric_name=f"{logging_prefix}_ep_action_max",
                metric_value=action.item(),
                env_step=step,
                agg_fn=np.max,
            )
            step += 1

        if record_video:
            recorder.save()

        rollout_metrics = rollout_metrics.compute_aggregated_metrics()
        for k, v in rollout_metrics.items():
            rollouts_metrics.accumulate_metric(
                metric_name=k, metric_value=v["value"], env_step=episode, agg_fn=np.mean
            )

    return rollouts_metrics.compute_aggregated_metrics()


@hydra.main(version_base=None, config_path="../configs/", config_name="test_agent")
def main(cfg: DictConfig):
    console_logger.info(f"Python : {sys.version}")
    console_logger.info(f"PyTorch : {torch.__version__}")
    console_logger.info(f"PyTorch CUDA : {torch.version.cuda}")

    agent_name = cfg["algo"]["name"]

    seed = cfg["eval"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    env = instantiate(cfg["env"])(seed=seed)

    state_size = env.observation_space["state"].shape[0]
    action_size = env.action_space.n

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    metrics_output_dir = Path(hydra_cfg["runtime"]["output_dir"]) / Path(
        "test_metrics/"
    )
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    video_output_dir = Path(hydra_cfg["runtime"]["output_dir"]) / Path("videos/")
    video_output_dir.mkdir(parents=True, exist_ok=True)

    n_eval_episodes = int(cfg["eval"]["n_eval_episodes"])
    weights_paths = cfg["eval"]["weights"]

    num_video_rollouts = cfg["logging"]["video_rollouts"]
    video_fps = cfg["logging"]["video_fps"]

    test_metrics = Metrics()
    logging_prefix = cfg["logging"]["prefix"]

    for i, weights_path in tqdm(enumerate(weights_paths), desc="Model Weights"):
        console_logger.info(f"Evaluating method={agent_name}, Model weights={i}")

        agent = instantiate(cfg["algo"]["agent"])(
            state_size=state_size, action_size=action_size, seed=seed
        )
        agent.load_pretrained_weights(weights_path)
        agent.eval()
        if agent_name == "ga":
            agent.set_env(env)

        rollouts_metrics = eval_policy(
            env,
            agent,
            n_eval_episodes=n_eval_episodes,
            logging_prefix=logging_prefix,
            seed=seed,
            num_video_rollouts=num_video_rollouts,
            video_output_dir=video_output_dir,
            video_fps=video_fps,
            agent_id=i,
        )

        for k, v in rollouts_metrics.items():
            test_metrics.accumulate_metric(
                metric_name=k, metric_value=v["value"], env_step=i, agg_fn=np.mean
            )

    aggregated_test_metrics = test_metrics.compute_aggregated_metrics()

    with open(
        metrics_output_dir / Path(f"{agent_name}_{logging_prefix}_metrics.json"),
        "w",
    ) as f:
        json.dump(aggregated_test_metrics, f)

    console_logger.info(f"Saved test metrics to : {str(metrics_output_dir)}")


if __name__ == "__main__":
    main()
