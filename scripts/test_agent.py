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
from scipy.stats import trim_mean
from ev_charging.visual_recorder import EvVizRecorder


def compute_stratified_bootstrap_ci(
    task_means_matrix, metric_fn, n_bootstraps=2000, confidence_level=0.95
):
    num_runs, num_tasks = task_means_matrix.shape
    bootstrapped_scores = []
    rng = np.random.default_rng()

    for _ in range(n_bootstraps):
        bootstrap_sample = np.zeros((num_runs, num_tasks))

        for task_idx in range(num_tasks):
            sampled_indices = rng.choice(num_runs, size=num_runs, replace=True)
            bootstrap_sample[:, task_idx] = task_means_matrix[sampled_indices, task_idx]

        metric_value = metric_fn(bootstrap_sample.flatten())
        bootstrapped_scores.append(metric_value)

    sorted_scores = np.sort(bootstrapped_scores)
    alpha = (1.0 - confidence_level) / 2.0
    lower_idx = int(alpha * n_bootstraps)
    upper_idx = int((1.0 - alpha) * n_bootstraps)

    return sorted_scores[lower_idx], sorted_scores[upper_idx]


def eval_policy_get_mean_return(
    env,
    agent,
    n_eval_episodes: int,
    logging_prefix: str,
    base_seed: int,
    num_video_rollouts: int,
    video_output_dir,
    video_fps: int,
    agent_id: int,
    env_seed_id: int,
):
    episode_returns = []

    collected_metrics = {}

    traffic_seed_base = base_seed * 10000

    for episode in tqdm(
        range(1, n_eval_episodes + 1), desc=f"  Env {env_seed_id} episodes", leave=False
    ):
        record_video = episode <= num_video_rollouts and agent_id == 0
        video_path = (
            video_output_dir
            / f"{logging_prefix}_agent_{agent_id}_env_{env_seed_id}_ep_{episode}.mp4"
        )

        if record_video:
            recorder = EvVizRecorder(
                env, output_path=str(video_path), fps=video_fps, snapshot_every=1
            )

        state_dict, info = env.reset(seed=traffic_seed_base + episode)
        state = state_dict["state"]
        done = False
        episode_reward = 0.0

        episode_metrics = {}

        while not done:
            action = agent.test_action(state).to(env._device)
            state_dict, reward, terminated, truncated, info = env.step(action)
            state = state_dict["state"]
            done = terminated or truncated
            episode_reward += reward.item()

            if info is not None:
                for key, value in info.items():
                    if isinstance(value, (int, float, np.number)):
                        if key not in episode_metrics:
                            episode_metrics[key] = []
                        episode_metrics[key].append(float(value))

            if record_video:
                recorder.record_step(action, reward, done, info)

        if record_video:
            recorder.save()

        episode_returns.append(episode_reward)

        for metric_name, values in episode_metrics.items():
            if metric_name not in collected_metrics:
                collected_metrics[metric_name] = {
                    "mean": [],
                    "sum": [],
                    "min": [],
                    "max": [],
                    "std": [],
                }

            collected_metrics[metric_name]["mean"].append(np.mean(values))
            collected_metrics[metric_name]["sum"].append(np.sum(values))
            collected_metrics[metric_name]["min"].append(np.min(values))
            collected_metrics[metric_name]["max"].append(np.max(values))
            collected_metrics[metric_name]["std"].append(np.std(values))

    detailed_metrics = {}
    for metric_name, agg_dict in collected_metrics.items():
        detailed_metrics[metric_name] = {
            "mean": {
                "value": float(np.mean(agg_dict["mean"])),
                "std": float(np.std(agg_dict["mean"])),
            },
            "sum": {
                "value": float(np.mean(agg_dict["sum"])),
                "std": float(np.std(agg_dict["sum"])),
            },
            "min": {
                "value": float(np.mean(agg_dict["min"])),
                "std": float(np.std(agg_dict["min"])),
            },
            "max": {
                "value": float(np.mean(agg_dict["max"])),
                "std": float(np.std(agg_dict["max"])),
            },
            "std": {
                "value": float(np.mean(agg_dict["std"])),
                "std": float(np.std(agg_dict["std"])),
            },
        }

    return np.mean(episode_returns), detailed_metrics


@hydra.main(version_base=None, config_path="../configs/", config_name="test_agent")
def main(cfg: DictConfig):
    console_logger.info(f"Python: {sys.version}")
    console_logger.info(f"PyTorch: {torch.__version__}")

    global_seed = int(cfg["eval"]["global_seed"])
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_seed)

    agent_name = str(cfg["algo"]["name"])
    test_env_seeds = list(cfg["eval"]["test_seeds"])
    weights_paths = list(cfg["eval"]["weights"])
    n_eval_episodes = int(cfg["eval"]["n_eval_episodes"])

    num_runs = len(weights_paths)  # N
    num_tasks = len(test_env_seeds)  # M

    console_logger.info(f"\n{'='*70}")
    console_logger.info(f"EVALUATION PROTOCOL (following paper's formalism)")
    console_logger.info(f"{'='*70}")
    console_logger.info(f"Algorithm: {agent_name}")
    console_logger.info(f"M tasks (test envs): {num_tasks}")
    console_logger.info(f"N runs (trained models): {num_runs}")
    console_logger.info(f"Episodes per (run, task): {n_eval_episodes}")
    console_logger.info(
        f"Total evaluations: {num_runs} x {num_tasks} x {n_eval_episodes} = {num_runs * num_tasks * n_eval_episodes}"
    )
    console_logger.info(
        f"Aggregate metrics computed over: {num_runs} x {num_tasks} = {num_runs * num_tasks} task-averaged scores"
    )
    console_logger.info(f"{'='*70}\n")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = Path(hydra_cfg["runtime"]["output_dir"])
    metrics_output_dir = output_dir / "test_metrics"
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    video_output_dir = output_dir / "videos"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    task_means_matrix = np.zeros((num_runs, num_tasks))

    detailed_metrics_matrices = {}

    all_models_details = []

    for run_idx, weights_path in enumerate(weights_paths):
        console_logger.info(f"\n{'='*70}")
        console_logger.info(f"EVALUATING RUN {run_idx+1}/{num_runs}")
        console_logger.info(f"Weights: {weights_path}")
        console_logger.info(f"{'='*70}")

        env_dummy = instantiate(cfg["env"])(seed=0)
        state_size = env_dummy.observation_space["state"].shape[0]
        action_size = env_dummy.action_space.n
        del env_dummy

        agent = instantiate(cfg["algo"]["agent"])(
            state_size=state_size, action_size=action_size, seed=global_seed
        )
        agent.load_pretrained_weights(weights_path)
        agent.eval()

        model_scores_per_env = []
        model_detailed_metrics = []

        for task_idx, env_seed in enumerate(
            tqdm(
                test_env_seeds,
                desc=f"Run {run_idx+1}/{num_runs} - Testing Envs",
                position=0,
            )
        ):
            env = instantiate(cfg["env"])(seed=env_seed)

            if agent_name == "ga":
                agent.set_env(env)

            mean_return, detailed_metrics = eval_policy_get_mean_return(
                env=env,
                agent=agent,
                n_eval_episodes=n_eval_episodes,
                logging_prefix=str(cfg["logging"]["prefix"]),
                base_seed=env_seed,
                num_video_rollouts=int(cfg["logging"]["video_rollouts"]),
                video_output_dir=video_output_dir,
                video_fps=int(cfg["logging"]["video_fps"]),
                agent_id=run_idx,
                env_seed_id=env_seed,
            )

            task_means_matrix[run_idx, task_idx] = mean_return
            model_scores_per_env.append(mean_return)
            model_detailed_metrics.append(detailed_metrics)

            for metric_name, agg_dict in detailed_metrics.items():
                if metric_name not in detailed_metrics_matrices:
                    detailed_metrics_matrices[metric_name] = {
                        "mean": np.zeros((num_runs, num_tasks)),
                        "sum": np.zeros((num_runs, num_tasks)),
                        "min": np.zeros((num_runs, num_tasks)),
                        "max": np.zeros((num_runs, num_tasks)),
                        "std": np.zeros((num_runs, num_tasks)),
                    }

                for agg_type in ["mean", "sum", "min", "max", "std"]:
                    detailed_metrics_matrices[metric_name][agg_type][
                        run_idx, task_idx
                    ] = agg_dict[agg_type]["value"]

            env.close()

        console_logger.info(
            f"\nRun {run_idx+1} - Mean per env: {np.array2string(np.array(model_scores_per_env), precision=2)}"
        )
        console_logger.info(
            f"Run {run_idx+1} - Overall mean: {np.mean(model_scores_per_env):.2f}"
        )

        all_models_details.append(
            {
                "run_id": run_idx,
                "weights_path": str(weights_path),
                "scores_per_env": model_scores_per_env,
                "mean_over_envs": float(np.mean(model_scores_per_env)),
                "detailed_metrics_per_env": model_detailed_metrics,
            }
        )

    console_logger.info(f"\n{'='*70}")
    console_logger.info(f"COMPUTING AGGREGATE METRICS")
    console_logger.info(f"{'='*70}")

    all_task_means = task_means_matrix.flatten()

    console_logger.info(f"Task means matrix shape: {task_means_matrix.shape}")
    console_logger.info(f"Total scores for aggregation: {len(all_task_means)}")

    def iqm_fn(x):
        return trim_mean(x, proportiontocut=0.25)

    mean_val = np.mean(all_task_means)
    median_val = np.median(all_task_means)
    iqm_val = iqm_fn(all_task_means)

    console_logger.info(f"Computing stratified bootstrap CIs (50000 resamples)...")
    console_logger.info(f"This may take a minute...")

    mean_ci = compute_stratified_bootstrap_ci(
        task_means_matrix, np.mean, n_bootstraps=50000
    )
    median_ci = compute_stratified_bootstrap_ci(
        task_means_matrix, np.median, n_bootstraps=50000
    )
    iqm_ci = compute_stratified_bootstrap_ci(
        task_means_matrix, iqm_fn, n_bootstraps=50000
    )

    console_logger.info(f"\nComputing aggregate metrics for detailed info metrics...")

    detailed_metrics_aggregates = {}

    for metric_name, agg_matrices in detailed_metrics_matrices.items():
        detailed_metrics_aggregates[metric_name] = {}

        for agg_type, matrix in agg_matrices.items():
            all_values = matrix.flatten()

            iqm_value = iqm_fn(all_values)
            mean_value = np.mean(all_values)
            median_value = np.median(all_values)

            iqm_ci_detailed = compute_stratified_bootstrap_ci(
                matrix, iqm_fn, n_bootstraps=10000
            )
            mean_ci_detailed = compute_stratified_bootstrap_ci(
                matrix, np.mean, n_bootstraps=10000
            )

            detailed_metrics_aggregates[metric_name][agg_type] = {
                "iqm": {
                    "value": float(iqm_value),
                    "ci_lower": float(iqm_ci_detailed[0]),
                    "ci_upper": float(iqm_ci_detailed[1]),
                },
                "mean": {
                    "value": float(mean_value),
                    "ci_lower": float(mean_ci_detailed[0]),
                    "ci_upper": float(mean_ci_detailed[1]),
                },
                "median": {"value": float(median_value)},
            }

    console_logger.success(
        f"Computed aggregates for {len(detailed_metrics_aggregates)} detailed metrics"
    )

    console_logger.success(f"Aggregate metrics for {agent_name}")
    console_logger.success(
        f"Computed over {num_runs} runs x {num_tasks} tasks = {num_runs * num_tasks} scores\n"
    )

    console_logger.info(f"Return:")
    console_logger.info(f"{'='*70}")
    console_logger.info(
        f"Mean:   {mean_val:8.2f}  [95% CI: {mean_ci[0]:7.2f}, {mean_ci[1]:7.2f}]"
    )
    console_logger.info(
        f"Median: {median_val:8.2f}  [95% CI: {median_ci[0]:7.2f}, {median_ci[1]:7.2f}]"
    )
    console_logger.info(
        f"IQM:    {iqm_val:8.2f}  [95% CI: {iqm_ci[0]:7.2f}, {iqm_ci[1]:7.2f}]"
    )

    if detailed_metrics_aggregates:
        console_logger.info(f"\n{'='*70}")
        console_logger.info(f"DETAILED INFO METRICS (Top metrics):")
        console_logger.info(f"{'='*70}")

        for idx, (metric_name, agg_dict) in enumerate(
            list(detailed_metrics_aggregates.items())[:5]
        ):
            console_logger.info(f"\n{metric_name}:")
            mean_agg = agg_dict["mean"]
            console_logger.info(
                f"  IQM: {mean_agg['iqm']['value']:8.2f}  "
                f"[{mean_agg['iqm']['ci_lower']:7.2f}, {mean_agg['iqm']['ci_upper']:7.2f}]"
            )

        if len(detailed_metrics_aggregates) > 5:
            console_logger.info(
                f"\n... and {len(detailed_metrics_aggregates) - 5} more metrics (see JSON output)"
            )
    console_logger.info(f"{'='*70}\n")

    final_results = {
        "agent": agent_name,
        "evaluation_protocol": {
            "num_runs": num_runs,
            "num_tasks": num_tasks,
            "episodes_per_task": n_eval_episodes,
            "test_env_seeds": test_env_seeds,
            "bootstrap_samples": 50000,
            "confidence_level": 0.95,
        },
        "task_means_matrix": task_means_matrix.tolist(),
        "aggregate_metrics": {
            "episode_return": {
                "mean": {
                    "value": float(mean_val),
                    "ci_lower": float(mean_ci[0]),
                    "ci_upper": float(mean_ci[1]),
                    "ci_width": float(mean_ci[1] - mean_ci[0]),
                },
                "median": {
                    "value": float(median_val),
                    "ci_lower": float(median_ci[0]),
                    "ci_upper": float(median_ci[1]),
                    "ci_width": float(median_ci[1] - median_ci[0]),
                },
                "iqm": {
                    "value": float(iqm_val),
                    "ci_lower": float(iqm_ci[0]),
                    "ci_upper": float(iqm_ci[1]),
                    "ci_width": float(iqm_ci[1] - iqm_ci[0]),
                    "recommended": True,
                },
            },
            "detailed_info_metrics": detailed_metrics_aggregates,
        },
        "detailed_metrics_matrices": {
            metric_name: {
                agg_type: matrix.tolist() for agg_type, matrix in agg_dict.items()
            }
            for metric_name, agg_dict in detailed_metrics_matrices.items()
        },
        "per_run_details": all_models_details,
    }

    # Save detailed JSON
    output_file = metrics_output_dir / f"{agent_name}_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)

    console_logger.success(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
