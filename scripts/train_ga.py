import hydra
import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from loguru import logger as console_logger
from hydra.utils import instantiate
from tqdm import tqdm
from ev_charging.logger import CometmlLogger, StdoutLogger


def log_generation_metrics(
    gen_idx, best_genome, agent, env, metrics_logger, n_logging_episodes
):
    """
    Evaluates the best genome and logs detailed metrics for the generation.
    """

    _, returns, rewards, actions, all_infos = agent.evaluate_genome(
        agent.best_genome, env, n_eval_episodes=n_logging_episodes
    )

    if all_infos:
        keys = all_infos[0].keys()
        info_metrics = {k: np.array([d[k] for d in all_infos]) for k in keys}
    else:
        info_metrics = {}

    for k, values in info_metrics.items():
        if values.size > 0:
            metrics_logger.log_metric(
                name=f"train_gen_{k}_mean", value=np.mean(values), step=gen_idx
            )
            metrics_logger.log_metric(
                name=f"train_gen_{k}_std", value=np.std(values), step=gen_idx
            )
            metrics_logger.log_metric(
                name=f"train_gen_{k}_min", value=np.min(values), step=gen_idx
            )
            metrics_logger.log_metric(
                name=f"train_gen_{k}_max", value=np.max(values), step=gen_idx
            )
            metrics_logger.log_metric(
                name=f"train_gen_{k}_sum", value=np.sum(values), step=gen_idx
            )

    # GA-Specific Metrics
    metrics_logger.log_metric(
        name="ga_best_fitness", value=agent.best_fitness, step=gen_idx
    )
    metrics_logger.log_metric(
        name="ga_gen_best_fitness", value=np.mean(returns), step=gen_idx
    )

    # RL-Style Metrics
    metrics_logger.log_metric(
        name="train_gen_return_mean", value=np.mean(returns), step=gen_idx
    )

    metrics_logger.log_metric(
        name="train_gen_reward_mean", value=np.mean(rewards), step=gen_idx
    )
    metrics_logger.log_metric(
        name="train_gen_reward_std", value=np.std(rewards), step=gen_idx
    )
    metrics_logger.log_metric(
        name="train_gen_reward_min", value=np.min(rewards), step=gen_idx
    )
    metrics_logger.log_metric(
        name="train_gen_reward_max", value=np.max(rewards), step=gen_idx
    )

    # Action stats
    metrics_logger.log_metric(
        name="train_gen_action_median", value=np.median(actions), step=gen_idx
    )
    metrics_logger.log_metric(
        name="train_gen_action_std", value=np.std(actions), step=gen_idx
    )
    metrics_logger.log_metric(
        name="train_gen_action_min", value=np.min(actions), step=gen_idx
    )
    metrics_logger.log_metric(
        name="train_gen_action_max", value=np.max(actions), step=gen_idx
    )


@hydra.main(version_base=None, config_path="../configs/", config_name="train_ga")
def main(cfg: DictConfig):
    console_logger.info(f"Python : {sys.version}")
    console_logger.info(f"PyTorch : {torch.__version__}")
    console_logger.info(f"PyTorch CUDA : {torch.version.cuda}")

    agent_name = cfg["algo"]["name"]
    seed = cfg["training"]["seed"]

    console_logger.info("Creating training environment...")
    env = instantiate(cfg["env"])(seed=seed)
    env.reset(seed=seed)

    console_logger.info(f"Initializing {agent_name} agent...")
    agent = instantiate(cfg["algo"]["agent"])(
        genome_dim=cfg["algo"]["agent"]["genome_dim"],
        population_size=cfg["algo"]["agent"]["population_size"],
        n_generations=cfg["algo"]["agent"]["n_generations"],
        elite_frac=cfg["algo"]["agent"]["elite_frac"],
        mutation_std=cfg["algo"]["agent"]["mutation_std"],
        mutation_prob=cfg["algo"]["agent"]["mutation_prob"],
        n_eval_episodes=cfg["algo"]["agent"]["n_eval_episodes"],
        seed=seed,
    )

    n_logging_episodes = cfg["training"]["n_logging_episodes"]

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = Path(hydra_cfg["runtime"]["output_dir"])

    if cfg["logging"]["use_cometml"]:
        metrics_logger = CometmlLogger()
    else:
        metrics_logger = StdoutLogger()

    metrics_logger.log_parameters(OmegaConf.to_container(cfg, resolve=True))
    metrics_logger.log_code("./src/")
    metrics_logger.log_code("./scripts/")
    metrics_logger.log_code("./configs/")

    console_logger.info("Starting GA evolution...")
    console_logger.info(f"Population size: {cfg['algo']['agent']['population_size']}")
    console_logger.info(f"Generations: {cfg['algo']['agent']['n_generations']}")
    console_logger.info(f"Elite fraction: {cfg['algo']['agent']['elite_frac']}")
    console_logger.info(
        f"Mutation probability: {cfg['algo']['agent']['mutation_prob']}"
    )
    console_logger.info(f"Mutation std: {cfg['algo']['agent']['mutation_std']}")
    console_logger.info(
        f"Eval episodes per genome: {cfg['algo']['agent']['n_eval_episodes']}"
    )

    agent.initialize_training(env)

    pbar = tqdm(range(1, agent.n_generations + 1), desc="GA Generations")

    for gen in pbar:
        best_gen_genome, best_gen_fit = agent.run_generation()

        log_generation_metrics(
            gen, agent.best_genome, agent, env, metrics_logger, n_logging_episodes
        )

        pbar.set_postfix(
            gen_best=f"{best_gen_fit:.3f}", overall_best=f"{agent.best_fitness:.3f}"
        )

        if gen % 10 == 0 or gen == agent.n_generations:
            console_logger.info(
                f"[GA] Gen {gen}/{agent.n_generations} | Gen Best: {best_gen_fit:.3f} | Overall Best: {agent.best_fitness:.3f}"
            )

    best_genome = agent.best_genome
    best_fitness = agent.best_fitness
    history = agent.best_fitness_history

    save_path = output_dir / f"{agent_name}_best_genome_seed_{seed}.pt"
    agent.save(str(save_path))

    console_logger.info("=" * 60)
    console_logger.info("Training complete!")
    console_logger.info(f"Best fitness achieved: {best_fitness:.3f}")
    console_logger.info(f"Best genome: {best_genome.cpu().numpy()}")
    console_logger.info(f"Saved to: {save_path}")
    console_logger.info("=" * 60)

    history_path = output_dir / f"{agent_name}_fitness_history_seed_{seed}.pt"
    torch.save(
        {
            "fitness_history": history,
            "best_fitness": best_fitness,
        },
        history_path,
    )
    console_logger.info(f"Fitness history saved to: {history_path}")


if __name__ == "__main__":
    main()
