import hydra
import sys
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from loguru import logger as console_logger
from hydra.utils import get_original_cwd

from ev_charging.env import EvChargingEnv

# correct these ones !!!!!
from agents.ddqn import DDQNAgentTorch
from agents.a2c import ActorCriticAgent
from agents.ga import GeneticAlgorithmEV


def eval_policy(env, action_fn, n_eval_episodes: int):
    """Runs deterministic evaluation episodes and returns per-episode returns."""
    returns = []

    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0

        while not done: 
            action = torch.tensor(int(action_fn(obs)), device=env._device)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward = float(reward.item()) if isinstance(reward, torch.Tensor) else float(reward)
            episode_reward += reward

        returns.append(episode_reward)

    return torch.tensor(returns, dtype=torch.float32)


def make_ddqn_action_fn(agent, device):
    """DDQN evaluation uses greedy Q argmax."""
    def _act(obs):
        state = obs["state"]
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(device)
        return agent.test_action(state)
    return _act


def make_a2c_action_fn(agent, device):
    """A2C evaluation uses argmax over actor probabilities."""
    def _act(obs):
        state = obs["state"]
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(device)
        return agent.greedy_action(state)
    return _act


def make_ga_action_fn(genome, env):
    """GA evaluation uses the heuristic scoring rule (deterministic)."""
    def _act(obs):
        return heuristic_action_from_genome(genome, env, obs)
    return _act

@hydra.main(version_base=None, config_path="../configs/", config_name="test_agent")
def main(cfg: DictConfig):
    console_logger.info(f"Python : {sys.version}")
    console_logger.info(f"PyTorch : {torch.__version__}")
    console_logger.info(f"PyTorch CUDA : {torch.version.cuda}")

    env_device = str(cfg["env"]["env_config"]["device"])
    console_logger.info(f"Env Device: {env_device}")

    methods = list(cfg["eval"]["methods"])                 # ["ddqn","a2c","ga"]
    seeds = list(cfg["eval"]["seeds"])                     # [0,1,2]
    n_eval_episodes = int(cfg["eval"]["n_eval_episodes"])
    weights_cfg = cfg["eval"]["weights"]

    results = {}  # results[method][seed] = torch tensor of returns

    for method in methods:
        results[method] = {}

        for seed in seeds:
            console_logger.info(f"Evaluating method={method}, seed={seed}")

            # fresh env per (method, seed) for consistent seeded initialization
            env = EvChargingEnv(**cfg["env"]["env_config"], seed=int(seed))

            if method == "ddqn":
                weights_path = str(weights_cfg["ddqn"][str(seed)])

                state_size = env.observation_space["state"].shape[0]
                action_size = env.action_space.n

                agent = DDQNAgentTorch(state_size, action_size, device=env_device)
                agent.model.load_state_dict(torch.load(weights_path, map_location=agent.device))
                agent.model.eval()

                action_fn = make_ddqn_action_fn(agent, agent.device)

            elif method == "a2c":
                weights_path = str(weights_cfg["a2c"][str(seed)])

                state_size = env.observation_space["state"].shape[0]
                action_size = env.action_space.n

                agent = ActorCriticAgent(state_size, action_size)
                ckpt = torch.load(weights_path, map_location=agent.device)

                # expected format: {"actor": actor_state_dict, "critic": critic_state_dict} !!!!!
                agent.actor.load_state_dict(ckpt["actor"])
                agent.critic.load_state_dict(ckpt["critic"])
                agent.actor.eval()
                agent.critic.eval()

                action_fn = make_a2c_action_fn(agent, agent.device)

            elif method == "ga":
                genome_path = str(weights_cfg["ga"][str(seed)])
                genome = np.load(genome_path)

                action_fn = make_ga_action_fn(genome, env)

            else:
                raise ValueError(f"Unknown method: {method}")

            episode_returns = eval_policy(env, action_fn, n_eval_episodes=n_eval_episodes)
            results[method][seed] = episode_returns

            console_logger.info(
                f"{method} seed={seed} -> mean={episode_returns.mean().item():.3f}, std={episode_returns.std().item():.3f}"
            )

    # Comparison DataFrames (per-seed & method)
    rows = []
    for method in methods:
        for seed in seeds:
            results = results[method][seed].cpu().numpy()
            rows.append({
                "method": method,
                "seed": int(seed),
                "mean_return": float(np.mean(results)),
                "std_return": float(np.std(results)),
                "n_eval_episodes": int(len(results)),
            })
    df_seed = pd.DataFrame(rows).sort_values(["method", "seed"])

    # Save CSV
    out_dir = get_original_cwd()
    excel_path = os.path.join(out_dir, str(cfg["output"]["excel_name"]))

    df_seed.to_excel(excel_path, index=False)

    console_logger.info(f"Saved per-seed Excel to: {excel_path}")
    

        # recorder.record_step(prev_obs, obs, action, reward, done)

    # save metrics
    # recorder.save()


if __name__ == "__main__":
    main()
