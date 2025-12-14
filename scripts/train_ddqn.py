import hydra
import sys
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from loguru import logger as console_logger
from ev_charging.env import EvChargingEnv
from ev_charging.visual_recorder import EvVizRecorder


@hydra.main(version_base=None, config_path="../configs/", config_name="train_ddqn")
def main(cfg: DictConfig):

    console_logger.info(f"Python : {sys.version}")
    console_logger.info(f"PYTHONPATH : {sys.path}")
    console_logger.info(f"PyTorch : {torch.__version__}")
    console_logger.info(f"PyTorch CUDA : {torch.version.cuda}")

    env = EvChargingEnv(**cfg["env"]["env_config"])

    recorder = EvVizRecorder(
        env, output_path="videos/ev_rollout.mp4", fps=int(cfg["logging"]["fps"])
    )
    env_device = cfg["env"]["env_config"]["device"]
    console_logger.info(f"Env Device: {env_device}")

    # num_episodes = cfg["training"]["nb_episodes"]
    # for _ in tqdm(range(num_episodes), desc="Episodes"):
    #     obs, info = env.reset()
    #     done = False
    #     while not done:
    #         action = env.action_space.sample()
    #         prev_obs = obs
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated

    #         recorder.record_step(prev_obs, obs, action, reward, done)

    state_size = env.observation_space["state"].shape[0]
    console_logger.info(f"State size: {state_size}")
    nb_actions = env.action_space.n
    console_logger.info(f"Nb actions: {nb_actions}")

    DDQN_agent = DDQNAgentTorch(state_size, nb_actions)
    DDQN_return = []

    batch_size = 64

    for episode in range(num_episodes):
        obs, info = env.reset()
        state = obs["state"].detach().cpu().numpy()  # (state_size,)

        done = False
        truncated = False
        episode_return = 0.0

        while not (done or truncated):
            # choose action (int)
            action = DDQN_agent.action(state)

            # env expecrequires tensor action on env._device
            action_tensor = torch.tensor(action, device=env._device)
            next_obs, reward, done, truncated, info = env.step(action_tensor)

            next_state = next_obs["state"].detach().cpu().numpy()

            reward = float(
                reward.item() if isinstance(reward, torch.Tensor) else reward
            )
            episode_return += reward

            DDQN_agent.remember(state, action, reward, next_state, done or truncated)

            if len(DDQN_agent.memory) > batch_size:
                DDQN_agent.experience_replay(batch_size=batch_size)

            state = next_state

        DDQNAgentTorch.decay_epsilon()
        DDQN_agent.update_target_model()  # At the end of each episode

        DDQN_return.append(episode_return)
        print(f"Episode {episode}, return = {episode_return:.3f}")

    recorder.save()


if __name__ == "__main__":
    main()
