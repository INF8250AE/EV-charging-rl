import hydra
import sys
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from loguru import logger as console_logger
from hydra.utils import instantiate
from ev_charging.env import EvChargingEnv
from ev_charging.visual_recorder import EvVizRecorder


@hydra.main(version_base=None, config_path="../configs/", config_name="explore_env")
def main(cfg: DictConfig):

    console_logger.info(f"Python : {sys.version}")
    console_logger.info(f"PYTHONPATH : {sys.path}")
    console_logger.info(f"PyTorch : {torch.__version__}")
    console_logger.info(f"PyTorch CUDA : {torch.version.cuda}")

    env = instantiate(cfg["env"])(seed=cfg["exploration"]["seed"])

    recorder = EvVizRecorder(
        env, output_path="videos/ev_rollout.mp4", fps=int(cfg["logging"]["fps"])
    )

    num_episodes = cfg["exploration"]["nb_episodes"]
    for _ in tqdm(range(num_episodes), desc="Episodes"):
        state_dict, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            state_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            recorder.record_step(action, reward, done)

    recorder.save()


if __name__ == "__main__":
    main()
