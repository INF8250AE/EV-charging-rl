import hydra
import sys
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from loguru import logger as console_logger
from hydra.utils import instantiate
from pathlib import Path
from ev_charging.visual_recorder import EvVizRecorder


@hydra.main(version_base=None, config_path="../configs/", config_name="explore_env")
def main(cfg: DictConfig):

    console_logger.info(f"Python : {sys.version}")
    console_logger.info(f"PYTHONPATH : {sys.path}")
    console_logger.info(f"PyTorch : {torch.__version__}")
    console_logger.info(f"PyTorch CUDA : {torch.version.cuda}")

    env = instantiate(cfg["env"])(seed=cfg["exploration"]["seed"])

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]
    video_output_dir = Path(output_dir) / Path("videos/")
    video_output_dir.mkdir()
    video_file = video_output_dir / Path("ev_rollout.mp4")

    recorder = EvVizRecorder(
        env, output_path=str(video_file), fps=int(cfg["logging"]["fps"])
    )

    num_episodes = cfg["exploration"]["nb_episodes"]
    for _ in tqdm(range(num_episodes), desc="Episodes"):
        state_dict, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            state_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if len(recorder.snaps) < cfg["logging"]["max_video_frames"]:
                recorder.record_step(action, reward, done)

    recorder.save()


if __name__ == "__main__":
    main()
