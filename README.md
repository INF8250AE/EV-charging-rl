# EV-charging-rl
Reinforcement learning models for EV charging assignment and scheduling.

# Setup
Assuming you've installed [`uv`](https://docs.astral.sh/uv/getting-started/installation/)  

1. Install python dependencies
```bash
uv sync
```

# Explore Env
<video width="640" height="480" autoplay src="https://github.com/user-attachments/assets/786091e3-dba6-4ff2-a3de-031dd9f52f3e.mp4"></video>

```bash
uv run scripts/explore_env.py
```

# Train Agent
## Logging
### CometML  

(Optional): If you want to use CometML logging, set the env variables:  
```bash
export COMET_API_KEY=<YOUR_VALUE>
export COMET_PROJECT_NAME=<YOUR_VALUE>
export COMET_WORKSPACE=<YOUR_VALUE>
```  
Then add this to your train command : `logging.use_cometml=false`
### Stdout  

Use `logging.use_cometml=false` when launching the training.  

## DDQN
```bash
uv run scripts/train_agent.py algo=ddqn
```