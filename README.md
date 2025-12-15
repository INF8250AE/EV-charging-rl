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

# Logging
## CometML  

(Optional): If you want to use CometML logging, set the env variables:  
```bash
export COMET_API_KEY=<YOUR_VALUE>
export COMET_PROJECT_NAME=<YOUR_VALUE>
export COMET_WORKSPACE=<YOUR_VALUE>
```  
Then add this to your train/test command : `logging.use_cometml=true`
## Stdout  

Use `logging.use_cometml=false` when launching the training (eg: `uv run scripts/train_agent.py algo=ddqn logging.use_cometml=false`).  

# Train Agent
## DDQN
```bash
uv run scripts/train_agent.py algo=ddqn
```

## Genetic Algorithm
```bash
uv run scripts/train_ga.py algo.agent.n_generations=5 algo.agent.n_eval_episodes=10 training.n_logging_episodes=30
```

# Test Agent
## DDQN
```bash
uv run scripts/test_agent.py \
    algo=ddqn \
    'eval.weights=["trained_agent_weights_1.pt", "trained_agent_weights_2.pt", "trained_agent_weights_3.pt"]'
```
You need to replace `"trained_agent_weights_X.pt"` with the path to the trained weights (here X represents different training seeds).

## Genetic Algorithm
```bash
uv run scripts/test_agent.py \
    algo=ga \
    'eval.weights=["best_genome_1.pt", "best_genome_2.pt", "best_genome_3.pt"]'
```
You need to replace `"best_genome_X.pt"` with the path to the best genome obtained during training (here X represents different training seeds).

## Random Agent
```bash
uv run scripts/test_agent.py algo=random
```