import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from ev_charging.stats import RunningMeanStd


class DDQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        target_update_freq: int,
        learning_starts: int,
        eps_start: float,
        eps_end: float,
        eps_decay_steps: int,
        gamma: float,
        learning_rate: float,
        memory_size: int,
        device: str,
        model_hidden_dim: int,
        model_nb_layers: int,
    ):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.epsilon = self.eps_start
        self.train_updates = 0
        self.learning_starts = learning_starts
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.seed = seed
        self.target_update_freq = target_update_freq
        self.generator = torch.Generator().manual_seed(seed)
        self.model_hidden_dim = model_hidden_dim
        self.model_nb_layers = model_nb_layers
        self.reward_rms = RunningMeanStd()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Online and target networks
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()  # copy initial weights

        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.HuberLoss()

    def _build_model(self):
        layers = []
        L = max(1, self.model_nb_layers)  # total Linear layers

        in_dim = self.state_size
        hidden_dim = self.model_hidden_dim

        for i in range(L - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, self.action_size))
        return nn.Sequential(*layers)

    def load_pretrained_weights(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def save(self, path):
        torch.save(
            self.model.state_dict(),
            path,
        )

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        """
        Store a transition in memory.
        state, next_state should be 1D arrays (state_size,)
        action: int, reward: float, done: bool
        """
        r = reward.item()

        self.reward_rms.update(r)
        r_norm = self.reward_rms.normalize(r)

        self.memory.append(
            (
                state.detach().to("cpu"),
                action.detach().to("cpu"),
                torch.tensor(r_norm, dtype=torch.float32, device="cpu"),
                next_state.detach().to("cpu"),
                done,
            )
        )

    def action(self, state):
        """
        ε-greedy policy using the online model.
        state is expected to be shape (state_size,) or (1, state_size).
        Output: int action.
        """
        # Explore
        if torch.rand((), generator=self.generator).item() <= self.epsilon:
            return torch.randint(
                low=0, high=self.action_size, size=(1,), generator=self.generator
            ).to(self.device)

        # Exploit
        state = state.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_size)

        with torch.no_grad():
            q_values = self.model(state)  # (1, action_size)
        action = torch.argmax(q_values[0])
        return action.unsqueeze(0)

    @torch.inference_mode()
    def test_action(self, state):
        state = state.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        q_values = self.model(state)  # (1, A)
        action = torch.argmax(q_values[0])
        return action

    def update(self, batch_size, env_step) -> dict:
        """
        Perform update using a minibatch from memory.
        """
        if len(self.memory) < batch_size or env_step < self.learning_starts:
            return {}
        indices = torch.randperm(len(self.memory), generator=self.generator)[
            :batch_size
        ]

        minibatch = [self.memory[i] for i in indices.tolist()]

        # Stack states and next_states for vectorized prediction
        states = torch.vstack([exp[0] for exp in minibatch]).to(
            self.device
        )  # (B, state_size)
        next_states = torch.vstack([exp[3] for exp in minibatch]).to(
            self.device
        )  # (B, state_size)
        actions = (
            torch.stack([exp[1] for exp in minibatch]).to(self.device).long()
        )  # (B, 1)

        rewards = (
            torch.stack([exp[2] for exp in minibatch]).to(self.device).float()
        )  # (B,)

        dones = torch.tensor(
            [exp[4] for exp in minibatch], device=self.device, dtype=torch.float32
        )  # (B,)

        q_values = self.model(states)  # (B, A)
        q_sa = q_values.gather(1, actions).squeeze(1)  # (B,)

        with torch.no_grad():
            # online selects best next action
            q_next_online = self.model(next_states)  # (B, A)
            best_next_actions = q_next_online.argmax(dim=1, keepdim=True)  # (B, 1)

            # target evaluates that action
            q_next_target = self.target_model(next_states)  # (B, A)
            next_v = q_next_target.gather(1, best_next_actions).squeeze(1)  # (B,)

            target = rewards + (1.0 - dones) * self.gamma * next_v  # (B,)

        self.optimizer.zero_grad()
        loss = self.criterion(q_sa, target)
        loss.backward()
        self.optimizer.step()

        self.train_updates += 1

        return {"train_loss": loss.item()}

    def on_train_step_done(self, env_step: int):
        self.update_epsilon(env_step)

        if self.train_updates > 0 and self.train_updates % self.target_update_freq == 0:
            self.update_target_model()

    def update_epsilon(self, env_step: int):
        if env_step < self.learning_starts:
            self.epsilon = self.eps_start
            return

        t = env_step - self.learning_starts
        frac = min(1.0, t / self.eps_decay_steps)  # goes 0 to 1 over eps_decay_steps
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def update_target_model(self):
        """
        Copy weights from online model to target model.
        """
        self.target_model.load_state_dict(self.model.state_dict())
