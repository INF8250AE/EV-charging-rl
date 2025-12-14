import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DDQNAgentTorch:
    # Shared epsilon for all instances
    common_epsilon = 1.0
    common_epsilon_min = 0.0001
    common_epsilon_decay = 0.995

    def __init__(
        self,
        state_size,
        action_size,
        seed: int,
        gamma=0.99,
        learning_rate=1e-3,
        memory_size=2500,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)  # control memory size
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed=seed)
        self.loss = []

        # Device: GPU if available, else CPU
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
        self.criterion = nn.MSELoss()

    def _build_model(self):
        """
        state -> 2 layers -> Q(s, a) for all actions
        """
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition in memory.
        state, next_state should be 1D arrays (state_size,)
        action: int, reward: float, done: bool
        """
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        """
        ε-greedy policy using the online model.
        state is expected to be shape (state_size,) or (1, state_size).
        Output: int action.
        """
        # Explore
        if torch.rand(generator=self.generator) <= DDQNAgentTorch.common_epsilon:
            return torch.randint(
                low=0, high=self.action_size, generator=self.generator
            ).to(self.device)

        # Exploit
        state = state.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_size)

        with torch.no_grad():
            q_values = self.model(state)  # (1, action_size)
        action = torch.argmax(q_values[0]).item()
        return action

    def test_action(self, state):
        """
        Greedy action (no exploration), for evaluation.
        """
        state = state.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state)
        action = torch.argmax(q_values[0]).item()
        return action

    def experience_replay(self, batch_size):
        """
        Perform update using a minibatch from memory.
        """
        minibatch = random.sample(self.memory, batch_size)

        # Stack states and next_states for vectorized prediction
        states = np.vstack([exp[0] for exp in minibatch])  # (B, state_size)
        next_states = np.vstack([exp[3] for exp in minibatch])  # (B, state_size)

        states = torch.from_numpy(states).float().to(self.device)  # (B, state_size)
        next_states = torch.from_numpy(next_states).float().to(self.device)

        # Q(s, ·) from online network
        q_values = self.model(states)  # (B, action_size)

        # Q(s', ·) from online network (for argmax)
        with torch.no_grad():
            q_next_online = self.model(next_states)  # (B, action_size)
            # Q(s', ·) from target network (for value)
            q_next_target = self.target_model(next_states)  # (B, action_size)

        # Build target Q-values
        target_q = q_values.clone().detach()  # (B, action_size)

        for idx, (_, action, reward, _, done) in enumerate(minibatch):

            if done:
                target = reward
            else:
                # Double DQN: choose action using online net, value from target net
                best_next_action = torch.argmax(q_next_online[idx]).item()
                target = (
                    reward + self.gamma * q_next_target[idx, best_next_action].item()
                )

            target_q[idx, action] = target

        # Optimize the online network
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, target_q)
        loss.backward()
        self.optimizer.step()

        self.loss.append(loss.item())

    @classmethod
    def decay_epsilon(cls):
        if cls.common_epsilon > cls.common_epsilon_min:
            cls.common_epsilon *= cls.common_epsilon_decay

    def update_target_model(self):
        """
        Copy weights from online model to target model.
        """
        self.target_model.load_state_dict(self.model.state_dict())
