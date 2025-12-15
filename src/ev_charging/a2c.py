import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from ev_charging.stats import RunningMeanStd


class ActorNet(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_dim: int, nb_layers: int):
        super().__init__()
        layers = []
        L = max(1, nb_layers)

        in_dim = state_size
        for i in range(L - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_size))  
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits (NOT softmax)
        return self.net(x)
        

class CriticNet(nn.Module):
    def __init__(self, state_size: int, hidden_dim: int, nb_layers: int):
        super().__init__()
        layers = []
        L = max(1, nb_layers)

        in_dim = state_size
        for i in range(L - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))  # V(s)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 1)


class ActorCriticAgent:

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        gamma: float,
        actor_lr: float,
        critic_lr: float,
        device: str,
        model_hidden_dim: int,
        model_nb_layers: int,
        normalize_rewards: bool = False,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.gamma = gamma
        self.normalize_rewards = normalize_rewards

        self.generator = torch.Generator().manual_seed(seed)

  
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build actor and critic neural networks
        self.actor = ActorNet(state_size, action_size, model_hidden_dim, model_nb_layers).to(self.device)
        self.critic = CriticNet(state_size, model_hidden_dim, model_nb_layers).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_criterion = nn.HuberLoss()

        # Optional reward normalization like DDQN
        self.reward_rms = RunningMeanStd()

        # Logging
        self.train_updates = 0



    # Action selection by actor NN
    def action(self, state: torch.Tensor):
        """
        Stochastic action for training.
        Input: state tensor (state_size,) or (1, state_size)
        Output: (action_tensor(1,), log_prob_tensor(1,))
        """
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_size)

        logits = self.actor(state)  # (1, action_size)
        dist = Categorical(logits=logits)  
        action = dist.sample(generator=self.generator)  # (1,)
        log_prob = dist.log_prob(action)               # (1,)

        return action, log_prob


    def test_action(self, state: torch.Tensor) -> int:
        """
        Deterministic action for evaluation: argmax over logits.
        """
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(state)  # (1, action_size)
            action = torch.argmax(logits[0]).item()
        return action

    # A2C update (update actor and critic NNs each time step)
    def train_step(self, state, action_log_prob, reward, next_state, done):
        """
        A2C update (one-step TD):
          td_target = r + gamma * V(next_state) * (1 - done)
          td_error  = td_target - V(state)

        actor_loss  = -log π(a|s) * td_error.detach()
        critic_loss = td_error**2
        """
        # Convert to tensors
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).float()

        state = state.to(self.device)
        next_state = next_state.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)

        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.tensor([float(done)], dtype=torch.float32, device=self.device)

        # Critic values
        value = self.critic(state).squeeze(1)        # (1,)
        next_value = self.critic(next_state).squeeze(1)  # (1,)

        td_target = reward + self.gamma * next_value * (1.0 - done)
        td_error = td_target - value                 # (1,)

        # Critic loss (MSE)
        critic_loss = (td_error.pow(2)).mean()

        #update critic network's parameters
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss (advantage = td_error.detach())
        advantage = td_error.detach()
        actor_loss = -(action_log_prob * advantage).mean()

        #update actor network's parameters
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())


    def update(self, state, action, log_prob, reward, next_state, done) -> dict:
        """
        A2C update (one-step TD):
          td_target = r + gamma * V(next_state) * (1 - done)
          td_error  = td_target - V(state)

        actor_loss  = -log π(a|s) * td_error.detach()
        critic_loss = td_error**2
        """
        # Move states to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)

        # Reward handling
        if isinstance(reward, torch.Tensor):
            reward = float(reward.item())
        else:
            reward = float(reward)

        if self.normalize_rewards:
            self.reward_rms.update(reward)
            reward = float(self.reward_rms.normalize(r))

        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)  # (1,)
        done_t = torch.tensor([float(done)], dtype=torch.float32, device=self.device)  # (1,)

        # Critic values
        value = self.critic(state).squeeze(1)        # (1,)
        with torch.no_grad():
            next_value = self.critic(next_state).squeeze(1)  # (1,)
            td_target = reward_t + (1.0 - done_t) * self.gamma * next_value  # (1,)

        advantage = td_target - value  # (1,)

        # Critic loss (Huber like DDQN)
        critic_loss = self.critic_criterion(value, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        # Maximize log_prob * advantage  == minimize -(log_prob * advantage)
        actor_loss = -(log_prob * advantage.detach()).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.train_updates += 1

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "advantage_mean": float(advantage.mean().item()),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
