import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from ev_charging.stats import RunningMeanStd


class ActorNet(nn.Module):
    def __init__(
        self, state_size: int, action_size: int, hidden_dim: int, nb_layers: int
    ):
        super().__init__()
        layers = []
        L = max(1, nb_layers)

        in_dim = state_size
        for _ in range(L - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_size))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits


class CriticNet(nn.Module):
    def __init__(self, state_size: int, hidden_dim: int, nb_layers: int):
        super().__init__()
        layers = []
        L = max(1, nb_layers)

        in_dim = state_size
        for _ in range(L - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))  # V(s)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 1)


class ActorCriticAgent:
    """
    A2C update (one-step TD):
      td_target = r + gamma * V(next_state) * (1 - done)
      advantage  = td_target - V(state)                 # advantage = td_error

    actor_loss  = -log π(a|s) * td_error.detach()
    actor_loss ​ = −logπ(a∣s)⋅advantage.detach() - entropy_coef * entropy ---> when we consider entropy term
    critic_loss = td_error**2 -->  HERE:  critic_loss = Huber( V(s), td_target )
    """

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
        entropy_coef: float = 0.0,
        normalize_rewards: bool = False,
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.gamma = gamma
        self.entropy_coef = float(entropy_coef)
        self.normalize_rewards = bool(normalize_rewards)

        self._last_log_prob = None
        self._last_entropy = None
        self._last_action = None

        # seeded generator (for reproducible sampling)
        self.generator = torch.Generator().manual_seed(seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build actor and critic neural networks
        self.actor = ActorNet(
            state_size, action_size, model_hidden_dim, model_nb_layers
        ).to(self.device)
        self.critic = CriticNet(state_size, model_hidden_dim, model_nb_layers).to(
            self.device
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_criterion = nn.HuberLoss()

        # optional reward normalization (like DDQN)
        self.reward_rms = RunningMeanStd()

        # simple logs
        self.train_updates = 0
        self.actor_losses = []
        self.critic_losses = []

    def train(self):
        """Put actor and critic networks into training mode."""
        self.actor.train()
        self.critic.train()

    def eval(self):
        """Put actor and critic networks into evaluation mode."""
        self.actor.eval()
        self.critic.eval()

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

        action = dist.sample()  # (1,)
        log_prob = dist.log_prob(action)  # (1,)
        entropy = dist.entropy()  # (1,)

        # cache for update()
        self._last_action = action
        self._last_log_prob = log_prob
        self._last_entropy = entropy

        return action

    def test_action(self, state: torch.Tensor) -> int:
        """
        Deterministic action for evaluation: argmax over logits.
        """
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(state)  # (1, action_size)
            return torch.argmax(logits[0])

    def update(self, state, reward, next_state, done) -> dict:

        state = state.to(self.device)
        next_state = next_state.to(self.device)

        # Use cached values
        if self._last_log_prob is None:
            raise RuntimeError("A2C update called before action() produced log_prob.")

        log_prob = self._last_log_prob
        entropy = self._last_entropy

        if state.dim() == 1:
            state = state.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)

        # reward -> float
        if isinstance(reward, torch.Tensor):
            r = float(reward.item())
        else:
            r = float(reward)

        # optional reward normalization
        if self.normalize_rewards:
            self.reward_rms.update(r)
            r = float(self.reward_rms.normalize(r))

        reward_t = torch.tensor([r], dtype=torch.float32, device=self.device)  # (1,)
        done_t = torch.tensor(
            [float(done)], dtype=torch.float32, device=self.device
        )  # (1,)

        # Critic values
        value = self.critic(state).squeeze(1)  # (1,)

        with torch.no_grad():
            next_value = self.critic(next_state).squeeze(1)  # (1,)
            td_target = reward_t + (1.0 - done_t) * self.gamma * next_value  # (1,)

        advantage = td_target - value  # (1,)
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # critic update
        critic_loss = self.critic_criterion(value, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        actor_loss = -(log_prob * advantage.detach()).mean()

        # entropy bonus (encourage exploration)
        if self.entropy_coef > 0.0:
            actor_loss = actor_loss - self.entropy_coef * entropy.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.train_updates += 1
        self.actor_losses.append(float(actor_loss.item()))
        self.critic_losses.append(float(critic_loss.item()))

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "advantage_mean": float(advantage.mean().item()),
        }

    def on_train_step_done(self, env_step: int):
        # decay entropy coefficient
        self.entropy_coef = max(0.0001, self.entropy_coef * 0.99999)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load_pretrained_weights(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt)

    def load_state_dict(self, ckpt: dict):
        """
        Load both actor and critic weights.
        """
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    def on_env_transition(
        self, state, action, reward, next_state, done, env_step
    ) -> dict:
        return self.update(state, reward, next_state, done)
