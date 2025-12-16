import torch
import numpy as np
from ev_charging.env import NB_STATE_PER_CAR


class GeneticAlgorithmEV:
    def __init__(
        self,
        genome_dim: int,
        population_size: int,
        n_generations: int,
        elite_frac: float,
        mutation_std: float,
        mutation_prob: float,
        n_eval_episodes: int,
        seed: int,
        state_size=None,  # For compatibility with test script
        action_size=None,  # For compatibility with test script
    ):
        self.genome_dim = genome_dim
        self.population_size = population_size
        self.n_generations = n_generations
        self.elite_frac = elite_frac
        self.mutation_std = mutation_std
        self.mutation_prob = mutation_prob
        self.n_eval_episodes = n_eval_episodes
        self.generator = torch.Generator().manual_seed(seed)

        self.best_genome = None  # Set by run() or load_pretrained_weights()
        self.best_fitness = float("-inf")
        self.best_fitness_history = []

        self.env = None  # Set during testing
        self.population = None  # Initialized in initialize_training
        self.n_elite = None  # Initialized in initialize_training

    def _init_population(self):
        return (
            torch.rand(self.population_size, self.genome_dim, generator=self.generator)
            * 2.0
        ) - 1.0

    def heuristic_action_from_genome(self, genome, env, state):
        # Ensure state is a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)

        car_vec = state[0:NB_STATE_PER_CAR]

        urgency = car_vec[4]
        soc = car_vec[1]

        scores = []
        w0, w1, w2, w3, w4, w5 = genome

        for station in env.stations:
            station_state = station.get_state()

            charge_speed_norm = station_state[0]
            nb_free_chargers_norm = station_state[3]
            nb_cars_waiting_norm = station_state[6]

            x1 = charge_speed_norm
            x2 = nb_free_chargers_norm
            x3 = nb_cars_waiting_norm
            x4 = urgency
            x5 = 1.0 - soc

            score = w0 + w1 * x1 + w2 * x2 - w3 * x3 + w4 * x4 + w5 * x5
            scores.append(score)

        scores_tensor = torch.stack(scores)
        best_station = torch.argmax(scores_tensor).item()
        return best_station

    def test_action(self, state):
        if self.best_genome is None:
            raise ValueError(
                "No genome loaded! Must call load_pretrained_weights() first."
            )
        if self.env is None:
            raise ValueError("Environment not set! Must call set_env() first.")

        action_idx = self.heuristic_action_from_genome(
            self.best_genome, self.env, state
        )
        return torch.tensor(action_idx)

    def set_env(self, env):
        self.env = env

    def load_pretrained_weights(self, weights_path):
        checkpoint = torch.load(weights_path, weights_only=True)
        self.best_genome = checkpoint["best_genome"]
        self.best_fitness = checkpoint.get("best_fitness", float("-inf"))
        print(f"Loaded genome with fitness: {self.best_fitness:.3f}")

    def save(self, path):
        torch.save(
            {
                "best_genome": self.best_genome,
                "best_fitness": self.best_fitness,
                "best_fitness_history": self.best_fitness_history,
            },
            path,
        )

    def eval(self):
        pass

    def train(self):
        pass

    def evaluate_genome(self, genome, env, n_eval_episodes=3):
        all_returns = []
        all_rewards = []
        all_actions = []
        all_infos = []

        for ep in range(n_eval_episodes):
            obs, info = env.reset()
            state = obs["state"]
            episode_return = 0.0
            episode_rewards = []
            episode_actions = []
            done = False
            truncated = False

            while not (done or truncated):
                action_idx = self.heuristic_action_from_genome(genome, env, state)
                action_tensor = torch.tensor(action_idx, device=env._device)
                obs, reward, done, truncated, info = env.step(action_tensor)
                state = obs["state"]

                reward_val = reward.item()
                episode_return += reward_val

                episode_rewards.append(reward_val)
                episode_actions.append(action_idx)
                all_infos.append(info)

            all_returns.append(episode_return)
            all_rewards.extend(episode_rewards)
            all_actions.extend(episode_actions)

        mean_return = np.mean(all_returns) if all_returns else 0.0

        return (
            mean_return,
            np.array(all_returns),
            np.array(all_rewards),
            np.array(all_actions),
            all_infos,
        )

    def _evaluate_population(self, population, env):
        fitnesses = []
        for genome in population:
            fit, _, _, _, _ = self.evaluate_genome(
                genome, env, n_eval_episodes=self.n_eval_episodes
            )
            fitnesses.append(fit)
        return torch.tensor(fitnesses)

    def initialize_training(self, env):
        self.env = env
        self.population = self._init_population()
        self.n_elite = max(1, int(self.elite_frac * self.population_size))

    def run_generation(self):
        if self.env is None or self.population is None:
            raise ValueError("Must call initialize_training(env) first.")

        env = self.env

        fitnesses = self._evaluate_population(self.population, env)

        best_idx = torch.argmax(fitnesses).item()
        best_fit = fitnesses[best_idx].item()
        best_gen = self.population[best_idx]

        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            self.best_genome = best_gen.clone()

        self.best_fitness_history.append(self.best_fitness)

        # Selection: keep top performers
        elite_indices = torch.argsort(fitnesses)[-self.n_elite :]
        elites = self.population[elite_indices]

        new_pop = [elites[-1]]  # Keep best

        while len(new_pop) < self.population_size:
            # Crossover
            idx1 = torch.randint(0, self.n_elite, (1,), generator=self.generator).item()
            idx2 = torch.randint(0, self.n_elite, (1,), generator=self.generator).item()
            p1, p2 = elites[idx1], elites[idx2]

            mask = torch.rand(self.genome_dim, generator=self.generator) < 0.5
            child = torch.where(mask, p1, p2)

            # Mutation
            if torch.rand(1, generator=self.generator).item() < self.mutation_prob:
                child = (
                    child
                    + torch.randn(self.genome_dim, generator=self.generator)
                    * self.mutation_std
                )

            new_pop.append(child)

        self.population = torch.stack(new_pop)

        return best_gen, best_fit
