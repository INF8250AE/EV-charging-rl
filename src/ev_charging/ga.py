import torch
from ev_charging.env import NB_STATE_PER_CAR


class GeneticAlgorithmEV:
    def __init__(
        self,
        genome_dim: int,
        population_size: int,
        n_generations: int,
        elite_frac: float,  # The top 20% of genomes (by fitness) survive to the next generation
        mutation_std: float,  # Standard deviation of Gaussian noise added to genome values during mutation.
        mutation_prob: float,  # Probability that a newly created child (via crossover) gets mutated -> 30% get Gaussian noise, others unchanged
        n_eval_episodes: int,
        seed: int,
    ):
        self.genome_dim = genome_dim
        self.population_size = population_size
        self.n_generations = n_generations
        self.elite_frac = elite_frac
        self.mutation_std = mutation_std
        self.mutation_prob = mutation_prob
        self.n_eval_episodes = n_eval_episodes
        self.generator = torch.Generator().manual_seed(seed)

        self.best_genome = None
        self.best_fitness = float("-inf")
        self.best_fitness_history = []

    def _init_population(self):
        # Uniform init in [-1, 1]
        return (
            torch.rand(self.population_size, self.genome_dim, generator=self.generator)
            * 2.0
        ) - 1.0

    def heuristic_action_from_genome(self, genome, env, state):
        """
        genome: np.array of shape (6,) -> weights [w0,w1,w2,w3,w4,w5]
        env: EvChargingEnv instance
        obs: observation dict from env ({"state": torch.Tensor})

        Output: int action = selected station index
        """

        car_vec = state[0:NB_STATE_PER_CAR]

        travel_time_norm = car_vec[0]
        soc = car_vec[1]
        desired_soc = car_vec[2]
        capacity_norm = car_vec[3]
        urgency = car_vec[4]

        scores = []

        # genome = [w0, w1, w2, w3, w4, w5]
        w0, w1, w2, w3, w4, w5 = genome

        # Normalize features in get_state())
        for station in env.stations:
            station_state_t = station.get_state()  # torch.Tensor
            station_state = station_state_t.detach().cpu().numpy()

            # First 8 entries = station_only_state:[charge_speed_norm, charge_speed_sharpness_norm, nb_free_chargers_norm, nb_cars_traveling_norm,
            #  nb_cars_charging_norm, nb_cars_waiting_norm, is_max_nb_cars_traveling_reached, is_max_nb_cars_waiting_reached]
            charge_speed_norm = station_state[0]
            nb_free_chargers_norm = station_state[3]
            nb_cars_waiting_norm = station_state[6]

            # Define features for scoring
            x1 = charge_speed_norm
            x2 = nb_free_chargers_norm
            x3 = nb_cars_waiting_norm
            x4 = urgency
            x5 = 1.0 - soc

            # Linear scoring heuristic
            score = w0 + w1 * x1 + w2 * x2 - w3 * x3 + w4 * x4 + w5 * x5
            scores.append(score)

        best_station = int(torch.argmax(torch.tensor(scores)).item())
        return best_station

    def evaluate_genome(self, genome, env, n_eval_episodes=3):
        total_return = 0.0

        for ep in range(n_eval_episodes):
            obs, info = env.reset()

            episode_return = 0.0
            done = False
            truncated = False

            while not (done or truncated):

                # Select action from heuristic
                action_idx = self.heuristic_action_from_genome(genome, env, obs)
                action_tensor = torch.tensor(
                    action_idx, device=env._device
                )  # Env needs a torch.Tensor action

                obs, reward, done, truncated, info = env.step(action_tensor)

                reward = (
                    float(reward.item()) if isinstance(reward, torch.Tensor) else reward
                )  # reward: torch scalar -> float
                episode_return += reward

            total_return += episode_return

        mean_return = total_return / n_eval_episodes

        return mean_return

    def _evaluate_population(self, population, env):
        fitnesses = []
        for genome in population:
            fit = self.evaluate_genome(
                genome, env, n_eval_episodes=self.n_eval_episodes
            )
            fitnesses.append(fit)
        return torch.tensor(fitnesses)

    def run(self, env):
        population = self._init_population()
        n_elite = max(1, int(self.elite_frac * self.population_size))

        for gen in range(self.n_generations):
            fitnesses = self._evaluate_population(population, env)

            # Track best
            best_idx = torch.argmax(fitnesses).item()
            best_fit = fitnesses[best_idx]
            best_gen = population[best_idx]

            if best_fit > self.best_fitness:
                self.best_fitness = best_fit
                self.best_genome = best_gen.copy()

            self.best_fitness_history.append(self.best_fitness)

            print(
                f"[GA] Gen {gen} | gen_best={best_fit:.3f} | overall_best={self.best_fitness:.3f}"
            )

            # Select elites
            elite_indices = torch.argsort(fitnesses)[-n_elite:]
            elites = population[elite_indices]

            # Build new population
            new_pop = [elites[0]]  # keep the best individual always

            while len(new_pop) < self.population_size:
                # Select two parents randomly from elite set
                p1, p2 = (
                    elites[torch.randint(0, n_elite, generator=self.generator)],
                    elites[torch.randint(0, n_elite, generator=self.generator)],
                )

                # Uniform crossover
                mask = torch.rand(self.genome_dim, generator=self.generator) < 0.5
                child = torch.where(mask, p1, p2)

                # Mutation
                if torch.rand(generator=self.generator) < self.mutation_prob:
                    child = child + torch.normal(
                        0,
                        self.mutation_std,
                        size=self.genome_dim,
                        generator=self.generator,
                    )

                new_pop.append(child)

            population = torch.tensor(new_pop)

        return self.best_genome, self.best_fitness, self.best_fitness_history
