import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

NB_STATE_PER_CAR = 5

NB_STATION_ONLY_STATE_PER_STATION = 8

FILLER_VALUE = -1.0


class EvCar:
    def __init__(
        self,
        id: int,
        device: str,
        capacity: int,
        max_capacity: int,
        soc: float,
        desired_soc: float,
        urgency: float,
    ):
        self.id = id
        self.soc = torch.tensor(soc, dtype=torch.float32, device=device)
        self.desired_soc = torch.tensor(desired_soc, dtype=torch.float32, device=device)
        self.urgency = torch.tensor(urgency, dtype=torch.float32, device=device)
        self.capacity = torch.tensor(capacity, dtype=torch.float32, device=device)
        self.travel_time_remaining = 0
        self.max_travel_time = 1
        self.max_capacity = max_capacity
        self.device = device

    def start_traveling(self, travel_time: int):
        self.max_travel_time = travel_time
        self.travel_time_remaining = travel_time

    def advance(self):
        if not self.has_reached_destination():
            self.travel_time_remaining -= 1

    def has_reached_destination(self) -> bool:
        return self.travel_time_remaining == 0

    def get_state(self) -> torch.Tensor:
        travel_time_norm = torch.tensor(
            self.travel_time_remaining / self.max_travel_time,
            dtype=torch.float32,
            device=self.device,
        )
        capacity_norm = self.capacity / self.max_capacity

        return torch.stack(
            [travel_time_norm, self.soc, self.desired_soc, capacity_norm, self.urgency],
            dim=0,
        )


class EvChargingStation:
    def __init__(
        self,
        id: int,
        device: str,
        charge_speed: int,
        max_charge_speed_all_stations: int,
        charge_speed_sharpness: int,
        max_charge_speed_sharpness_all_stations: int,
        nb_chargers: int,
        travel_distribution: dist.Distribution,
        min_travel_time: int,
        max_travel_time: int,
        max_nb_cars_traveling: int,
        max_nb_cars_waiting: int,
        station_full_penalty: float,
    ):
        self.id = id
        self.charge_speed = charge_speed
        self.max_charge_speed_all_stations = max_charge_speed_all_stations
        self.charge_speed_sharpness = charge_speed_sharpness
        self.max_charge_speed_sharpness_all_stations = (
            max_charge_speed_sharpness_all_stations
        )
        self.nb_chargers = nb_chargers
        self.travel_distribution = travel_distribution
        self.min_travel_time = min_travel_time
        self.max_travel_time = max_travel_time
        self.max_nb_cars_traveling = max_nb_cars_traveling
        self.max_nb_cars_waiting = max_nb_cars_waiting
        self.station_full_penalty = station_full_penalty
        self.device = device
        self.cars_traveling = []
        self.cars_charging = []
        self.cars_waiting = []

    @property
    def free_chargers(self):
        return self.nb_chargers - len(self.cars_charging)

    def reset(self):
        self.cars_traveling = []
        self.cars_charging = []
        self.cars_waiting = []

    def get_max_nb_cars(self):
        return self.max_nb_cars_traveling + self.max_nb_cars_waiting + self.nb_chargers

    def sample_travel_time(self) -> int:
        travel_time = int(
            torch.clamp(
                self.travel_distribution.sample().to(self.device),
                min=self.min_travel_time,
                max=self.max_travel_time,
            ).item()
        )
        return travel_time

    def add_traveling_car(self, car: EvCar, travel_time: int) -> float:
        if self.max_nb_cars_traveling == len(self.cars_traveling):
            return self.station_full_penalty
        car.start_traveling(travel_time)
        self.cars_traveling.append(car)
        return 0.0

    def step(self):
        self.step_charging_cars()
        self.step_waiting_cars()
        self.step_traveling_cars()

    def step_charging_cars(self):
        idx_to_remove = []
        for car_idx, car in enumerate(self.cars_charging):
            charge_speed = self.charge_speed * (
                1 - torch.exp(-self.charge_speed_sharpness * car.soc)
            )
            car.soc += charge_speed / car.capacity
            if car.soc >= car.desired_soc:
                car.soc = car.desired_soc
                idx_to_remove.append(car_idx)

        for idx in sorted(idx_to_remove, reverse=True):
            self.cars_charging.pop(idx)

    def step_waiting_cars(self):
        idx_to_remove = []
        for car_idx, car in enumerate(self.cars_waiting):
            if self.free_chargers > 0:
                self.cars_charging.append(car)
                idx_to_remove.append(car_idx)

        for idx in sorted(idx_to_remove, reverse=True):
            self.cars_waiting.pop(idx)

    def step_traveling_cars(self):
        idx_to_remove = []
        for car_idx, car in enumerate(self.cars_traveling):
            if car.has_reached_destination():
                if self.free_chargers > 0:
                    self.cars_charging.append(car)
                    idx_to_remove.append(car_idx)
                elif len(self.cars_waiting) < self.max_nb_cars_waiting:
                    self.cars_waiting.append(car)
                    idx_to_remove.append(car_idx)
            else:
                car.advance()

        for idx in sorted(idx_to_remove, reverse=True):
            self.cars_traveling.pop(idx)

    def get_state(self) -> torch.Tensor:
        charge_speed_norm = self.charge_speed / self.max_charge_speed_all_stations
        charge_speed_sharpness_norm = (
            self.charge_speed_sharpness / self.max_charge_speed_sharpness_all_stations
        )
        nb_free_chargers_norm = self.free_chargers / self.nb_chargers
        nb_cars_traveling_norm = len(self.cars_traveling) / self.max_nb_cars_traveling
        nb_cars_charging_norm = len(self.cars_charging) / self.nb_chargers
        nb_cars_waiting_norm = len(self.cars_waiting) / self.max_nb_cars_waiting
        is_max_nb_cars_traveling_reached = float(
            len(self.cars_traveling) == self.max_nb_cars_traveling
        )
        is_max_nb_cars_waiting_reached = float(
            len(self.cars_waiting) == self.max_nb_cars_waiting
        )

        station_only_state = torch.tensor(
            [
                charge_speed_norm,
                charge_speed_sharpness_norm,
                nb_free_chargers_norm,
                nb_cars_traveling_norm,
                nb_cars_charging_norm,
                nb_cars_waiting_norm,
                is_max_nb_cars_traveling_reached,
                is_max_nb_cars_waiting_reached,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        traveling_states_padded = torch.full(
            (self.max_nb_cars_traveling, NB_STATE_PER_CAR),
            fill_value=FILLER_VALUE,
            dtype=torch.float32,
            device=self.device,
        )

        for idx, car in enumerate(self.cars_traveling):
            traveling_states_padded[idx] = car.get_state()

        charging_states_padded = torch.full(
            (self.nb_chargers, NB_STATE_PER_CAR),
            fill_value=FILLER_VALUE,
            dtype=torch.float32,
            device=self.device,
        )
        for idx, car in enumerate(self.cars_charging):
            charging_states_padded[idx] = car.get_state()

        waiting_states_padded = torch.full(
            (self.max_nb_cars_waiting, NB_STATE_PER_CAR),
            fill_value=FILLER_VALUE,
            dtype=torch.float32,
            device=self.device,
        )
        for idx, car in enumerate(self.cars_waiting):
            waiting_states_padded[idx] = car.get_state()

        state = torch.cat(
            [
                station_only_state,
                traveling_states_padded.flatten(),
                charging_states_padded.flatten(),
                waiting_states_padded.flatten(),
            ],
            dim=0,
        )

        return state


class EvChargingEnv(gym.Env):
    def __init__(
        self,
        device: str = "cuda",
        seed: int = 42,
        nb_stations: int = 5,
        min_charge_speed: int = 1,
        max_charge_speed: int = 10,
        min_charge_speed_sharpness: int = 1,
        max_charge_speed_sharpness: int = 5,
        min_chargers_per_station: int = 1,
        max_chargers_per_station: int = 5,
        min_travel_time_to_station: int = 20,
        max_travel_time_to_station: int = 100,
        max_nb_cars_traveling_to_station: int = 10,
        max_nb_cars_waiting_at_station: int = 10,
        min_car_capacity: int = 50,
        max_car_capacity: int = 500,
        max_car_init_soc: float = 0.85,
        station_full_penalty: float = 10.0,
        min_steps_between_arrivals: int = 1,
        max_steps_between_arrivals: int = 5,
        max_steps: int = 1000,
        advance_until_next_request: bool = False,
    ):
        super().__init__()
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self._device = device
        self.min_car_capacity = min_car_capacity
        self.max_car_capacity = max_car_capacity
        self.max_car_init_soc = max_car_init_soc
        self.system_full_penalty = station_full_penalty
        self.min_steps_between_arrivals = min_steps_between_arrivals
        self.max_steps_between_arrivals = max_steps_between_arrivals
        self.steps_until_next_arrival = 0
        self.max_steps = max_steps
        self.step_count = 0
        self.advance_until_next_request = advance_until_next_request
        self.stations = []
        for station_id in range(nb_stations):
            station = EvChargingStation(
                id=station_id,
                device=device,
                charge_speed=torch.randint(
                    low=min_charge_speed,
                    high=max_charge_speed + 1,
                    size=(1,),
                    generator=self.generator,
                    device=device,
                ).item(),
                max_charge_speed_all_stations=max_charge_speed,
                charge_speed_sharpness=torch.randint(
                    low=min_charge_speed_sharpness,
                    high=max_charge_speed_sharpness + 1,
                    size=(1,),
                    generator=self.generator,
                    device=device,
                ).item(),
                max_charge_speed_sharpness_all_stations=max_charge_speed_sharpness,
                nb_chargers=torch.randint(
                    low=min_chargers_per_station,
                    high=max_chargers_per_station + 1,
                    size=(1,),
                    generator=self.generator,
                    device=device,
                ).item(),
                travel_distribution=dist.Uniform(
                    low=min_travel_time_to_station, high=max_travel_time_to_station
                ),
                min_travel_time=min_travel_time_to_station,
                max_travel_time=max_travel_time_to_station,
                max_nb_cars_traveling=max_nb_cars_traveling_to_station,
                max_nb_cars_waiting=max_nb_cars_waiting_at_station,
                station_full_penalty=station_full_penalty,
            )
            self.stations.append(station)

        self.max_nb_cars_routed = sum(
            [station.get_max_nb_cars() for station in self.stations]
        )
        padded_state_dim = (
            1  # steps until arrival
            + NB_STATE_PER_CAR
            + self.max_nb_cars_routed * NB_STATE_PER_CAR
            + nb_stations * NB_STATION_ONLY_STATE_PER_STATION
        )
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(padded_state_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(nb_stations)
        self.car_id = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        self.step_count = 0

        for station in self.stations:
            station.reset()

        self.car_to_route = self.generate_car_to_route()

        self.steps_until_next_arrival = torch.zeros((1,), device=self._device)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def generate_car_to_route(self) -> EvCar:
        soc = (
            torch.rand(
                size=(1,),
                generator=self.generator,
                dtype=torch.float32,
                device=self._device,
            ).item()
            * self.max_car_init_soc
        )

        car = EvCar(
            id=self.car_id,
            device=self._device,
            capacity=torch.randint(
                low=self.min_car_capacity,
                high=self.max_car_capacity + 1,
                size=(1,),
                generator=self.generator,
                device=self._device,
            ).item(),
            max_capacity=self.max_car_capacity,
            soc=soc,
            desired_soc=soc
            + torch.rand(
                size=(1,),
                generator=self.generator,
                dtype=torch.float32,
                device=self._device,
            ).item()
            * (1.0 - soc),
            urgency=torch.rand(
                size=(1,),
                generator=self.generator,
                dtype=torch.float32,
                device=self._device,
            ).item(),
        )

        self.car_id += 1

        return car

    def _get_obs(self):
        if self.car_to_route is not None:
            car_to_route_state = self.car_to_route.get_state()
        else:
            car_to_route_state = torch.full(
                (NB_STATE_PER_CAR,),
                fill_value=FILLER_VALUE,
                dtype=torch.float32,
                device=self._device,
            )
        state = torch.concat(
            [self.steps_until_next_arrival]
            + [car_to_route_state]
            + [station.get_state() for station in self.stations]
        )

        return {
            "state": state,
        }

    def _get_info(self):
        return {}

    def step(self, action: torch.Tensor):
        observation, reward, terminated, truncated, info = self._step(action)

        if self.steps_until_next_arrival > 0 and self.advance_until_next_request:
            for _ in range(self.steps_until_next_arrival.item()):
                observation, new_reward, terminated, truncated, info = self._step(
                    action
                )
                reward += new_reward
                done = terminated or truncated
                if done:
                    break

        return observation, reward, terminated, truncated, info

    def _step(self, action: torch.Tensor):
        if self.car_to_route is not None:
            selected_station = self.stations[action.item()]
            actual_travel_time = selected_station.sample_travel_time()
            station_full_penalty = selected_station.add_traveling_car(
                self.car_to_route, actual_travel_time
            )
            self.car_to_route = None
            self.steps_until_next_arrival = torch.randint(
                low=self.min_steps_between_arrivals,
                high=self.max_steps_between_arrivals + 1,
                size=(1,),
                generator=self.generator,
                device=self._device,
            )
        else:
            self.steps_until_next_arrival -= 1
            station_full_penalty = 0.0

        for station in self.stations:
            station.step()

        if self.steps_until_next_arrival <= 0:
            self.car_to_route = self.generate_car_to_route()
        else:
            self.car_to_route = None

        self.step_count += 1

        observation = self._get_obs()
        reward = self.compute_reward(station_full_penalty)
        terminated = False
        truncated = self.step_count >= self.max_steps
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def compute_reward(self, station_full_penalty: float) -> float:
        nb_cars_routed = 0
        cars_urgency = []
        for station in self.stations:
            nb_cars_routed += len(station.cars_traveling)
            nb_cars_routed += len(station.cars_charging)
            nb_cars_routed += len(station.cars_waiting)
            for car in (
                station.cars_traveling + station.cars_charging + station.cars_waiting
            ):
                cars_urgency.append(car.urgency)
        if len(cars_urgency) == 0:
            mean_urgency = 0.0
        else:
            mean_urgency = torch.mean(torch.stack(cars_urgency))

        reward = -(nb_cars_routed / self.max_nb_cars_routed + mean_urgency)

        return reward - station_full_penalty
