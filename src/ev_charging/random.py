import torch


class RandomAgent:
    def __init__(self, state_size: int, action_size: int, seed: int, device: str):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.device = device
        self.loaded_count = 0

    def load_pretrained_weights(self, path: str):
        if self.loaded_count > 0:
            self.seed += 1
            self.generator = torch.Generator().manual_seed(self.seed)

        self.loaded_count += 1

    def eval(self):
        pass

    def train(self):
        pass

    def save(self, path):
        pass

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        pass

    def action(self, state):
        return torch.randint(
            low=0,
            high=self.action_size,
            size=(1,),
            device=self.device,
            dtype=torch.int32,
            generator=self.generator,
        )

    @torch.inference_mode()
    def test_action(self, state):
        return self.action(state)

    def update(self, batch_size, env_step) -> dict:
        pass

    def on_train_step_done(self, env_step: int):
        pass
