class RunningMeanStd:
    def __init__(self, eps: float = 1e-4, momentum: float = 0.999):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
        self.momentum = momentum

    def update(self, x: float):
        delta = x - self.mean
        self.mean += (1.0 - self.momentum) * delta
        self.var = self.momentum * self.var + (1.0 - self.momentum) * delta * delta

    @property
    def std(self):
        return max(self.var**0.5, 1e-6)

    def normalize(self, x: float):
        return (x - self.mean) / self.std
