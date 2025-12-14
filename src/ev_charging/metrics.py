import numpy as np


class Metrics:
    def __init__(self):
        self.data = {}

    def accumulate_metric(
        self,
        metric_name: str,
        metric_value: float,
        env_step: int,
        agg_fn=lambda x: np.array([x]),
    ):
        if self.data.get(metric_name) is None:
            self.data[metric_name] = {
                "values": [metric_value],
                "steps": [env_step],
                "agg_fn": agg_fn,
            }
        else:
            self.data[metric_name]["values"].append(metric_value)
            self.data[metric_name]["steps"].append(env_step)

    def compute_aggregated_metrics(self) -> dict:
        aggregated_metrics = {}
        for metric_name in self.data.keys():
            values = self.data[metric_name]["values"]
            agg_fn = self.data[metric_name]["agg_fn"]
            aggregated_metrics[metric_name] = {
                "value": agg_fn(values),
                "step": self.data[metric_name]["steps"][-1],
            }
        self.data = {}
        return aggregated_metrics
