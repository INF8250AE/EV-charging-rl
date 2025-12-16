import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from matplotlib.ticker import FuncFormatter

METRIC_KEY = "train_ep_congestion_norm_mean"

FOLDER_PATH = f"./results/{METRIC_KEY}/"
OUTPUT_FOLDER_PATH = Path("./plots/")
PLOT_TITLE = "Training Congestion Mean"
Y_LABEL = "Episode Normalized Congestion Mean"
X_LABEL = "Training Steps"


SMOOTHING_WEIGHT = 0.9

METHODS_TO_PLOT = ["a2c", "ddqn"]  # "ga", "random"]
METHOD_COLORS = {"a2c": "#1f77b4", "ddqn": "#d62728", "ga": "#2ca02c", "random": "gray"}
METHOD_DISPLAY_NAMES = {
    "a2c": "A2C",
    "ddqn": "DDQN",
    "ga": "Evolutionary (GA)",
    "random": "Random Baseline",
}


def smooth_data(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def format_func(value, tick_number):
    return f"{int(value / 1000)}k"


def load_and_plot():
    plt.figure(figsize=(10, 6))

    for method in METHODS_TO_PLOT:

        # Assuming filename format: {method}_seed_{seed}_{METRIC_KEY}.json

        seeds_data_y = []
        seeds_data_x = []

        for seed in range(5):
            filename = f"{method}_seed_{seed}_{METRIC_KEY}.json"
            filepath = os.path.join(FOLDER_PATH, filename)

            with open(filepath, "r") as f:
                data = json.load(f)

            run_data = data[0]
            seeds_data_x.append(run_data["x"])
            seeds_data_y.append(run_data["y"])

        assert (
            len(seeds_data_y) == 5
        ), f"Expected 5 seeds for {method}, found {len(seeds_data_y)}"

        min_len = min(len(y) for y in seeds_data_y)

        seeds_data_y_truncated = [y[:min_len] for y in seeds_data_y]
        seeds_data_x_truncated = seeds_data_x[0][:min_len]

        x_axis = np.array(seeds_data_x_truncated)
        y_matrix = np.array(seeds_data_y_truncated)

        mean_y = np.mean(y_matrix, axis=0)
        std_y = np.std(y_matrix, axis=0)

        smoothed_mean = smooth_data(mean_y, SMOOTHING_WEIGHT)

        color = METHOD_COLORS.get(method, "black")
        label = METHOD_DISPLAY_NAMES.get(method, method.upper())

        plt.plot(x_axis, smoothed_mean, color=color, linewidth=2, label=label)

        plt.fill_between(
            x_axis,
            smoothed_mean - std_y,
            smoothed_mean + std_y,
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    plt.title(PLOT_TITLE, fontsize=14, fontweight="bold")
    plt.xlabel(X_LABEL, fontsize=12)
    plt.ylabel(Y_LABEL, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
    plt.tight_layout()

    OUTPUT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    output_file_path = OUTPUT_FOLDER_PATH / Path(f"{METRIC_KEY}.png")
    plt.savefig(output_file_path, dpi=300)
    print(f"Plot saved to {output_file_path}")


if __name__ == "__main__":
    load_and_plot()
