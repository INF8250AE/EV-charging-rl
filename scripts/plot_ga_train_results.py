import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from matplotlib.ticker import MaxNLocator

METRIC_KEY = "train_gen_return"

FOLDER_PATH = f"./results/{METRIC_KEY}/"
OUTPUT_FOLDER_PATH = Path("./plots/")
PLOT_TITLE = "Training Episode Return (\u2191 better)"
Y_LABEL = "Mean Episode Return"
X_LABEL = "Generations"

METHODS_TO_PLOT = ["ga"]
METHOD_COLORS = {"ga": "#2ca02c"}
METHOD_DISPLAY_NAMES = {"ga": "Evolutionary (GA)"}


def load_and_plot():
    plt.figure(figsize=(10, 6))

    for method in METHODS_TO_PLOT:
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

        color = METHOD_COLORS.get(method, "black")
        label = METHOD_DISPLAY_NAMES.get(method, method.upper())

        for i, seed_y in enumerate(seeds_data_y_truncated):
            plt.plot(
                x_axis, seed_y, color=color, alpha=0.25, linewidth=1, linestyle="-"
            )

        plt.plot(
            x_axis,
            mean_y,
            color=color,
            linewidth=2.5,
            label=f"{label}",
            marker="o",
        )

        plt.fill_between(
            x_axis,
            mean_y - std_y,
            mean_y + std_y,
            color=color,
            alpha=0.1,
            linewidth=0,
        )

    plt.title(PLOT_TITLE, fontsize=14, fontweight="bold")
    plt.xlabel(X_LABEL, fontsize=12)
    plt.ylabel(Y_LABEL, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    OUTPUT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    output_file_path = OUTPUT_FOLDER_PATH / Path(f"{METRIC_KEY}.png")
    plt.savefig(output_file_path, dpi=300)
    print(f"Plot saved to {output_file_path}")


if __name__ == "__main__":
    load_and_plot()
