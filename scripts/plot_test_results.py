import json
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import os

ROOT_DIR = "./results/test_in_distribution/"
OUTPUT_DIR = "./plots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METHODS = ["a2c", "ddqn", "ga", "random"]

METHOD_COLORS = {"a2c": "#1f77b4", "ddqn": "#d62728", "ga": "#2ca02c", "random": "gray"}
METHOD_DISPLAY_NAMES = {
    "a2c": "A2C",
    "ddqn": "DDQN",
    "ga": "Evolutionary (GA)",
    "random": "Random Baseline",
}

METRICS_CONFIG = {
    "episode_return": {
        "title": "Test Episode Return (\u2191 better)",
        "ylabel": "IQM Episode Return",
        "json_path": ["aggregate_metrics", "episode_return", "iqm"],
    },
    "received_station_full_penalty": {
        "title": "Test Station Full Penalty (\u2193 better)",
        "ylabel": "IQM Episode Penalty Count",
        "json_path": [
            "aggregate_metrics",
            "detailed_info_metrics",
            "received_station_full_penalty",
            "sum",
            "iqm",
        ],
    },
    "congestion_norm": {
        "title": "Test Episode Mean Congestion (\u2193 better)",
        "ylabel": "IQM Episode Mean Normalized Congestion",
        "json_path": [
            "aggregate_metrics",
            "detailed_info_metrics",
            "congestion_norm",
            "mean",
            "iqm",
        ],
    },
    # "nb_cars_waiting": {
    #     "title": "Test Episode Mean Nb Cars Waiting (\u2193 better)",
    #     "ylabel": "IQM Episode Mean Nb Cars Waiting",
    #     "json_path": [
    #         "aggregate_metrics",
    #         "detailed_info_metrics",
    #         "nb_cars_waiting",
    #         "mean",
    #         "iqm",
    #     ],
    # },
    # "nb_cars_traveling": {
    #     "title": "Test Episode Mean Nb Cars Traveling (\u2193 better)",
    #     "ylabel": "IQM Episode Mean Nb Cars Traveling",
    #     "json_path": [
    #         "aggregate_metrics",
    #         "detailed_info_metrics",
    #         "nb_cars_traveling",
    #         "mean",
    #         "iqm",
    #     ],
    # },
}


def get_value_from_path(data, path):
    curr = data
    for key in path:
        curr = curr[key]
    return curr


def plot_test_results():
    for metric_name, config in METRICS_CONFIG.items():
        plt.figure(figsize=(8, 6))

        bar_names = []
        bar_means = []
        bar_colors = []
        yerr_lower = []
        yerr_upper = []

        random_data = None

        for method in METHODS:
            file_path = os.path.join(ROOT_DIR, f"{method}_evaluation_results.json")

            if not os.path.exists(file_path):
                continue

            with open(file_path, "r") as f:
                data = json.load(f)

            iqm_data = get_value_from_path(data, config["json_path"])

            mean_val = iqm_data["value"]
            ci_lower = iqm_data["ci_lower"]
            ci_upper = iqm_data["ci_upper"]

            if method == "random":
                random_data = {"mean": mean_val, "lower": ci_lower, "upper": ci_upper}
            else:
                bar_names.append(METHOD_DISPLAY_NAMES.get(method, method.upper()))
                bar_means.append(mean_val)
                bar_colors.append(METHOD_COLORS.get(method, "blue"))

                yerr_lower.append(mean_val - ci_lower)
                yerr_upper.append(ci_upper - mean_val)

        if bar_names:
            x_pos = np.arange(len(bar_names))

            errors = np.array([yerr_lower, yerr_upper])

            bars = plt.bar(
                x_pos,
                bar_means,
                yerr=errors,
                align="center",
                alpha=0.85,
                color=bar_colors,
                capsize=8,
                edgecolor="black",
                linewidth=1.2,
            )

            plt.xticks(x_pos, bar_names, fontsize=12, fontweight="bold")

            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height / 2,
                    f"{height:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=11,
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                )

        if random_data:
            plt.axhline(
                y=random_data["mean"],
                color=METHOD_COLORS["random"],
                linestyle="--",
                linewidth=2,
                label="Random Baseline",
            )
            plt.axhspan(
                random_data["lower"],
                random_data["upper"],
                color=METHOD_COLORS["random"],
                alpha=0.15,
            )
            plt.legend(loc="best")

        plt.title(config["title"], fontsize=14, fontweight="bold")
        plt.ylabel(config["ylabel"], fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, f"test_{metric_name}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    plot_test_results()
