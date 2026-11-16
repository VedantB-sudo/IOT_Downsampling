
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "uniform": "Uniform Decimation",
    "std_dev": "Standard Deviation",
    "hp_pe": "HP-PE",
    "hp_energy": "HP-Energy",
    "hybrid": "Hybrid"
}

METRICS = {
    "PRDN_pct": "PRDN (%)",
    "RMSE": "RMSE",
    "MAE": "MAE",
    "P95_retained_pct": "P95 Retained (%)",
    "P99_retained_pct": "P99 Retained (%)",
    "selection_time_ms": "Processing Time (ms)"
}


def make_plots(df, folder):
    for sensor, group in df.groupby("sensor_type"):
        for metric, label in METRICS.items():
            if metric not in group.columns:
                continue

            fig, ax = plt.subplots(figsize=(9, 5))

            for method, name in LABELS.items():
                rows = group[group["algorithm"] == method]
                rows = rows.sort_values("target_reduction_pct")

                if rows.empty:
                    continue

                ax.plot(
                    rows["target_reduction_pct"],
                    rows[metric],
                    marker="o",
                    linewidth=2,
                    label=name
                )

            ax.set_title(f"{sensor} - {label}")
            ax.set_xlabel("Data Reduction (%)")
            ax.set_ylabel(label)
            ax.set_xticks([50, 60, 70, 80])
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

            fig.tight_layout()
            fig.savefig(folder / f"{sensor}_{metric}.png", dpi=160)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="benchmark_results/pamap2/subject101/activity4"
    )
    args = parser.parse_args()

    folder = Path(args.results)
    source = folder / "benchmark_comparison.csv"

    if not source.exists():
        parser.error(f"Benchmark results not found: {source}")

    df = pd.read_csv(source)

    required = ["sensor_type", "algorithm", "target_reduction_pct"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        parser.error(f"Missing columns: {missing}")

    output = folder / "plots"
    output.mkdir(parents=True, exist_ok=True)

    make_plots(df, output)
    print(f"Graphs saved to: {output.resolve()}")


if __name__ == "__main__":
    main()