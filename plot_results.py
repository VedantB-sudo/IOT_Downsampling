
"""
Plotting module for Intel IoT downsampling benchmarks.

Reads:
    benchmark_comparison.csv
    original_signal.csv (if available)
    *_selected.csv

Generates:
    Metric comparison graphs
    Original vs reconstructed signal plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "uniform": "Uniform Decimation",
    "std_dev": "Standard Deviation",
    "hp_pe": "HP-PE",
    "hp_energy": "HP-Energy",
    "hybrid": "Hybrid HP-PE + HP-Energy"
}

METRICS = {
    "RMSE": "RMSE (lower is better)",
    "MAE": "MAE (lower is better)",
    "PRDN_pct": "PRDN (%) (lower is better)",
    "selection_time_ms": "Selection time (ms)",
    "synthetic_event_recall": "Synthetic event recall",
    "synthetic_F1": "Synthetic F1",
    "synthetic_F2": "Synthetic F2"
}


# ==========================================================
# 1. COMPARISON GRAPHS
# ==========================================================

def make_comparison_plots(df, output):

    for column, axis_label in METRICS.items():

        if column not in df.columns:
            continue

        frame = df.copy()

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce"
        )

        if frame[column].notna().sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))

        for method in LABELS:

            subset = frame.loc[
                frame["algorithm"] == method
            ].copy()

            subset["data_reduction_pct"] = pd.to_numeric(
                subset["data_reduction_pct"],
                errors="coerce"
            )

            subset = subset.sort_values(
                "data_reduction_pct"
            )

            if subset[column].notna().any():

                ax.plot(
                    subset["data_reduction_pct"],
                    subset[column],
                    marker="o",
                    linewidth=2,
                    label=LABELS[method]
                )

        ax.set(
            title=f"Downsampling Benchmark: {column}",
            xlabel="Data Reduction (%)",
            ylabel=axis_label
        )

        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        fig.tight_layout()

        fig.savefig(
            output / f"compare_{column}.png",
            dpi=160
        )

        plt.close(fig)

        print(f"Saved compare_{column}.png")


# ==========================================================
# 2. ORIGINAL VS RECONSTRUCTED SIGNAL PLOTS
# ==========================================================

def make_trace_plots(
    results,
    output,
    target_retained_pct,
    max_points
):

    raw_file = results / "original_signal.csv"

    if not raw_file.exists():
        print(
            "original_signal.csv not found. "
            "Skipping reconstruction plots."
        )
        return

    raw = pd.read_csv(
        raw_file,
        parse_dates=["timestamp"]
    )

    raw = raw.sort_values(
        "sample_index"
    ).reset_index(drop=True)

    if raw.empty:
        print("Original signal is empty.")
        return

    raw_times = (
        raw["timestamp"] - raw["timestamp"].iloc[0]
    ).dt.total_seconds().to_numpy()

    limit = min(max_points, len(raw))

    for method in LABELS:

        tag = (
            f"{method}_keep_"
            f"{int(round(target_retained_pct))}"
        )

        selected_file = (
            results / f"{tag}_selected.csv"
        )

        if not selected_file.exists():
            print(f"Skipping {method}: missing selected CSV.")
            continue

        kept = pd.read_csv(
            selected_file,
            parse_dates=["timestamp"]
        )

        kept = kept.sort_values("sample_index")

        kept_times = (
            kept["timestamp"] - raw["timestamp"].iloc[0]
        ).dt.total_seconds().to_numpy()

        reconstructed = np.interp(
            raw_times,
            kept_times,
            kept["value"].to_numpy(dtype=float)
        )

        shown = kept.loc[
            kept["sample_index"] < limit
        ]

        fig, ax = plt.subplots(figsize=(11, 4))

        ax.plot(
            raw["timestamp"].iloc[:limit],
            raw["value"].iloc[:limit],
            label="Original",
            linewidth=1.3
        )

        ax.plot(
            raw["timestamp"].iloc[:limit],
            reconstructed[:limit],
            label="Reconstructed",
            linewidth=1
        )

        ax.scatter(
            shown["timestamp"],
            shown["value"],
            s=12,
            label="Retained Samples"
        )

        ax.set(
            title=(
                f"{LABELS[method]} - "
                f"{target_retained_pct:g}% Retained"
            ),
            xlabel="Time",
            ylabel="Sensor Reading"
        )

        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        fig.autofmt_xdate()
        fig.tight_layout()

        fig.savefig(
            output / f"trace_{tag}.png",
            dpi=160
        )

        plt.close(fig)

        print(f"Saved trace_{tag}.png")


# ==========================================================
# 3. MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Plot downsampling benchmark results"
    )

    parser.add_argument(
        "--results",
        default="benchmark_results/intel_mote_1_temperature"
    )

    parser.add_argument(
        "--retained",
        type=float,
        default=20
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=400
    )

    args = parser.parse_args()

    results = Path(args.results)

    comparison_file = (
        results / "benchmark_comparison.csv"
    )

    if not comparison_file.is_file():
        parser.error(
            f"Benchmark CSV not found: {comparison_file}"
        )

    if args.max_points < 1 or not 0 < args.retained <= 100:
        parser.error("Invalid plotting settings.")

    df = pd.read_csv(comparison_file)

    output = results / "plots"

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    make_comparison_plots(df, output)

    make_trace_plots(
        results,
        output,
        args.retained,
        args.max_points
    )

    print(
        f"\nAll graphs saved to: {output.resolve()}"
    )


if __name__ == "__main__":
    main()