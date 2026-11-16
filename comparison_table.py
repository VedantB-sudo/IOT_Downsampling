
import argparse
from pathlib import Path
import pandas as pd


LABELS = {
    "uniform": "Uniform Decimation",
    "std_dev": "Standard Deviation",
    "hp_pe": "HP-PE",
    "hp_energy": "HP-Energy",
    "hybrid": "Hybrid"
}

COLUMNS = {
    "algorithm": "Algorithm",
    "data_reduction_pct": "Reduction (%)",
    "RMSE": "RMSE",
    "MAE": "MAE",
    "PRDN_pct": "PRDN (%)",
    "P95_retained_pct": "P95 Retained (%)",
    "P99_retained_pct": "P99 Retained (%)",
    "selection_time_ms": "Time (ms)"
}


def make_markdown(df):
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |"
    ]

    for row in df.itertuples(index=False, name=None):
        values = [
            f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


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
        parser.error(f"Results not found: {source}")

    df = pd.read_csv(source)

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        parser.error(f"Missing columns: {missing}")

    df["algorithm"] = df["algorithm"].replace(LABELS)
    df = df.sort_values(["sensor_type", "target_reduction_pct", "algorithm"])

    table = df[list(COLUMNS)].rename(columns=COLUMNS)
    table = table.round(3)
    table.to_csv(folder / "comparison_table.csv", index=False)

    sections = ["# PAMAP2 Downsampling Comparison", ""]

    for (sensor, reduction), group in df.groupby(
        ["sensor_type", "target_reduction_pct"], sort=True
    ):
        sections.append(f"## {sensor} - {reduction}% Reduction")
        sections.append("")
        sections.append(make_markdown(
            group[list(COLUMNS)].rename(columns=COLUMNS).round(3)
        ))
        sections.append("")

        best = group.loc[group["PRDN_pct"].idxmin(), "algorithm"]
        sections.append(f"Lowest PRDN: {best}")
        sections.append("")

    sections.append(
        "P95/P99 show exact retention of high-change readings, "
        "not labelled event detection."
    )

    output = folder / "comparison_table.md"
    output.write_text("\n".join(sections), encoding="utf-8")

    print(f"CSV saved: {folder / 'comparison_table.csv'}")
    print(f"Markdown saved: {output}")


if __name__ == "__main__":
    main()