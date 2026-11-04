
"""
Comparison table generator for IoT downsampling.

Reads:
    benchmark_comparison.csv

Generates:
    comparison_table.csv
    comparison_table.md
"""

import argparse
from pathlib import Path
import pandas as pd


ORDER = [
    "uniform",
    "std_dev",
    "hp_pe",
    "hp_energy",
    "hybrid"
]

LABELS = {
    "uniform": "Uniform Decimation",
    "std_dev": "Standard Deviation",
    "hp_pe": "HP-PE",
    "hp_energy": "HP-Energy",
    "hybrid": "Hybrid HP-PE + HP-Energy"
}

NUMERIC_COLS = [
    "target_retained_pct",
    "data_reduction_pct",
    "RMSE",
    "MAE",
    "PRDN_pct",
    "selection_time_ms",
    "synthetic_event_recall",
    "synthetic_F1",
    "synthetic_F2"
]


# ==========================================================
# 1. FORMAT MARKDOWN TABLE
# ==========================================================

def md_table(frame, cols, headers):

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            "---" for _ in headers
        ) + " |"
    ]

    for _, row in frame.iterrows():

        formatted = []

        for col in cols:

            value = row[col]

            if pd.isna(value):
                formatted.append("—")

            elif col in NUMERIC_COLS:
                formatted.append(
                    f"{float(value):.3f}"
                )

            else:
                formatted.append(str(value))

        lines.append(
            "| " + " | ".join(formatted) + " |"
        )

    return "\n".join(lines)


# ==========================================================
# 2. MAIN TABLE GENERATOR
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Create benchmark comparison tables"
    )

    parser.add_argument(
        "--results",
        default="benchmark_results/intel_mote_1_temperature"
    )

    args = parser.parse_args()

    results = Path(args.results)

    infile = results / "benchmark_comparison.csv"

    if not infile.is_file():
        parser.error(
            f"Missing benchmark results: {infile}"
        )

    df = pd.read_csv(infile)

    required = [
        "algorithm",
        "target_retained_pct",
        "data_reduction_pct",
        "RMSE",
        "MAE",
        "PRDN_pct",
        "selection_time_ms"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        parser.error(
            f"Missing columns: {', '.join(missing)}"
        )

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df["algorithm_name"] = (
        df["algorithm"]
        .map(LABELS)
        .fillna(df["algorithm"])
    )

    df["method_order"] = (
        df["algorithm"]
        .map({
            name: i
            for i, name in enumerate(ORDER)
        })
        .fillna(99)
    )

    df = df.sort_values(
        ["target_retained_pct", "method_order"],
        ascending=[False, True]
    )

    event_columns = [
        "synthetic_event_recall",
        "synthetic_F1",
        "synthetic_F2"
    ]

    include_event = any(
        col in df.columns and df[col].notna().any()
        for col in event_columns
    )

    cols = [
        "algorithm_name",
        "data_reduction_pct",
        "RMSE",
        "MAE",
        "PRDN_pct",
        "selection_time_ms"
    ]

    headers = [
        "Algorithm",
        "Reduction %",
        "RMSE",
        "MAE",
        "PRDN %",
        "Time (ms)"
    ]

    if include_event:

        for col, title in [
            ("synthetic_event_recall", "Synthetic Recall"),
            ("synthetic_F1", "Synthetic F1"),
            ("synthetic_F2", "Synthetic F2")
        ]:

            if col in df.columns:
                cols.append(col)
                headers.append(title)

    # ======================================================
    # EXPORT CSV
    # ======================================================

    export = df[
        ["target_retained_pct"] + cols
    ].copy()

    csv_out = results / "comparison_table.csv"

    export.to_csv(
        csv_out,
        index=False
    )

    # ======================================================
    # CREATE MARKDOWN TABLES
    # ======================================================

    sections = [
        "# IoT Downsampling Benchmark Comparison",
        "",
        "Lower RMSE, MAE, PRDN and processing time "
        "are better. Higher recall, F1 and F2 "
        "are better.",
        ""
    ]

    for retention, group in df.groupby(
        "target_retained_pct",
        sort=False
    ):

        sections.append(
            f"## {retention:g}% Retained "
            f"({100 - retention:g}% Target Reduction)"
        )

        sections.append("")

        sections.append(
            md_table(
                group,
                cols,
                headers
            )
        )

        sections.append("")

        best_rmse = group.loc[
            group["RMSE"].idxmin(),
            "algorithm_name"
        ]

        sections.append(
            f"**Lowest RMSE:** {best_rmse}."
        )

        if (
            "synthetic_F2" in group.columns
            and group["synthetic_F2"].notna().any()
        ):

            best_f2 = group.loc[
                group["synthetic_F2"].idxmax(),
                "algorithm_name"
            ]

            sections.append(
                f"**Highest Synthetic F2:** {best_f2}."
            )

        sections.append("")

    sections.append(
        "*Synthetic event scores are proxy measurements "
        "using injected events and a simple detector, "
        "not real downstream classifier performance. "
        "No MQTT or PID results are included.*"
    )

    markdown_out = (
        results / "comparison_table.md"
    )

    markdown_out.write_text(
        "\n".join(sections) + "\n",
        encoding="utf-8"
    )

    print("\n".join(sections))

    print(
        f"\nCSV saved to: {csv_out.resolve()}"
    )

    print(
        f"Markdown saved to: {markdown_out.resolve()}"
    )


if __name__ == "__main__":
    main()