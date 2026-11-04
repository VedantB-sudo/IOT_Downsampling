
"""
Create a six-metric Excel comparison report
and a four-panel graph.

Input:
    benchmark_comparison.csv

Outputs:
    comparison_report.xlsx
    comparison_graph.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlsxwriter


# ============================================================
# 1. ALGORITHMS AND METRICS
# ============================================================

ALGORITHMS = {
    "uniform": (
        "Uniform Decimation",
        "#2563EB"
    ),
    "std_dev": (
        "Standard Deviation",
        "#64748B"
    ),
    "hp_pe": (
        "HP-PE",
        "#0D9488"
    ),
    "hp_energy": (
        "HP-Energy",
        "#D97706"
    ),
    "hybrid": (
        "Hybrid HP-PE + HP-Energy",
        "#9333EA"
    )
}

METRICS = [
    (
        "data_reduction_pct",
        "Data Reduction (%)",
        "higher"
    ),
    (
        "important_data_retained_pct",
        "Important Data Retained (%)",
        "higher"
    ),
    (
        "event_recall_pct",
        "Synthetic Event Recall (%)",
        "higher"
    ),
    (
        "F2_score",
        "Synthetic F2-Score",
        "higher"
    ),
    (
        "PRDN_pct",
        "PRDN (%)",
        "lower"
    ),
    (
        "selection_time_ms",
        "Selection Time (ms)",
        "lower"
    )
]


# ============================================================
# 2. READ RESULTS
# ============================================================

def read_results(folder):

    source = folder / "benchmark_comparison.csv"

    if not source.is_file():
        raise FileNotFoundError(
            f"Cannot find {source}. "
            "Run benchmark_all.py first."
        )

    df = pd.read_csv(source)

    expected = (
        ["algorithm"]
        + [key for key, _, _ in METRICS]
    )

    missing = [
        col for col in expected
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "CSV is missing the six-metric columns: "
            + ", ".join(missing)
        )

    df = df[
        df["algorithm"].isin(ALGORITHMS)
    ].copy()

    if df.empty:
        raise ValueError(
            "No expected algorithm names found in the CSV."
        )

    for metric, _, _ in METRICS:
        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce"
        )

    essential = [
        "data_reduction_pct",
        "important_data_retained_pct",
        "PRDN_pct",
        "selection_time_ms"
    ]

    if df[essential].isna().any().any():
        raise ValueError(
            "Missing required numeric benchmark measurements."
        )

    df["reduction_group"] = (
        df["data_reduction_pct"].round(2)
    )

    df["order"] = df["algorithm"].map({
        code: i
        for i, code in enumerate(ALGORITHMS)
    })

    if df.duplicated(
        ["algorithm", "reduction_group"]
    ).any():
        raise ValueError(
            "Duplicate algorithm/reduction combinations found."
        )

    return df.sort_values(
        ["reduction_group", "order"]
    ).reset_index(drop=True)


# ============================================================
# 3. GENERATE FOUR COMPARISON GRAPHS
# ============================================================

def create_image(df, path):

    plt.rcParams.update({
        "font.size": 10
    })

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.5, 8.5)
    )

    panels = [
        (
            "important_data_retained_pct",
            "Important Readings Preserved",
            "Retained Important Data (%)"
        ),
        (
            "F2_score",
            "Synthetic Event F2-Score",
            "F2-Score (0-1)"
        ),
        (
            "PRDN_pct",
            "Reconstruction Distortion",
            "PRDN (%)"
        ),
        (
            "selection_time_ms",
            "Algorithm Processing Time",
            "Selection Time (ms)"
        )
    ]

    reductions = sorted(
        df["reduction_group"].unique()
    )

    for ax, (
        metric,
        title,
        y_label
    ) in zip(axes.flat, panels):

        ax.set_title(
            title,
            fontsize=12,
            weight="bold",
            loc="left"
        )

        ax.set_xlabel(
            "Data Removed (%)"
        )

        ax.set_ylabel(y_label)

        ax.set_xticks(reductions)

        ax.set_xlim(
            min(reductions) - 2,
            max(reductions) + 2
        )

        ax.grid(alpha=0.20)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        available = False

        for code, (name, color) in ALGORITHMS.items():

            part = df.loc[
                df["algorithm"] == code
            ].sort_values(
                "reduction_group"
            )

            if part[metric].notna().any():

                available = True

                ax.plot(
                    part["reduction_group"],
                    part[metric],
                    marker="o",
                    linewidth=2.2,
                    markersize=6,
                    color=color,
                    label=name
                )

        if not available:

            ax.text(
                0.5,
                0.50,
                "Not available - run with "
                "--synthetic-events 12",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#64748B"
            )

        ax.set_ylim(bottom=0)

        if metric == "F2_score":
            ax.set_ylim(0, 1.05)

        if metric == "important_data_retained_pct":
            ax.set_ylim(0, 105)

    # Shared algorithm legend.
    legend_handles = []
    legend_labels = []

    for ax in axes.flat:

        handles, labels = ax.get_legend_handles_labels()

        if handles:
            legend_handles = handles
            legend_labels = labels
            break

    fig.suptitle(
        "IoT Downsampling | Six-Metric Evaluation",
        fontsize=18,
        weight="bold",
        y=0.975
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.035),
        frameon=False
    )

    fig.text(
        0.5,
        0.01,
        "Important readings = top 5% absolute changes | "
        "F2 uses injected events only | "
        "No network measurements",
        ha="center",
        color="#64748B",
        fontsize=9
    )

    fig.tight_layout(
        rect=[0.02, 0.12, 0.98, 0.935],
        w_pad=2,
        h_pad=2
    )

    fig.savefig(
        path,
        dpi=190,
        facecolor="white"
    )

    plt.close(fig)


# ============================================================
# 4. CREATE EXCEL WORKBOOK
# ============================================================

def create_workbook(df, plot_path, path):

    book = xlsxwriter.Workbook(
        str(path)
    )

    book.set_properties({
        "title": "IoT Downsampling | Six-Metric Comparison"
    })

    # ========================================================
    # EXCEL STYLES
    # ========================================================

    dark = "#17324F"
    teal = "#087D8B"

    title = book.add_format({
        "bold": True,
        "font_size": 18,
        "font_color": "#FFFFFF",
        "bg_color": dark,
        "valign": "vcenter"
    })

    header = book.add_format({
        "bold": True,
        "font_color": "#FFFFFF",
        "bg_color": teal,
        "text_wrap": True,
        "valign": "vcenter"
    })

    heading = book.add_format({
        "bold": True,
        "font_size": 12,
        "font_color": dark
    })

    note = book.add_format({
        "font_color": "#475569",
        "italic": True,
        "text_wrap": True,
        "valign": "vcenter"
    })

    text_fmt = book.add_format({
        "font_color": "#1F2937"
    })

    number = book.add_format({
        "num_format": "0.000"
    })

    percentage = book.add_format({
        "num_format": '0.0"%"'
    })

    green = book.add_format({
        "bold": True,
        "bg_color": "#DDF2E6",
        "font_color": "#166534",
        "num_format": "0.000"
    })

    blue = book.add_format({
        "bold": True,
        "bg_color": "#DBEAFE",
        "font_color": "#1D4ED8",
        "num_format": "0.000"
    })

    amber = book.add_format({
        "bold": True,
        "bg_color": "#FEF3C7",
        "font_color": "#92400E",
        "num_format": "0.000"
    })

    # ========================================================
    # SHEET 1: SUMMARY
    # ========================================================

    summary = book.add_worksheet(
        "Summary"
    )

    summary.hide_gridlines(2)
    summary.set_tab_color(teal)

    summary.set_column("A:A", 18)
    summary.set_column("B:E", 35)

    summary.set_row(0, 30)

    summary.merge_range(
        "A1:E2",
        "DOWNSAMPLING | SUMMARY",
        title
    )

    summary.merge_range(
        "A3:E3",
        "Each row compares algorithms at the same "
        "data-reduction percentage.",
        note
    )

    summary_headers = [
        "Data Removed (%)",
        "Most Important Data Kept",
        "Best Event F2",
        "Lowest Distortion (PRDN)",
        "Fastest Selection"
    ]

    summary.write_row(
        5,
        0,
        summary_headers,
        header
    )

    summary.set_row(5, 36)

    has_events = df["F2_score"].notna().any()

    for row, (
        reduction,
        group
    ) in enumerate(
        df.groupby("reduction_group", sort=True),
        start=6
    ):

        # Best important-data retention.
        important_winner = group.loc[
            group["important_data_retained_pct"].idxmax()
        ]

        # Lowest PRDN.
        distortion_winner = group.loc[
            group["PRDN_pct"].idxmin()
        ]

        # Fastest algorithm.
        fastest = group.loc[
            group["selection_time_ms"].idxmin()
        ]

        summary.write_number(
            row,
            0,
            float(reduction),
            percentage
        )

        summary.write(
            row,
            1,
            ALGORITHMS[
                important_winner["algorithm"]
            ][0],
            text_fmt
        )

        # Best synthetic F2.
        f2_rows = group.dropna(
            subset=["F2_score"]
        )

        if not f2_rows.empty:

            f2_winner = f2_rows.loc[
                f2_rows["F2_score"].idxmax()
            ]

            summary.write(
                row,
                2,
                ALGORITHMS[
                    f2_winner["algorithm"]
                ][0],
                text_fmt
            )

        else:

            summary.write(
                row,
                2,
                "Not measured",
                note
            )

        summary.write(
            row,
            3,
            ALGORITHMS[
                distortion_winner["algorithm"]
            ][0],
            text_fmt
        )

        summary.write(
            row,
            4,
            ALGORITHMS[
                fastest["algorithm"]
            ][0],
            text_fmt
        )

    # Explanation.
    summary.write(
        "A12",
        "HOW TO INTERPRET",
        heading
    )

    summary.merge_range(
        "A13:E14",
        "Important Data Retained: fraction of readings "
        "in the top 5% of absolute changes that were kept. "
        "This is a proxy, not labelled faults.",
        note
    )

    summary.merge_range(
        "A16:E17",
        "Event Recall and F2: measured using injected "
        "labelled events and a fixed detector. "
        "They are not real machine-fault scores.",
        note
    )

    summary.merge_range(
        "A19:E20",
        "PRDN and processing time: lower is better. "
        "Processing time measures offline algorithm "
        "selection, not network latency.",
        note
    )

    if not has_events:

        summary.merge_range(
            "A22:E23",
            "To populate Event Recall and F2, rerun "
            "benchmark_all.py with --synthetic-events 12.",
            note
        )

    # ========================================================
    # SHEET 2: DETAILED RESULTS
    # ========================================================

    details = book.add_worksheet(
        "Detailed Results"
    )

    details.hide_gridlines(2)
    details.set_tab_color("#64748B")

    details.freeze_panes(5, 2)

    details.set_column("A:A", 32)
    details.set_column("B:G", 22)

    details.merge_range(
        "A1:G2",
        "SIX-METRIC COMPARISON",
        title
    )

    details.merge_range(
        "A3:G3",
        "Green = best retention / lowest distortion; "
        "Amber = best event scores; "
        "Blue = fastest (within each reduction level).",
        note
    )

    headings = [
        "Algorithm",
        "Data Reduction (%)",
        "Important Data Retained (%)",
        "Event Recall (%)",
        "F2-Score",
        "PRDN (%)",
        "Selection Time (ms)"
    ]

    details.write_row(
        4,
        0,
        headings,
        header
    )

    details.set_row(4, 36)

    details.autofilter(
        4,
        0,
        4 + len(df),
        6
    )

    # Determine best values within each reduction level.
    group_best = {}

    for metric, _, direction in METRICS[1:]:

        grouped = df.groupby(
            "reduction_group"
        )[metric]

        group_best[metric] = grouped.transform(
            "max" if direction == "higher" else "min"
        )

    display = [
        key for key, _, _ in METRICS
    ]

    for idx, record in df.iterrows():

        out_row = 5 + idx

        details.write(
            out_row,
            0,
            ALGORITHMS[
                record["algorithm"]
            ][0],
            text_fmt
        )

        for col, metric in enumerate(
            display,
            start=1
        ):

            value = record[metric]

            if pd.isna(value):

                details.write(
                    out_row,
                    col,
                    "N/A",
                    note
                )

                continue

            if metric == "data_reduction_pct":

                chosen_fmt = percentage

            elif np.isclose(
                value,
                group_best[metric].iloc[idx],
                rtol=1e-9,
                atol=1e-12
            ):

                if metric == "selection_time_ms":
                    chosen_fmt = blue

                elif metric in (
                    "event_recall_pct",
                    "F2_score"
                ):
                    chosen_fmt = amber

                else:
                    chosen_fmt = green

            else:
                chosen_fmt = number

            details.write_number(
                out_row,
                col,
                float(value),
                chosen_fmt
            )

    # ========================================================
    # SHEET 3: GRAPHS
    # ========================================================

    graphs = book.add_worksheet(
        "Graphs"
    )

    graphs.hide_gridlines(2)
    graphs.set_tab_color("#9333EA")

    graphs.set_column(
        "A:N",
        13
    )

    graphs.merge_range(
        "A1:N2",
        "FOUR ESSENTIAL PERFORMANCE GRAPHS",
        title
    )

    graphs.insert_image(
        "A4",
        str(plot_path),
        {
            "x_scale": 0.67,
            "y_scale": 0.67
        }
    )

    # Save workbook.
    book.close()


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create the six-metric Excel and graph report"
        )
    )

    parser.add_argument(
        "--results",
        default=(
            "benchmark_results/"
            "intel_mote_1_temperature_synthetic_events_12"
        )
    )

    args = parser.parse_args()

    folder = Path(args.results)

    # Load benchmark CSV.
    data = read_results(folder)

    # Output paths.
    plot_file = folder / "comparison_graph.png"

    workbook_file = (
        folder / "comparison_report.xlsx"
    )

    # Create graph.
    create_image(
        data,
        plot_file
    )

    # Create Excel report.
    create_workbook(
        data,
        plot_file,
        workbook_file
    )

    print("\nREPORT GENERATED SUCCESSFULLY")

    print(
        f"\nExcel: {workbook_file.resolve()}"
    )

    print(
        f"Graph: {plot_file.resolve()}"
    )


if __name__ == "__main__":
    main()