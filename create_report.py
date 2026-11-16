
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
    "hybrid": "Hybrid"
}

METRICS = {
    "PRDN_pct_mean": "PRDN (%)",
    "P99_retained_pct_mean": "P99 Retained (%)"
}


def load_results(folder):
    summary_file = folder / "benchmark_summary.csv"
    activity_file = folder / "benchmark_by_activity.csv"

    if not summary_file.exists():
        raise FileNotFoundError(f"Missing file: {summary_file}")

    summary = pd.read_csv(summary_file)

    required = [
        "sensor_type", "target_reduction_pct", "algorithm",
        "PRDN_pct_mean", "PRDN_pct_std",
        "RMSE_mean", "MAE_mean",
        "P95_retained_pct_mean", "P99_retained_pct_mean",
        "selection_time_ms_mean", "window_count"
    ]

    missing = [col for col in required if col not in summary.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    activity = (
        pd.read_csv(activity_file)
        if activity_file.exists()
        else pd.DataFrame()
    )

    return summary, activity


def overall_results(df):
    rows = []

    for (method, reduction), group in df.groupby(
        ["algorithm", "target_reduction_pct"]
    ):
        counts = group["window_count"].to_numpy(dtype=float)

        row = {
            "Algorithm": LABELS.get(method, method),
            "Reduction (%)": reduction,
            "PRDN (%)": np.average(group["PRDN_pct_mean"], weights=counts),
            "P95 Retained (%)": np.average(
                group["P95_retained_pct_mean"], weights=counts
            ),
            "P99 Retained (%)": np.average(
                group["P99_retained_pct_mean"], weights=counts
            ),
            "Time (ms)": np.average(
                group["selection_time_ms_mean"], weights=counts
            ),
            "Sensor Channels": len(group)
        }

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["Reduction (%)", "Algorithm"]
    )


def make_overview(df):
    rows = []

    for (sensor, reduction), group in df.groupby(
        ["sensor_type", "target_reduction_pct"]
    ):
        def best(column, highest=False):
            index = group[column].idxmax() if highest else group[column].idxmin()
            return LABELS.get(group.loc[index, "algorithm"],
                              group.loc[index, "algorithm"])

        rows.append({
            "Sensor": sensor,
            "Reduction (%)": reduction,
            "Lowest PRDN": best("PRDN_pct_mean"),
            "Best P95": best("P95_retained_pct_mean", True),
            "Best P99": best("P99_retained_pct_mean", True),
            "Fastest": best("selection_time_ms_mean")
        })

    return pd.DataFrame(rows)


def make_graphs(df, folder):
    plots = folder / "plots"
    plots.mkdir(exist_ok=True)

    for sensor, group in df.groupby("sensor_type"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for ax, (metric, title) in zip(axes, METRICS.items()):
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

            ax.set_title(title)
            ax.set_xlabel("Data Reduction (%)")
            ax.set_ylabel(title)
            ax.set_xticks(sorted(group["target_reduction_pct"].unique()))
            ax.grid(alpha=0.3)

        axes[0].legend(fontsize=8)
        fig.suptitle(f"PAMAP2 - {sensor}")
        fig.tight_layout()

        fig.savefig(plots / f"{sensor}_comparison.png", dpi=160)
        plt.close(fig)

    return plots


def format_sheet(writer, name, df):
    sheet = writer.sheets[name]
    workbook = writer.book

    header = workbook.add_format({
        "bold": True,
        "font_color": "white",
        "bg_color": "#17324F",
        "border": 1,
        "text_wrap": True
    })

    number = workbook.add_format({"num_format": "0.000"})

    for col, title in enumerate(df.columns):
        sheet.write(0, col, title, header)

        width = min(max(len(str(title)) + 4, 17), 35)
        sheet.set_column(col, col, width)

        if pd.api.types.is_numeric_dtype(df.iloc[:, col]):
            sheet.set_column(col, col, width, number)

    sheet.set_row(0, 30)
    sheet.freeze_panes(1, 0)
    sheet.hide_gridlines(2)

    if not df.empty:
        sheet.autofilter(0, 0, len(df), len(df.columns) - 1)


def create_excel(summary, activity, plots, folder):
    overview = make_overview(summary)
    overall = overall_results(summary)

    details = summary.copy()
    details["algorithm"] = details["algorithm"].replace(LABELS)
    details = details.sort_values(
        ["sensor_type", "target_reduction_pct", "algorithm"]
    )

    if not activity.empty and "algorithm" in activity.columns:
        activity = activity.copy()
        activity["algorithm"] = activity["algorithm"].replace(LABELS)

    output = folder / "full_dataset_report.xlsx"

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        sheets = {
            "Overview": overview,
            "Overall Comparison": overall,
            "Sensor Results": details
        }

        if not activity.empty:
            sheets["Activity Results"] = activity

        for name, frame in sheets.items():
            frame.round(3).to_excel(writer, sheet_name=name, index=False)
            format_sheet(writer, name, frame)

        workbook = writer.book
        graphs = workbook.add_worksheet("Graphs")
        graphs.hide_gridlines(2)
        graphs.set_column("A:U", 12)

        heading = workbook.add_format({
            "bold": True,
            "font_size": 16,
            "font_color": "#17324F"
        })

        graphs.write("B2", "PAMAP2 Downsampling Comparison", heading)

        row = 4

        for sensor in sorted(summary["sensor_type"].unique()):
            image = plots / f"{sensor}_comparison.png"
            graphs.insert_image(row, 1, str(image), {
                "x_scale": 0.75,
                "y_scale": 0.75
            })
            row += 33

        notes = workbook.add_worksheet("Notes")
        notes.set_column("A:A", 28)
        notes.set_column("B:B", 100)

        notes.write_row(0, 0, ["Metric", "Meaning"])

        explanations = [
            ("PRDN", "Mean-normalised reconstruction error; lower is better."),
            ("RMSE", "Root mean squared reconstruction error."),
            ("MAE", "Mean absolute reconstruction error."),
            ("P95/P99", "Exact-index retention of high-change readings, not labelled event recall."),
            ("Processing Time", "Offline algorithm selection time, not network latency."),
            ("Overall Comparison", "Window-count-weighted average across sensor channels."),
            ("Dataset", "PAMAP2 Protocol and Optional recordings; eligible activity windows."),
            ("Network", "No MQTT or real network measurements.")
        ]

        for row, item in enumerate(explanations, start=1):
            notes.write_row(row, 0, item)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="benchmark_results/pamap2/full_dataset"
    )
    args = parser.parse_args()

    folder = Path(args.results)
    summary, activity = load_results(folder)

    plots = make_graphs(summary, folder)
    report = create_excel(summary, activity, plots, folder)

    print("\nFULL DATASET REPORT CREATED")
    print(f"Excel: {report.resolve()}")
    print(f"Graphs: {plots.resolve()}")
    print(f"Summary rows: {len(summary)}")


if __name__ == "__main__":
    main()