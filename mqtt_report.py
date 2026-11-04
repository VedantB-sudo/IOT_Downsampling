
"""
MQTT Comparison Report

Uses the existing Vicomtech MQTT experiment results.

Outputs:
    mqtt_results/comparison/mqtt_comparison.xlsx
    mqtt_results/comparison/mqtt_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xlsxwriter


# ============================================================
# 1. MQTT EXPERIMENT RUN IDs
# ============================================================

# These are your six experiments using fair replay timing.

RUNS = [
    ("raw", "Raw (No Filtering)", "2dbcc6e89d"),
    ("uniform", "Uniform Decimation", "4c56d8d082"),
    ("std_dev", "Standard Deviation", "9d8fefbdf3"),
    ("hp_pe", "HP-PE", "c14250be33"),
    ("hp_energy", "HP-Energy", "4df8b99920"),
    ("hybrid", "Hybrid HP-PE + Energy", "8ac5731a82"),
]

COLORS = [
    "#334155",
    "#2563EB",
    "#64748B",
    "#0D9488",
    "#D97706",
    "#9333EA"
]


# ============================================================
# 2. LOAD MQTT RESULTS
# ============================================================

def load_runs(root):

    records = []

    for code, label, run_id in RUNS:

        path = root / run_id / "summary.json"

        if not path.is_file():

            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Check that the experiment folder exists."
            )

        with path.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        # Validate algorithm name.
        if data.get("algorithm") != code:

            raise ValueError(
                f"Run {run_id} contains "
                f"{data.get('algorithm')}, "
                f"but expected {code}."
            )

        expected = int(
            data["messages_expected"]
        )

        received = int(
            data["messages_received"]
        )

        payload = int(
            data["payload_bytes_received"]
        )

        if expected <= 0 or received < 0 or payload < 0:
            raise ValueError(
                f"Invalid totals in {path}"
            )

        records.append({
            "code": code,
            "label": label,
            "run_id": run_id,
            "expected": expected,
            "received": received,
            "missing": int(
                data["missing_messages"]
            ),
            "duplicates": int(
                data["duplicate_messages"]
            ),
            "payload": payload,
            "delay_ms": data.get(
                "average_delivery_delay_ms"
            ),
            "rate": data.get(
                "reception_rate_messages_per_second"
            )
        })

    # Use raw transmission as baseline.
    raw_bytes = records[0]["payload"]

    if raw_bytes <= 0:
        raise ValueError(
            "Raw baseline contains no payload bytes."
        )

    for record in records:

        record["success_pct"] = (
            100
            * record["received"]
            / record["expected"]
        )

        record["savings_pct"] = (
            100
            * (1 - record["payload"] / raw_bytes)
        )

    return records


# ============================================================
# 3. GENERATE MQTT COMPARISON GRAPHS
# ============================================================

def create_chart(records, output):

    labels = [
        "Raw",
        "Uniform",
        "Std Dev",
        "HP-PE",
        "HP-Energy",
        "Hybrid"
    ]

    panels = [
        (
            "payload",
            "MQTT Payload Received",
            "Bytes",
            "{:,.0f}"
        ),
        (
            "savings_pct",
            "Payload Savings Compared with Raw",
            "Savings (%)",
            "{:.1f}%"
        ),
        (
            "delay_ms",
            "Average MQTT Delivery Delay",
            "Milliseconds",
            "{:.3f}"
        ),
        (
            "success_pct",
            "Message Delivery Success",
            "Received / Expected (%)",
            "{:.1f}%"
        )
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9)
    )

    for ax, (
        metric,
        title,
        ylabel,
        display
    ) in zip(axes.flat, panels):

        values = [
            float(record[metric])
            if record[metric] is not None
            else 0.0
            for record in records
        ]

        bars = ax.bar(
            range(len(values)),
            values,
            color=COLORS,
            width=0.65
        )

        ax.set_title(
            title,
            loc="left",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_ylabel(ylabel)

        ax.set_xticks(
            range(len(labels)),
            labels
        )

        ax.tick_params(
            axis="x",
            rotation=20
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="y",
            alpha=0.2
        )

        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)

        # Display values above each bar.
        for bar, value in zip(bars, values):

            ax.annotate(
                display.format(value),
                (
                    bar.get_x()
                    + bar.get_width() / 2,
                    bar.get_height()
                ),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 3),
                textcoords="offset points"
            )

        ax.margins(y=0.15)

    fig.suptitle(
        "MQTT Transmission Comparison | Intel Sensor 1",
        fontsize=18,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.017,
        "100 original temperature readings | "
        "0.2 s replay interval | "
        "80% reduction for filtered algorithms | "
        "Local MQTT experiments",
        ha="center",
        fontsize=9,
        color="#64748B"
    )

    fig.tight_layout(
        rect=[0, 0.045, 1, 0.94],
        h_pad=2.2,
        w_pad=2.2
    )

    fig.savefig(
        output,
        dpi=190,
        facecolor="white"
    )

    plt.close(fig)


# ============================================================
# 4. GENERATE EXCEL REPORT
# ============================================================

def create_excel(records, chart_file, output):

    workbook = xlsxwriter.Workbook(
        str(output)
    )

    # --------------------------------------------------------
    # FORMATTING
    # --------------------------------------------------------

    title_format = workbook.add_format({
        "bold": True,
        "font_size": 17,
        "font_color": "white",
        "bg_color": "#17324F",
        "valign": "vcenter"
    })

    header_format = workbook.add_format({
        "bold": True,
        "font_color": "white",
        "bg_color": "#087D8B",
        "text_wrap": True,
        "valign": "vcenter"
    })

    note_format = workbook.add_format({
        "font_color": "#475569",
        "italic": True,
        "text_wrap": True,
        "valign": "vcenter"
    })

    text_format = workbook.add_format({
        "font_color": "#263445"
    })

    number_format = workbook.add_format({
        "num_format": "#,##0"
    })

    decimal_format = workbook.add_format({
        "num_format": "0.000"
    })

    percent_format = workbook.add_format({
        "num_format": '0.0"%"'
    })

    green_format = workbook.add_format({
        "bg_color": "#DDF2E6",
        "font_color": "#166534",
        "bold": True,
        "num_format": '0.0"%"'
    })

    # ========================================================
    # SHEET 1: SUMMARY
    # ========================================================

    summary = workbook.add_worksheet(
        "Summary"
    )

    summary.hide_gridlines(2)

    summary.set_tab_color("#087D8B")

    summary.set_column("A:A", 27)
    summary.set_column("B:B", 24)
    summary.set_column("C:C", 4)
    summary.set_column("D:D", 48)

    summary.merge_range(
        "A1:D2",
        "MQTT EXPERIMENT | RESULTS SUMMARY",
        title_format
    )

    summary.write(
        "A4",
        "Raw Messages Received",
        text_format
    )

    summary.write_number(
        "B4",
        records[0]["received"],
        number_format
    )

    summary.write(
        "A5",
        "Raw Payload Bytes",
        text_format
    )

    summary.write_number(
        "B5",
        records[0]["payload"],
        number_format
    )

    summary.write(
        "A6",
        "Hybrid Messages Received",
        text_format
    )

    summary.write_number(
        "B6",
        records[-1]["received"],
        number_format
    )

    summary.write(
        "A7",
        "Hybrid Payload Savings",
        text_format
    )

    summary.write_number(
        "B7",
        records[-1]["savings_pct"],
        green_format
    )

    summary.merge_range(
        "D4:D5",
        "All algorithms were evaluated using the same "
        "100 original sensor readings and approximately "
        "the same 19.8-second replay duration.",
        note_format
    )

    summary.merge_range(
        "D6:D8",
        "All filtered methods transmitted 20 readings. "
        "Small differences in payload bytes reflect "
        "JSON message lengths, not better compression.",
        note_format
    )

    summary.merge_range(
        "A10:D11",
        "IMPORTANT: These results measure local MQTT "
        "application-payload transmission. They do not "
        "include MQTT/TCP protocol overhead, network "
        "congestion or PID adaptation. Delivery-delay "
        "differences from single runs are not statistically "
        "conclusive.",
        note_format
    )

    # ========================================================
    # SHEET 2: DETAILED RESULTS
    # ========================================================

    detail = workbook.add_worksheet(
        "Detailed Results"
    )

    detail.hide_gridlines(2)

    detail.freeze_panes(6, 1)

    detail.set_tab_color("#64748B")

    detail.set_column("A:A", 27)
    detail.set_column("B:F", 16)
    detail.set_column("G:J", 19)

    detail.merge_range(
        "A1:J2",
        "MQTT | SIX EXPERIMENTS",
        title_format
    )

    detail.merge_range(
        "A3:J3",
        "Measured at the consumer. "
        "Payload savings are calculated against Raw.",
        note_format
    )

    headers = [
        "Algorithm",
        "Expected Messages",
        "Received Messages",
        "Missing Messages",
        "Duplicate Messages",
        "Delivery Success (%)",
        "Payload Bytes",
        "Payload Savings (%)",
        "Average Delay (ms)",
        "Reception Rate (msg/s)"
    ]

    detail.write_row(
        5,
        0,
        headers,
        header_format
    )

    detail.set_row(5, 38)

    for i, record in enumerate(records):

        row = 6 + i
        excel_row = row + 1

        detail.write(
            row,
            0,
            record["label"],
            text_format
        )

        for col, key in enumerate([
            "expected",
            "received",
            "missing",
            "duplicates"
        ], start=1):

            detail.write_number(
                row,
                col,
                record[key],
                number_format
            )

        # Delivery success percentage.
        detail.write_formula(
            row,
            5,
            f"=IFERROR(C{excel_row}/B{excel_row}*100,0)",
            percent_format,
            record["success_pct"]
        )

        detail.write_number(
            row,
            6,
            record["payload"],
            number_format
        )

        # Payload savings relative to Raw (cell G7).
        detail.write_formula(
            row,
            7,
            f"=IFERROR((1-G{excel_row}/$G$7)*100,0)",
            green_format if i else percent_format,
            record["savings_pct"]
        )

        if record["delay_ms"] is not None:

            detail.write_number(
                row,
                8,
                float(record["delay_ms"]),
                decimal_format
            )

        if record["rate"] is not None:

            detail.write_number(
                row,
                9,
                float(record["rate"]),
                decimal_format
            )

    detail.autofilter(
        5,
        0,
        5 + len(records),
        9
    )

    detail.merge_range(
        "A16:J17",
        "The MQTT experiments use QoS 1 and measure "
        "successful reception on a local Windows computer. "
        "Repeated experiments under controlled network "
        "degradation are required before making "
        "conclusions about network reliability or latency.",
        note_format
    )

    # ========================================================
    # SHEET 3: GRAPHS
    # ========================================================

    graph_sheet = workbook.add_worksheet(
        "Graphs"
    )

    graph_sheet.hide_gridlines(2)

    graph_sheet.set_tab_color("#9333EA")

    graph_sheet.set_column(
        "A:N",
        12
    )

    graph_sheet.merge_range(
        "A1:N2",
        "MQTT | VISUAL COMPARISON",
        title_format
    )

    graph_sheet.insert_image(
        "A4",
        str(chart_file),
        {
            "x_scale": 0.67,
            "y_scale": 0.67
        }
    )

    workbook.close()


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate MQTT comparison report"
    )

    parser.add_argument(
        "--results",
        default="mqtt_results",
        help="Folder containing MQTT experiment results"
    )

    args = parser.parse_args()

    root = Path(args.results)

    # Load saved MQTT results.
    records = load_runs(root)

    # Create output folder.
    output_folder = root / "comparison"

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Output file paths.
    chart_file = (
        output_folder / "mqtt_comparison.png"
    )

    excel_file = (
        output_folder / "mqtt_comparison.xlsx"
    )

    # Generate graph.
    create_chart(
        records,
        chart_file
    )

    # Generate Excel workbook.
    create_excel(
        records,
        chart_file,
        excel_file
    )

    print("\nMQTT REPORT CREATED SUCCESSFULLY")

    print(
        f"\nExcel: {excel_file.resolve()}"
    )

    print(
        f"Graph: {chart_file.resolve()}"
    )


if __name__ == "__main__":
    main()