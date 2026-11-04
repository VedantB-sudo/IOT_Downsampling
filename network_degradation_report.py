
"""
MQTT Network Degradation Comparison Report

Project: Adaptive IoT Edge Downsampling

Generates:
1. network_degradation_comparison.xlsx
2. network_degradation_comparison.png

Run:
    python network_degradation_report.py

Required libraries:
    pip install matplotlib xlsxwriter
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import xlsxwriter


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

# Format:
# (Network scenario, Algorithm, Run ID, Trial number)

RUNS = [

    # Normal network - direct MQTT connection
    ("Unrestricted - direct", "raw", "2dbcc6e89d", 1),
    ("Unrestricted - direct", "uniform", "4c56d8d082", 1),
    ("Unrestricted - direct", "std_dev", "9d8fefbdf3", 1),
    ("Unrestricted - direct", "hp_pe", "c14250be33", 1),
    ("Unrestricted - direct", "hp_energy", "4df8b99920", 1),
    ("Unrestricted - direct", "hybrid", "8ac5731a82", 1),

    # Normal network through Toxiproxy
    ("Unrestricted - proxy", "hybrid", "93f5c2d029", 1),

    # 100 ms latency
    ("100 ms latency", "hybrid", "e0fe4000d5", 1),
    ("100 ms latency", "hybrid", "87c4b8dbc2", 2),

    # 300 ms latency
    ("300 ms latency", "hybrid", "4ed7a54fa4", 1),

    # Bandwidth restriction - 1 KB/s
    ("1 KB/s bandwidth", "raw", "74cf17abc3", 1),
    ("1 KB/s bandwidth", "uniform", "5001de5c2f", 1),
    ("1 KB/s bandwidth", "std_dev", "18576b816c", 1),
    ("1 KB/s bandwidth", "hp_pe", "f2f8f677e0", 1),
    ("1 KB/s bandwidth", "hp_energy", "ee542c5a89", 1),
    ("1 KB/s bandwidth", "hybrid", "6314b3ef52", 1),

]


ALGORITHM_NAMES = {
    "raw": "Raw",
    "uniform": "Uniform",
    "std_dev": "Standard Deviation",
    "hp_pe": "HP-PE",
    "hp_energy": "HP-Energy",
    "hybrid": "Hybrid"
}

ORDER = list(ALGORITHM_NAMES)


# ============================================================
# 2. PUBLISHER MEASUREMENTS
# ============================================================

# Values copied from MQTT publisher terminal screenshots.
#
# Format:
# Run ID:
# (Replay duration seconds,
#  Average scheduling delay ms,
#  Maximum scheduling delay ms)
#
# Missing values are intentionally left blank in the report.

PUBLISHER_MEASUREMENTS = {

    "2dbcc6e89d": (19.801, 0.364, 1.614),

    "8ac5731a82": (19.801, 0.349, 0.593),

    "e0fe4000d5": (19.902, 0.406, 1.569),

    "4ed7a54fa4": (20.506, 283.938, 810.164),

    "74cf17abc3": (20.856, 417.346, 845.693),

    "5001de5c2f": (20.014, 0.301, 0.562),

    "18576b816c": (20.013, 96.514, 212.849),

    "f2f8f677e0": (20.071, 44.123, 110.669),

    "ee542c5a89": (20.015, 112.810, 246.786),

    "6314b3ef52": (20.057, 28.851, 77.859),

}


# ============================================================
# 3. LOAD EXPERIMENT RESULTS
# ============================================================

def read_experiments(results_dir):

    records = []

    for scenario, algorithm, run_id, trial in RUNS:

        source = (
            results_dir
            / run_id
            / "summary.json"
        )

        if not source.is_file():

            raise FileNotFoundError(
                f"Cannot find: {source}\n"
                "Check that you are running the script "
                "from the IOT_Downsampling project folder."
            )

        with source.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        found_algorithm = data.get("algorithm")

        if found_algorithm != algorithm:

            raise ValueError(
                f"Wrong algorithm for {run_id}. "
                f"Expected {algorithm}, "
                f"found {found_algorithm}."
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

        missing = int(
            data["missing_messages"]
        )

        duplicates = int(
            data["duplicate_messages"]
        )

        delay = float(
            data["average_delivery_delay_ms"]
        )

        reception_rate = data.get(
            "reception_rate_messages_per_second"
        )

        if reception_rate is not None:

            reception_rate = float(reception_rate)

        if expected <= 0 or received < 0 or payload < 0:

            raise ValueError(
                f"Invalid consumer results in {source}"
            )

        replay, scheduling_avg, scheduling_max = (
            PUBLISHER_MEASUREMENTS.get(
                run_id,
                (None, None, None)
            )
        )

        success_percentage = (
            100 * received / expected
        )

        records.append({

            "scenario": scenario,

            "algorithm": algorithm,

            "name": ALGORITHM_NAMES[algorithm],

            "run_id": run_id,

            "trial": trial,

            "expected": expected,

            "received": received,

            "missing": missing,

            "duplicates": duplicates,

            "payload": payload,

            "delivery_success_pct": success_percentage,

            "delay_ms": delay,

            "reception_rate": reception_rate,

            "replay_s": replay,

            "scheduling_avg_ms": scheduling_avg,

            "scheduling_max_ms": scheduling_max,

        })

    return records


# ============================================================
# 4. FIND A SPECIFIC EXPERIMENT
# ============================================================

def find_one(
    records,
    scenario,
    algorithm,
    trial=1
):

    matches = [

        row for row in records

        if row["scenario"] == scenario

        and row["algorithm"] == algorithm

        and row["trial"] == trial

    ]

    if len(matches) != 1:

        raise ValueError(
            f"Expected one experiment for "
            f"{scenario}/{algorithm}/{trial}, "
            f"found {len(matches)}."
        )

    return matches[0]


# ============================================================
# 5. GENERATE COMPARISON GRAPHS
# ============================================================

def draw_plots(records, filename):

    direct = [

        find_one(
            records,
            "Unrestricted - direct",
            algorithm
        )

        for algorithm in ORDER

    ]

    limited = [

        find_one(
            records,
            "1 KB/s bandwidth",
            algorithm
        )

        for algorithm in ORDER

    ]

    latency = [

        find_one(
            records,
            "Unrestricted - proxy",
            "hybrid"
        ),

        find_one(
            records,
            "100 ms latency",
            "hybrid",
            1
        ),

        find_one(
            records,
            "100 ms latency",
            "hybrid",
            2
        ),

        find_one(
            records,
            "300 ms latency",
            "hybrid"
        ),

    ]

    names = [
        "Raw",
        "Uniform",
        "Std Dev",
        "HP-PE",
        "HP-Energy",
        "Hybrid"
    ]

    positions = list(range(6))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10)
    )

    # --------------------------------------------------------
    # GRAPH 1: Delivery delay by algorithm
    # --------------------------------------------------------

    ax = axes[0, 0]

    ax.bar(

        [p - 0.19 for p in positions],

        [r["delay_ms"] for r in direct],

        width=0.38,

        label="Unrestricted (direct)"

    )

    ax.bar(

        [p + 0.19 for p in positions],

        [r["delay_ms"] for r in limited],

        width=0.38,

        label="1 KB/s (proxy)"

    )

    ax.set_title(
        "Consumer Delivery Delay by Algorithm"
    )

    ax.set_ylabel(
        "Average Delivery Delay (ms)"
    )

    ax.set_xticks(
        positions,
        names,
        rotation=20
    )

    ax.legend(fontsize=9)

    # --------------------------------------------------------
    # GRAPH 2: Latency degradation
    # --------------------------------------------------------

    ax = axes[0, 1]

    latency_settings = [
        0,
        100,
        100,
        300
    ]

    measured_delays = [

        record["delay_ms"]

        for record in latency

    ]

    ax.plot(
        latency_settings,
        measured_delays,
        "o-"
    )

    for record, x in zip(
        latency,
        latency_settings
    ):

        ax.annotate(

            f'{record["delay_ms"]:.2f}',

            (x, record["delay_ms"]),

            xytext=(6, 5),

            textcoords="offset points",

            fontsize=8

        )

    ax.set_title(
        "Hybrid: Configured vs Measured Network Latency"
    )

    ax.set_xlabel(
        "Configured Additional Latency (ms)"
    )

    ax.set_ylabel(
        "Average Delivery Delay (ms)"
    )

    ax.set_xticks([
        0,
        100,
        300
    ])

    # --------------------------------------------------------
    # GRAPH 3: Publisher scheduling delay
    # --------------------------------------------------------

    ax = axes[1, 0]

    scheduling_values = [

        record["scheduling_avg_ms"]

        for record in limited

    ]

    ax.bar(
        names,
        scheduling_values
    )

    ax.set_title(
        "Publisher Scheduling Delay at 1 KB/s"
    )

    ax.set_ylabel(
        "Average Scheduling Delay (ms)"
    )

    ax.tick_params(
        axis="x",
        rotation=20
    )

    # --------------------------------------------------------
    # GRAPH 4: Application payload bytes
    # --------------------------------------------------------

    ax = axes[1, 1]

    payload_values = [

        record["payload"]

        for record in limited

    ]

    ax.bar(
        names,
        payload_values
    )

    ax.set_title(
        "Received Application Payload at 1 KB/s"
    )

    ax.set_ylabel(
        "Payload Bytes"
    )

    ax.tick_params(
        axis="x",
        rotation=20
    )

    # --------------------------------------------------------
    # Common graph formatting
    # --------------------------------------------------------

    for ax in axes.flat:

        ax.grid(
            axis="y",
            alpha=0.2
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)

    fig.suptitle(

        "MQTT Network Degradation | Intel Sensor 1",

        fontsize=17,

        fontweight="bold"

    )

    fig.text(

        0.5,
        0.015,

        "100 original readings | "
        "20% retention for filtered methods | "
        "Single-run comparisons are descriptive",

        ha="center",

        fontsize=9

    )

    fig.tight_layout(

        rect=[0, 0.045, 1, 0.94],

        h_pad=2.5,

        w_pad=2.0

    )

    fig.savefig(

        filename,

        dpi=190,

        facecolor="white"

    )

    plt.close(fig)


# ============================================================
# 6. GENERATE EXCEL WORKBOOK
# ============================================================

def make_report(
    records,
    image_file,
    output_file
):

    workbook = xlsxwriter.Workbook(
        str(output_file)
    )

    # --------------------------------------------------------
    # Excel formatting
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

    text_format = workbook.add_format({
        "font_color": "#1E293B"
    })

    number_format = workbook.add_format({
        "num_format": "#,##0"
    })

    decimal_format = workbook.add_format({
        "num_format": "0.000"
    })

    percent_format = workbook.add_format({
        "num_format": '0.00"%"'
    })

    note_format = workbook.add_format({

        "italic": True,

        "font_color": "#475569",

        "text_wrap": True,

        "valign": "top"

    })

    # --------------------------------------------------------
    # Helper: Create a formatted worksheet
    # --------------------------------------------------------

    def new_sheet(
        name,
        headers,
        widths
    ):

        sheet = workbook.add_worksheet(name)

        sheet.hide_gridlines(2)

        sheet.merge_range(

            0,
            0,
            1,
            len(headers) - 1,

            "MQTT NETWORK DEGRADATION | "
            + name.upper(),

            title_format

        )

        sheet.set_row(0, 23)

        sheet.set_row(1, 21)

        sheet.write_row(
            3,
            0,
            headers,
            header_format
        )

        sheet.set_row(3, 42)

        sheet.freeze_panes(4, 1)

        for index, width in enumerate(widths):

            sheet.set_column(
                index,
                index,
                width
            )

        return sheet

    # --------------------------------------------------------
    # Helper: Write a cell
    # --------------------------------------------------------

    def put(
        sheet,
        row,
        col,
        value,
        kind="text"
    ):

        if value is None:

            sheet.write_blank(
                row,
                col,
                None
            )

        elif kind == "int":

            sheet.write_number(
                row,
                col,
                int(value),
                number_format
            )

        elif kind == "float":

            sheet.write_number(
                row,
                col,
                float(value),
                decimal_format
            )

        elif kind == "pct":

            sheet.write_number(
                row,
                col,
                float(value),
                percent_format
            )

        else:

            sheet.write_string(
                row,
                col,
                str(value),
                text_format
            )

    # ========================================================
    # SHEET 1: ALL EXPERIMENTS
    # ========================================================

    log_columns = [

        ("Scenario", "scenario", "text"),

        ("Algorithm", "name", "text"),

        ("Trial", "trial", "int"),

        ("Run ID", "run_id", "text"),

        ("Expected", "expected", "int"),

        ("Received", "received", "int"),

        ("Missing", "missing", "int"),

        ("Duplicates", "duplicates", "int"),

        ("Success (%)", "delivery_success_pct", "pct"),

        ("Payload Bytes", "payload", "int"),

        ("Avg Delivery (ms)", "delay_ms", "float"),

        ("Reception (msg/s)", "reception_rate", "float"),

        ("Replay (s)*", "replay_s", "float"),

        ("Avg Scheduling (ms)*", "scheduling_avg_ms", "float"),

        ("Max Scheduling (ms)*", "scheduling_max_ms", "float"),

    ]

    log = new_sheet(

        "All Runs",

        [column[0] for column in log_columns],

        [
            25, 22, 9, 16, 12,
            12, 11, 12, 14, 16,
            20, 19, 15, 22, 22
        ]

    )

    for index, record in enumerate(records, 4):

        for col, (_, key, kind) in enumerate(log_columns):

            put(
                log,
                index,
                col,
                record[key],
                kind
            )

    log.autofilter(
        3,
        0,
        3 + len(records),
        len(log_columns) - 1
    )

    log.merge_range(

        22, 0,
        23, 14,

        "Publisher replay and scheduling measurements "
        "were transcribed from terminal screenshots. "
        "Blank cells indicate unavailable publisher data. "
        "Consumer measurements are loaded directly "
        "from summary.json.",

        note_format

    )

    # ========================================================
    # SHEET 2: BANDWIDTH COMPARISON
    # ========================================================

    bandwidth_headers = [

        "Algorithm",

        "Direct Avg Delay (ms)",

        "1 KB/s Avg Delay (ms)",

        "Delay Change (ms)",

        "1 KB/s Payload (B)",

        "Payload Savings vs Raw (%)",

        "Delivered / Expected",

        "Avg Scheduling (ms)*",

        "Replay (s)*",

        "Run ID"

    ]

    bandwidth = new_sheet(

        "Bandwidth",

        bandwidth_headers,

        [
            24, 22, 23, 20, 20,
            24, 21, 22, 17, 17
        ]

    )

    raw_limited = find_one(

        records,

        "1 KB/s bandwidth",

        "raw"

    )

    for index, algorithm in enumerate(ORDER):

        row = 4 + index

        direct = find_one(

            records,

            "Unrestricted - direct",

            algorithm

        )

        limited = find_one(

            records,

            "1 KB/s bandwidth",

            algorithm

        )

        savings = (

            1
            - limited["payload"]
            / raw_limited["payload"]

        ) * 100

        values = [

            (limited["name"], "text"),

            (direct["delay_ms"], "float"),

            (limited["delay_ms"], "float"),

            (
                limited["delay_ms"] - direct["delay_ms"],
                "float"
            ),

            (limited["payload"], "int"),

            (savings, "pct"),

            (
                f'{limited["received"]}/{limited["expected"]}',
                "text"
            ),

            (limited["scheduling_avg_ms"], "float"),

            (limited["replay_s"], "float"),

            (limited["run_id"], "text"),

        ]

        for col, (value, kind) in enumerate(values):

            put(
                bandwidth,
                row,
                col,
                value,
                kind
            )

    bandwidth.autofilter(
        3,
        0,
        9,
        9
    )

    bandwidth.merge_range(

        12, 0,
        14, 9,

        "IMPORTANT: Unrestricted results used a direct "
        "MQTT connection, while 1 KB/s tests used Toxiproxy. "
        "These comparisons are descriptive. "
        "Application payload excludes MQTT/TCP overhead. "
        "Delivery delay is measured locally.",

        note_format

    )

    # ========================================================
    # SHEET 3: HYBRID LATENCY
    # ========================================================

    latency_headers = [

        "Scenario",

        "Added Latency (ms)",

        "Trial",

        "Run ID",

        "Avg Delivery (ms)",

        "Increase vs Proxy Baseline (ms)",

        "Messages Received",

        "Missing",

        "Payload Bytes",

        "Replay (s)*",

        "Avg Scheduling (ms)*"

    ]

    latency_sheet = new_sheet(

        "Hybrid Latency",

        latency_headers,

        [
            23, 23, 10, 17, 23,
            28, 20, 13, 18, 18, 24
        ]

    )

    proxy_baseline = find_one(

        records,

        "Unrestricted - proxy",

        "hybrid"

    )

    latency_experiments = [

        (0, proxy_baseline),

        (
            100,
            find_one(
                records,
                "100 ms latency",
                "hybrid",
                1
            )
        ),

        (
            100,
            find_one(
                records,
                "100 ms latency",
                "hybrid",
                2
            )
        ),

        (
            300,
            find_one(
                records,
                "300 ms latency",
                "hybrid"
            )
        ),

    ]

    for index, (
        configured_ms,
        record
    ) in enumerate(latency_experiments, 4):

        values = [

            (record["scenario"], "text"),

            (configured_ms, "int"),

            (record["trial"], "int"),

            (record["run_id"], "text"),

            (record["delay_ms"], "float"),

            (
                record["delay_ms"]
                - proxy_baseline["delay_ms"],
                "float"
            ),

            (
                f'{record["received"]}/{record["expected"]}',
                "text"
            ),

            (record["missing"], "int"),

            (record["payload"], "int"),

            (record["replay_s"], "float"),

            (record["scheduling_avg_ms"], "float"),

        ]

        for col, (value, kind) in enumerate(values):

            put(
                latency_sheet,
                index,
                col,
                value,
                kind
            )

    latency_sheet.autofilter(
        3,
        0,
        7,
        10
    )

    latency_sheet.merge_range(

        10, 0,
        12, 10,

        "Hybrid latency tests use the Toxiproxy "
        "normal-network experiment as their baseline. "
        "Both 100 ms trials are included. "
        "Reported delivery delay measures local application "
        "delivery, not network RTT.",

        note_format

    )

    # ========================================================
    # SHEET 4: METHODOLOGY AND LIMITATIONS
    # ========================================================

    notes = workbook.add_worksheet(
        "Read Me"
    )

    notes.hide_gridlines(2)

    notes.set_column(
        "A:A",
        30
    )

    notes.set_column(
        "B:B",
        100
    )

    notes.merge_range(

        "A1:B2",

        "METHOD AND INTERPRETATION",

        title_format

    )

    notes_rows = [

        (
            "Dataset",
            "Intel Berkeley Research Lab Sensor Dataset."
        ),

        (
            "Sensor",
            "Sensor 1, temperature readings."
        ),

        (
            "Sample size",
            "100 original readings per experiment."
        ),

        (
            "Retention",
            "Raw uses 100%; filtered methods use 20%."
        ),

        (
            "MQTT transport",
            "Local Mosquitto MQTT broker, QoS 1."
        ),

        (
            "Network emulator",
            "Toxiproxy, applied to the producer-to-broker TCP connection."
        ),

        (
            "Latency experiments",
            "Hybrid tested under 0, 100 and 300 ms added upstream latency."
        ),

        (
            "Bandwidth experiments",
            "All six methods tested under a 1 KB/s upstream bandwidth limit."
        ),

        (
            "Application payload",
            "Received message payload bytes, excluding MQTT/TCP overhead."
        ),

        (
            "Delivery delay",
            "Measured by the MQTT consumer using local application timestamps."
        ),

        (
            "Scheduling delay",
            "Difference between intended replay schedule and actual publishing time."
        ),

        (
            "Comparison fairness",
            "All filtered algorithms use the same sample budget but may retain different timestamps."
        ),

        (
            "Limitations",
            "One bandwidth trial per algorithm. Results are descriptive and not statistically conclusive."
        ),

        (
            "Baseline limitation",
            "Direct MQTT results are compared against Toxiproxy results, so configurations are not fully matched."
        ),

        (
            "Future evaluation",
            "PID adaptation, dynamic retention, repeated trials and additional network scenarios."
        ),

    ]

    for index, (
        heading,
        content
    ) in enumerate(notes_rows, 4):

        notes.write(
            index,
            0,
            heading,
            header_format
        )

        notes.write(
            index,
            1,
            content,
            text_format
        )

        notes.set_row(
            index,
            30
        )

    # ========================================================
    # SHEET 5: GRAPHS
    # ========================================================

    graphs = workbook.add_worksheet(
        "Graphs"
    )

    graphs.hide_gridlines(2)

    graphs.set_column(
        "A:N",
        12
    )

    graphs.merge_range(

        "A1:N2",

        "NETWORK DEGRADATION | GRAPHS",

        title_format

    )

    graphs.insert_image(

        "A4",

        str(image_file),

        {
            "x_scale": 0.70,
            "y_scale": 0.70
        }

    )

    workbook.close()


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Generate MQTT network degradation comparison report"
        )

    )

    parser.add_argument(

        "--results",

        default="mqtt_results",

        help="Directory containing MQTT experiment folders"

    )

    args = parser.parse_args()

    results_dir = Path(args.results)

    # Read all experiment results.
    records = read_experiments(
        results_dir
    )

    # Output directory.
    output_dir = (
        results_dir / "network_degradation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Output files.
    chart_path = (

        output_dir
        / "network_degradation_comparison.png"

    )

    excel_path = (

        output_dir
        / "network_degradation_comparison.xlsx"

    )

    # Generate graphs.
    draw_plots(
        records,
        chart_path
    )

    # Generate Excel report.
    make_report(
        records,
        chart_path,
        excel_path
    )

    print(
        "\nNETWORK DEGRADATION REPORT CREATED SUCCESSFULLY"
    )

    print(
        f"Experiments included: {len(records)}"
    )

    print(
        f"Excel: {excel_path.resolve()}"
    )

    print(
        f"Graphs: {chart_path.resolve()}"
    )


if __name__ == "__main__":
    main()