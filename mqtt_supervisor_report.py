
"""MQTT-only supervisor benchmark report.

Run from the IOT_Downsampling project folder:
    python mqtt_supervisor_report.py

Dependencies:
    pip install numpy xlsxwriter

Uses the saved MQTT experiments:
    - received.csv
    - summary.json
    - publisher_summary.json (where available)

Generates:
    mqtt_results/supervisor_mqtt_benchmarks.xlsx
"""

import csv
import json
from pathlib import Path

import numpy as np
import xlsxwriter

from benchmark_all import load_intel


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

RESULTS = ROOT / "mqtt_results"

OUTPUT = RESULTS / "supervisor_mqtt_benchmarks.xlsx"

DATASET = ROOT / "data" / "data.txt.gz"

SENSOR = 1
FEATURE = "temperature"
N = 100

ORDER = [
    "raw",
    "uniform",
    "std_dev",
    "hp_pe",
    "hp_energy",
    "hybrid",
]

NAMES = [
    "Raw",
    "Uniform",
    "Std Dev",
    "HP-PE",
    "HP-Energy",
    "Hybrid",
]


# ============================================================
# 2. MQTT EXPERIMENT RUN IDS
# ============================================================

RUNS = {
    "Normal (direct)": {
        "raw": "2dbcc6e89d",
        "uniform": "4c56d8d082",
        "std_dev": "9d8fefbdf3",
        "hp_pe": "c14250be33",
        "hp_energy": "4df8b99920",
        "hybrid": "8ac5731a82",
    },

    "1 KB/s (proxy)": {
        "raw": "74cf17abc3",
        "uniform": "5001de5c2f",
        "std_dev": "18576b816c",
        "hp_pe": "f2f8f677e0",
        "hp_energy": "ee542c5a89",
        "hybrid": "6314b3ef52",
    },
}


# Scheduling measurements from previous experiment outputs.
# Missing measurements are left blank rather than estimated.

KNOWN_PUBLISHER_SCHEDULING_MS = {
    "2dbcc6e89d": 0.364,
    "4c56d8d082": 0.380,
    "9d8fefbdf3": 0.307,
    "c14250be33": 0.314,
    "8ac5731a82": 0.349,

    "74cf17abc3": 417.346,
    "5001de5c2f": 0.301,
    "18576b816c": 96.514,
    "f2f8f677e0": 44.123,
    "ee542c5a89": 112.810,
    "6314b3ef52": 28.851,
}


# ============================================================
# 3. DATA LOADING FUNCTIONS
# ============================================================

def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as file:

        return list(csv.DictReader(file))


# ============================================================
# 4. CALCULATE MQTT PERFORMANCE METRICS
# ============================================================

def calculate(scenario, alg, run_id, original):

    folder = RESULTS / run_id

    summary = load_json(
        folder / "summary.json"
    )

    rows = load_csv(
        folder / "received.csv"
    )

    if summary.get("algorithm") != alg:
        raise ValueError(
            f"Algorithm mismatch for run {run_id}"
        )

    expected = int(
        summary["messages_expected"]
    )

    received = int(
        summary["messages_received"]
    )

    if received != len(rows):
        raise ValueError(
            f"Consumer summary / CSV count mismatch: {run_id}"
        )

    indices = []
    delays = []
    sizes = []

    seen = set()

    mismatches = 0
    invalid = 0

    # --------------------------------------------------------
    # Message correctness verification
    # --------------------------------------------------------

    for rec in rows:

        idx = int(
            rec["sample_index"]
        )

        value = float(
            rec["value"]
        )

        msg_id = int(
            rec["message_id"]
        )

        indices.append(idx)

        delays.append(
            float(rec["delivery_delay_ms"])
        )

        sizes.append(
            int(rec["payload_bytes"])
        )

        # Compare received value against Intel source dataset.

        if (
            idx < 0
            or idx >= len(original)
            or not np.isfinite(value)
        ):
            invalid += 1

        elif not np.isclose(
            value,
            original[idx],
            atol=1e-5,
            rtol=0
        ):
            mismatches += 1

        # Detect repeated message IDs.

        if msg_id in seen:
            invalid += 1

        seen.add(msg_id)

    # --------------------------------------------------------
    # Validate saved consumer statistics
    # --------------------------------------------------------

    payload = sum(sizes)

    if payload != int(
        summary["payload_bytes_received"]
    ):
        raise ValueError(
            f"Payload mismatch for run {run_id}"
        )

    if received > expected:
        raise ValueError(
            f"Unexpected received > published: {run_id}"
        )

    # --------------------------------------------------------
    # Correctness and delivery metrics
    # --------------------------------------------------------

    bad = mismatches + invalid

    valid = received - bad

    correctness = (
        100 * valid / received
        if received
        else None
    )

    delivery = (
        100 * received / expected
        if expected
        else None
    )

    unique_sample_count = len(
        set(indices)
    )

    # --------------------------------------------------------
    # Publisher metrics
    # --------------------------------------------------------

    pub_path = (
        folder / "publisher_summary.json"
    )

    if pub_path.exists():
        pub = json.loads(
            pub_path.read_text(
                encoding="utf-8"
            )
        )
    else:
        pub = {}

    sched = pub.get(
        "average_scheduling_delay_ms",
        KNOWN_PUBLISHER_SCHEDULING_MS.get(run_id)
    )

    # --------------------------------------------------------
    # MQTT reception throughput
    # --------------------------------------------------------

    throughput = summary.get(
        "reception_rate_messages_per_second"
    )

    if throughput is not None:
        throughput = float(throughput)

    payload_rate = (
        payload / received * throughput
        if received and throughput is not None
        else None
    )

    # --------------------------------------------------------
    # Delay statistics
    # --------------------------------------------------------

    avg_delay = (
        float(np.mean(delays))
        if delays
        else None
    )

    min_delay = (
        float(np.min(delays))
        if delays
        else None
    )

    max_delay = (
        float(np.max(delays))
        if delays
        else None
    )

    p95_delay = (
        float(np.percentile(delays, 95))
        if delays
        else None
    )

    return {
        "scenario": scenario,
        "algorithm": alg,
        "name": NAMES[ORDER.index(alg)],
        "run_id": run_id,

        "original": len(original),
        "selected": expected,
        "received": received,

        "retention": (
            100 * expected / len(original)
        ),

        "reduction": (
            100 * (1 - expected / len(original))
        ),

        "delivery": delivery,

        "missing": int(
            summary["missing_messages"]
        ),

        "duplicates": int(
            summary["duplicate_messages"]
        ),

        "correct": valid,
        "incorrect": bad,
        "correctness": correctness,

        "unique_sample_indices": unique_sample_count,

        "avg_delay": avg_delay,
        "min_delay": min_delay,
        "max_delay": max_delay,
        "p95_delay": p95_delay,

        "messages_per_s": throughput,

        "bytes": payload,

        "app_bytes_per_s": payload_rate,

        "scheduling_ms": (
            float(sched)
            if sched is not None
            else None
        ),
    }


# ============================================================
# 5. CREATE EXCEL BENCHMARK TABLES
# ============================================================

def write_table(
    ws,
    records,
    title,
    header_fmt,
    title_fmt,
    percent_fmt,
    number_fmt,
    raw_baseline=None
):

    headers = [
        "Method",
        "Run ID",
        "Original",
        "Selected/Published",
        "Received",
        "Retention %",
        "Reduction %",
        "Delivery Success %",
        "Missing",
        "Duplicates",
        "Correct Values",
        "Incorrect Values",
        "Correctness %",
        "Avg Delay ms",
        "Min Delay ms",
        "Max Delay ms",
        "P95 Delay ms",
        "Messages/s",
        "Payload Bytes",
        "App Payload B/s",
        "Avg Scheduling Delay ms",
        "Payload Savings vs Raw %",
    ]

    # Title

    ws.merge_range(
        0,
        0,
        1,
        len(headers) - 1,
        title,
        title_fmt
    )

    # Header

    ws.set_row(2, 42)

    ws.write_row(
        2,
        0,
        headers,
        header_fmt
    )

    ws.freeze_panes(3, 2)

    ws.set_column(0, 0, 17)
    ws.set_column(1, 1, 16)

    ws.set_column(
        2,
        len(headers) - 1,
        18
    )

    ws.autofilter(
        2,
        0,
        2 + len(records),
        len(headers) - 1
    )

    if raw_baseline is None:

        raw_baseline = next(
            x["bytes"]
            for x in records
            if x["algorithm"] == "raw"
        )

    # Write results

    for i, r in enumerate(records, 3):

        values = [
            r["name"],
            r["run_id"],
            r["original"],
            r["selected"],
            r["received"],
            r["retention"],
            r["reduction"],
            r["delivery"],
            r["missing"],
            r["duplicates"],
            r["correct"],
            r["incorrect"],
            r["correctness"],
            r["avg_delay"],
            r["min_delay"],
            r["max_delay"],
            r["p95_delay"],
            r["messages_per_s"],
            r["bytes"],
            r["app_bytes_per_s"],
            r["scheduling_ms"],
            100 * (
                1 - r["bytes"] / raw_baseline
            ),
        ]

        for j, value in enumerate(values):

            if value is None:

                ws.write_blank(
                    i,
                    j,
                    None
                )

            elif isinstance(value, str):

                ws.write_string(
                    i,
                    j,
                    value
                )

            elif j in [
                5, 6, 7, 12, 21
            ]:

                ws.write_number(
                    i,
                    j,
                    value,
                    percent_fmt
                )

            else:

                ws.write_number(
                    i,
                    j,
                    value,
                    number_fmt
                )

    # Highlight metric differences

    ws.conditional_format(
        3,
        12,
        2 + len(records),
        12,
        {
            "type": "3_color_scale"
        }
    )

    ws.conditional_format(
        3,
        13,
        2 + len(records),
        13,
        {
            "type": "3_color_scale"
        }
    )


# ============================================================
# 6. MAIN REPORT GENERATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Load original Intel sensor readings
    # --------------------------------------------------------

    if not DATASET.exists():

        raise FileNotFoundError(
            f"Intel dataset not found: {DATASET}"
        )

    df = load_intel(
        DATASET,
        SENSOR,
        FEATURE,
        N
    )

    original = df[
        "value"
    ].to_numpy(dtype=float)

    if len(original) != N:

        raise ValueError(
            f"Expected {N} readings, "
            f"received {len(original)}"
        )

    # --------------------------------------------------------
    # Calculate metrics for all 12 MQTT runs
    # --------------------------------------------------------

    rows = []

    for scenario, methods in RUNS.items():

        for alg in ORDER:

            rows.append(
                calculate(
                    scenario,
                    alg,
                    methods[alg],
                    original
                )
            )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create Excel workbook
    # --------------------------------------------------------

    book = xlsxwriter.Workbook(
        str(OUTPUT)
    )

    book.set_properties({
        "title": "MQTT Supervisor Benchmarks",
        "comments": (
            "Measured results; not ML inference "
            "or total wire bandwidth"
        )
    })

    # --------------------------------------------------------
    # Excel formatting
    # --------------------------------------------------------

    title = book.add_format({
        "bold": True,
        "font_size": 17,
        "font_color": "white",
        "bg_color": "#17324F",
        "valign": "vcenter"
    })

    header = book.add_format({
        "bold": True,
        "font_color": "white",
        "bg_color": "#087D8B",
        "text_wrap": True,
        "valign": "vcenter",
        "border": 1,
        "border_color": "#DDE6ED"
    })

    pct = book.add_format({
        "num_format": '0.00"%"'
    })

    num = book.add_format({
        "num_format": "#,##0.000"
    })

    normal = rows[:6]
    limited = rows[6:]

    # ========================================================
    # SHEET 1: NORMAL MQTT BENCHMARKS
    # ========================================================

    ws1 = book.add_worksheet(
        "Normal MQTT"
    )

    write_table(
        ws1,
        normal,
        "MQTT NORMAL NETWORK | SIX METHODS",
        header,
        title,
        pct,
        num
    )

    # ========================================================
    # SHEET 2: DEGRADED NETWORK BENCHMARKS
    # ========================================================

    ws2 = book.add_worksheet(
        "1 KBs MQTT"
    )

    write_table(
        ws2,
        limited,
        "MQTT 1 KB/s UPSTREAM BANDWIDTH | SIX METHODS",
        header,
        title,
        pct,
        num
    )

    # ========================================================
    # SHEET 3: MESSAGE VERIFICATION
    # ========================================================

    verify = book.add_worksheet(
        "Message Verification"
    )

    verify.merge_range(
        "A1:H2",
        "PER-MESSAGE CORRECTNESS CHECK",
        title
    )

    cols = [
        "Scenario",
        "Method",
        "Run ID",
        "Sample Index",
        "Received Value",
        "Original Value",
        "Match",
        "Delivery Delay ms"
    ]

    verify.write_row(
        2,
        0,
        cols,
        header
    )

    verify.set_row(2, 34)

    verify.set_column(
        "A:A",
        23
    )

    verify.set_column(
        "B:C",
        17
    )

    verify.set_column(
        "D:H",
        18
    )

    verify.freeze_panes(
        3,
        3
    )

    outrow = 3

    for r in rows:

        events = load_csv(
            RESULTS
            / r["run_id"]
            / "received.csv"
        )

        for event in events:

            idx = int(
                event["sample_index"]
            )

            value = float(
                event["value"]
            )

            valid = (
                0 <= idx < len(original)
            )

            original_val = (
                float(original[idx])
                if valid
                else None
            )

            match = bool(
                valid
                and np.isclose(
                    value,
                    original_val,
                    atol=1e-5,
                    rtol=0
                )
            )

            verify.write_row(
                outrow,
                0,
                [
                    r["scenario"],
                    r["name"],
                    r["run_id"],
                    idx,
                    value
                ]
            )

            if original_val is not None:

                verify.write_number(
                    outrow,
                    5,
                    original_val
                )

            else:

                verify.write_blank(
                    outrow,
                    5,
                    None
                )

            verify.write(
                outrow,
                6,
                "MATCH" if match else "MISMATCH"
            )

            verify.write_number(
                outrow,
                7,
                float(
                    event["delivery_delay_ms"]
                ),
                num
            )

            outrow += 1

    verify.autofilter(
        2,
        0,
        outrow - 1,
        7
    )

    # ========================================================
    # SHEET 4: BENCHMARK GRAPHS
    # ========================================================

    charts = book.add_worksheet(
        "Benchmark Graphs"
    )

    charts.set_column(
        "A:V",
        11
    )

    charts.merge_range(
        "A1:V2",
        "MQTT COMPARISON | SUPERVISOR CHARTS",
        title
    )

    # Chart title, Excel column index, source sheet,
    # chart type, placement row and column.

    specs = [
        (
            "Retention (%)",
            5,
            "Normal MQTT",
            "bar",
            3,
            0
        ),

        (
            "Delivery delay (ms)",
            13,
            "Normal MQTT",
            "column",
            3,
            9
        ),

        (
            "Payload bytes",
            18,
            "Normal MQTT",
            "column",
            19,
            0
        ),

        (
            "Message correctness (%)",
            12,
            "Normal MQTT",
            "column",
            19,
            9
        ),

        (
            "Throughput (msg/s)",
            17,
            "1 KBs MQTT",
            "column",
            35,
            0
        ),

        (
            "Publisher scheduling delay (ms)",
            20,
            "1 KBs MQTT",
            "column",
            35,
            9
        ),

        (
            "Delivery delay: Normal vs 1 KB/s",
            13,
            "comparison",
            "column",
            51,
            0
        ),
    ]

    for (
        plot_title,
        col,
        source,
        kind,
        r0,
        c0
    ) in specs:

        chart = book.add_chart({
            "type": kind
        })

        chart.set_title({
            "name": plot_title
        })

        chart.set_legend({
            "position": "bottom"
        })

        chart.set_style(10)

        if source == "comparison":

            sources = [
                ("Normal", "Normal MQTT"),
                ("1 KB/s", "1 KBs MQTT")
            ]

        else:

            sources = [
                (source, source)
            ]

        for label, src in sources:

            chart.add_series({
                "name": label,

                "categories": [
                    src,
                    3,
                    0,
                    8,
                    0
                ],

                "values": [
                    src,
                    3,
                    col,
                    8,
                    col
                ],

                "data_labels": {
                    "value": False
                }
            })

        chart.set_size({
            "width": 630,
            "height": 280
        })

        charts.insert_chart(
            r0,
            c0,
            chart
        )

    # ========================================================
    # SHEET 5: BASELINE PAPER METRIC ALIGNMENT
    # ========================================================

    mapping = book.add_worksheet(
        "Baseline Alignment"
    )

    mapping.set_column(
        "A:A",
        26
    )

    mapping.set_column(
        "B:B",
        40
    )

    mapping.set_column(
        "C:C",
        74
    )

    mapping.merge_range(
        "A1:C2",
        "BASELINE PAPER ALIGNMENT & LIMITATIONS",
        title
    )

    mapping.write_row(
        3,
        0,
        [
            "Baseline theme",
            "MQTT benchmark metric",
            "Scope and limitations"
        ],
        header
    )

    align = [

        (
            "Downsampling / data reduction",
            "Message retention %, Data reduction %",
            "Calculated from original readings and "
            "actual selected/published messages."
        ),

        (
            "Network efficiency / bandwidth",
            "Telemetry payload bytes, app payload B/s",
            "Only JSON application payload; not "
            "TCP/MQTT wire overhead or link utilization."
        ),

        (
            "Latency / soft real-time",
            "Mean, min, max, p95 delivery delay; "
            "scheduling lag",
            "Local producer-to-consumer application "
            "timestamps; not remote RTT."
        ),

        (
            "Data integrity",
            "Delivery success %, missing/duplicates, "
            "value correctness %",
            "Correctness compares received retained "
            "values to source dataset; does NOT "
            "quantify discarded-sample fidelity."
        ),

        (
            "Streaming performance",
            "Received messages/s",
            "Reception rate is retained-message "
            "throughput, not original sensor arrival rate."
        ),

        (
            "PID adaptation",
            "NOT INCLUDED in fixed-method comparison",
            "Use separate adaptive-PID trials; "
            "not equivalent to fixed algorithms."
        ),

        (
            "ML state prediction / accuracy",
            "NOT MEASURED",
            "No ML prediction evaluation in current "
            "MQTT consumer/producer results."
        ),

        (
            "Filter computation / edge resources",
            "NOT MEASURED in these runs",
            "Current MQTT logs do not isolate "
            "algorithm CPU time, CPU use, or memory."
        ),

        (
            "Reconstruction / upsampling",
            "NOT INCLUDED (MQTT-only report)",
            "Value correctness does not equal "
            "signal reconstruction correctness."
        ),
    ]

    for i, row in enumerate(align, 4):

        mapping.write_row(
            i,
            0,
            row
        )

    mapping.write(
        16,
        0,
        "Paper / DOI"
    )

    mapping.write_url(
        16,
        1,
        "https://doi.org/10.1016/j.array.2025.100488"
    )

    mapping.write(
        18,
        0,
        "Fairness note"
    )

    mapping.write(
        18,
        1,
        "All six methods use the same 100 source "
        "readings and replay interval. "
        "Normal tests are direct MQTT; "
        "restricted tests use Toxiproxy."
    )

    mapping.write(
        20,
        0,
        "Study limitation"
    )

    mapping.write(
        20,
        1,
        "Single runs per method/condition; "
        "no statistical significance claims. "
        "Equal sample budgets produce "
        "similar payload sizes."
    )

    # ========================================================
    # SAVE WORKBOOK
    # ========================================================

    book.close()

    print(
        "\nMQTT SUPERVISOR BENCHMARK REPORT "
        "CREATED SUCCESSFULLY"
    )

    print(
        "Experiments:",
        len(rows)
    )

    print(
        "Verification rows:",
        outrow - 3
    )

    print(
        "Excel:",
        OUTPUT.resolve()
    )

    # Print experiment summary

    for r in rows:

        print(
            f"{r['scenario']:18} "
            f"{r['name']:10} "
            f"published={r['selected']:3} "
            f"received={r['received']:3} "
            f"correctness={r['correctness']:.1f}% "
            f"avg_delay={r['avg_delay']:.3f}ms"
        )


# ============================================================
# 7. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()