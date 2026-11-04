
"""
PID FIDELITY COMPARISON REPORT
==============================

Project:
Adaptive IoT Edge Downsampling

Compares:
1. Fixed Hybrid
2. PID-Adaptive Hybrid

Dataset:
Intel Berkeley Research Lab Sensor Dataset
Sensor 1 - Temperature
300 original readings

Evaluation metrics:
- RMSE
- MAE
- PRDN
- Data retention
- Data reduction
- Important-change proxy event recall

Analysis:
- Overall
- Congestion phase (samples 0-149)
- Recovery phase (samples 150-299)

Outputs:
mqtt_results/pid_fidelity/
    pid_fidelity_comparison.csv
    pid_reconstruction_samples.csv
    pid_fidelity_reconstruction.png

Run:
    python pid_fidelity_report.py
"""

import argparse
import csv
import math

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from benchmark_all import load_intel


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

FIXED_RUN = "01249630bf"

PID_RUN = "6d88062047"

RUN_LABELS = (
    "Fixed Hybrid",
    "PID Hybrid"
)


# ============================================================
# 2. LOAD RETAINED MQTT SENSOR READINGS
# ============================================================

def load_received(path, original):
    """
    Load the actual sensor readings received by
    the MQTT consumer.

    Verify:
    - File exists
    - Sample indices are valid
    - No duplicate sample indices
    - Values match the original Intel dataset
    """

    if not path.is_file():

        raise FileNotFoundError(
            f"MQTT consumer results not found: {path}"
        )

    indices = []
    observed_values = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        reader = csv.DictReader(handle)

        required = {
            "sample_index",
            "value"
        }

        if not required.issubset(
            reader.fieldnames or []
        ):

            raise ValueError(
                f"Missing required columns in {path}"
            )

        for row in reader:

            indices.append(
                int(row["sample_index"])
            )

            observed_values.append(
                float(row["value"])
            )

    indices = np.asarray(
        indices,
        dtype=int
    )

    observed_values = np.asarray(
        observed_values,
        dtype=float
    )

    # --------------------------------------------------------
    # VALIDATE READINGS
    # --------------------------------------------------------

    if len(indices) == 0:

        raise ValueError(
            f"No sensor readings found in {path}"
        )

    if (
        np.any(indices < 0)
        or np.any(indices >= len(original))
        or len(np.unique(indices)) != len(indices)
    ):

        raise ValueError(
            f"Invalid or duplicate indices in {path}"
        )

    if not np.all(
        np.isfinite(observed_values)
    ):

        raise ValueError(
            f"Non-finite sensor values in {path}"
        )

    if not np.allclose(
        observed_values,
        original[indices],
        atol=1e-5,
        rtol=0
    ):

        raise ValueError(
            f"Received sensor values do not match "
            f"the Intel dataset: {path}"
        )

    # Sort by original sample position.
    order = np.argsort(indices)

    return (
        indices[order],
        observed_values[order]
    )


# ============================================================
# 3. RECONSTRUCT THE ORIGINAL SENSOR SIGNAL
# ============================================================

def reconstruct_signal(
    original_length,
    retained_indices,
    retained_values
):
    """
    Reconstruct original signal using
    linear interpolation.

    This is retrospective reconstruction.

    It is not real-time, causal interpolation.

    At either boundary, the nearest retained
    value is extended to the edge.
    """

    original_positions = np.arange(
        original_length
    )

    reconstructed = np.interp(
        original_positions,
        retained_indices,
        retained_values
    )

    return reconstructed


# ============================================================
# 4. IDENTIFY IMPORTANT TEMPERATURE CHANGES
# ============================================================

def important_change_indices(
    original,
    fraction=0.05
):
    """
    Identify the top 5% largest absolute
    changes between consecutive readings.

    These are proxy events, NOT labelled faults.

    Event index represents the later sample
    in each pair of consecutive readings.
    """

    changes = np.abs(
        np.diff(original)
    )

    event_count = max(
        1,
        math.ceil(
            len(changes) * fraction
        )
    )

    # Sort changes from largest to smallest.
    # Use original sample order for tie-breaking.
    order = np.lexsort((
        np.arange(len(changes)),
        -changes
    ))

    event_indices = np.sort(
        order[:event_count] + 1
    )

    return event_indices


# ============================================================
# 5. CALCULATE DATA FIDELITY METRICS
# ============================================================

def calculate_metrics(
    original,
    reconstruction,
    retained_indices,
    important_events,
    start,
    end
):
    """
    Calculate fidelity metrics for a selected
    original-sample range.

    Metrics:
    - RMSE
    - MAE
    - PRDN
    - Retention percentage
    - Reduction percentage
    - Exact proxy-event recall
    - Proxy-event recall within +/-1 sample
    """

    actual = original[start:end]

    estimated = reconstruction[start:end]

    errors = actual - estimated

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = float(
        np.sqrt(
            np.mean(errors ** 2)
        )
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    # --------------------------------------------------------
    # PRDN
    # Percent Root-mean-square Difference Normalized
    # --------------------------------------------------------

    signal_energy = float(
        np.sum(
            (
                actual - np.mean(actual)
            ) ** 2
        )
    )

    error_energy = float(
        np.sum(
            errors ** 2
        )
    )

    if signal_energy > 1e-12:

        prdn = (
            100.0
            * np.sqrt(
                error_energy / signal_energy
            )
        )

    else:

        prdn = float("nan")

    # --------------------------------------------------------
    # RETAINED SENSOR READINGS WITHIN THIS PHASE
    # --------------------------------------------------------

    retained = retained_indices[
        (retained_indices >= start)
        & (retained_indices < end)
    ]

    original_count = end - start

    retained_count = len(retained)

    retention_pct = (
        100.0
        * retained_count
        / original_count
    )

    reduction_pct = (
        100.0 - retention_pct
    )

    # --------------------------------------------------------
    # IMPORTANT-CHANGE PROXY EVENTS
    # --------------------------------------------------------

    phase_events = important_events[
        (important_events >= start)
        & (important_events < end)
    ]

    event_count = len(
        phase_events
    )

    retained_set = set(
        retained.tolist()
    )

    # Exact retention:
    # The actual event sample is transmitted.
    exact_hits = sum(
        1
        for event in phase_events
        if int(event) in retained_set
    )

    # Near retention:
    # At least one retained reading lies
    # within +/-1 original sample position.
    near_hits = sum(
        1
        for event in phase_events
        if np.any(
            np.abs(retained - event) <= 1
        )
    )

    if event_count > 0:

        exact_recall = (
            100.0
            * exact_hits
            / event_count
        )

        near_recall = (
            100.0
            * near_hits
            / event_count
        )

    else:

        exact_recall = float("nan")
        near_recall = float("nan")

    # --------------------------------------------------------
    # RETURN ALL METRICS
    # --------------------------------------------------------

    return {

        "original_samples": original_count,

        "retained_samples": retained_count,

        "retention_pct": retention_pct,

        "data_reduction_pct": reduction_pct,

        "rmse_celsius": rmse,

        "mae_celsius": mae,

        "prdn_pct": prdn,

        "proxy_event_count": event_count,

        "proxy_event_exact_hits": exact_hits,

        "proxy_event_exact_recall_pct": exact_recall,

        "proxy_event_near_hits_pm1": near_hits,

        "proxy_event_near_recall_pct": near_recall

    }


# ============================================================
# 6. SAVE CSV FILE
# ============================================================

def save_csv(path, rows):

    if not rows:
        raise ValueError("No rows to save.")

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# 7. GENERATE RECONSTRUCTION GRAPHS
# ============================================================

def create_reconstruction_plot(
    output_file,
    original,
    results,
    phase_split
):

    positions = np.arange(
        len(original)
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        sharex=True
    )

    for ax, (
        algorithm,
        result
    ) in zip(
        axes,
        results.items()
    ):

        reconstructed = result[
            "reconstructed"
        ]

        selected_indices = result[
            "indices"
        ]

        # Original temperature readings.
        ax.plot(
            positions,
            original,
            linewidth=1.4,
            label="Original temperature"
        )

        # Reconstructed temperature signal.
        ax.plot(
            positions,
            reconstructed,
            linewidth=1.2,
            linestyle="--",
            label="Reconstructed temperature"
        )

        # Selected readings.
        ax.scatter(
            selected_indices,
            original[selected_indices],
            s=16,
            label="Retained sensor readings",
            zorder=3
        )

        # Network recovery boundary.
        ax.axvline(
            phase_split - 0.5,
            linestyle=":",
            linewidth=1.5,
            label="Network recovery boundary"
        )

        ax.set_title(
            f"{algorithm}: "
            f"{len(selected_indices)} retained readings"
        )

        ax.set_ylabel(
            "Temperature (°C)"
        )

        ax.grid(
            alpha=0.2
        )

        ax.legend(
            loc="best",
            fontsize=9
        )

    axes[-1].set_xlabel(
        "Original sensor sample index"
    )

    fig.suptitle(
        "Fixed Hybrid vs PID Hybrid - "
        "Sensor Data Reconstruction",
        fontsize=16,
        fontweight="bold"
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    fig.savefig(
        output_file,
        dpi=180,
        facecolor="white"
    )

    plt.close(fig)


# ============================================================
# 8. MAIN EVALUATION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare Fixed Hybrid and PID Hybrid "
            "sensor-data fidelity"
        )
    )

    parser.add_argument(
        "--data",
        default="data/data.txt.gz"
    )

    parser.add_argument(
        "--fixed-run",
        default=FIXED_RUN
    )

    parser.add_argument(
        "--pid-run",
        default=PID_RUN
    )

    parser.add_argument(
        "--sensor",
        type=int,
        default=1
    )

    parser.add_argument(
        "--feature",
        default="temperature"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=300
    )

    parser.add_argument(
        "--phase-split",
        type=int,
        default=150
    )

    parser.add_argument(
        "--results",
        default="mqtt_results"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # LOAD ORIGINAL INTEL DATASET
    # --------------------------------------------------------

    print("\nLoading original Intel sensor readings...")

    data_path = Path(
        args.data
    )

    df = load_intel(
        data_path,
        args.sensor,
        args.feature,
        args.max_samples
    )

    original = df[
        "value"
    ].to_numpy(
        dtype=float
    )

    if len(original) != args.max_samples:

        raise ValueError(
            f"Expected {args.max_samples} readings, "
            f"but received {len(original)}."
        )

    if not np.all(
        np.isfinite(original)
    ):

        raise ValueError(
            "Original sensor signal contains "
            "non-finite values."
        )

    if not (
        1 < args.phase_split < len(original) - 1
    ):

        raise ValueError(
            "Invalid phase split position."
        )

    # --------------------------------------------------------
    # PREPARE OUTPUT FOLDER
    # --------------------------------------------------------

    results_root = Path(
        args.results
    )

    output_folder = (
        results_root / "pid_fidelity"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # IDENTIFY PROXY EVENTS
    # --------------------------------------------------------

    important_events = important_change_indices(
        original,
        fraction=0.05
    )

    # --------------------------------------------------------
    # LOAD ACTUAL MQTT EXPERIMENTS
    # --------------------------------------------------------

    experiments = [
        (
            "Fixed Hybrid",
            args.fixed_run
        ),
        (
            "PID Hybrid",
            args.pid_run
        )
    ]

    results = {}

    for algorithm, run_id in experiments:

        received_file = (
            results_root
            / run_id
            / "received.csv"
        )

        indices, retained_values = (
            load_received(
                received_file,
                original
            )
        )

        reconstructed = reconstruct_signal(
            len(original),
            indices,
            retained_values
        )

        results[algorithm] = {

            "run_id": run_id,

            "indices": indices,

            "retained_values": retained_values,

            "reconstructed": reconstructed

        }

    # --------------------------------------------------------
    # EVALUATION PHASES
    # --------------------------------------------------------

    phases = [

        (
            "Overall",
            0,
            len(original)
        ),

        (
            "Congestion",
            0,
            args.phase_split
        ),

        (
            "Recovery",
            args.phase_split,
            len(original)
        )

    ]

    # --------------------------------------------------------
    # CALCULATE FIDELITY
    # --------------------------------------------------------

    comparison_rows = []

    for phase, start, end in phases:

        for algorithm, result in results.items():

            measurements = calculate_metrics(
                original,
                result["reconstructed"],
                result["indices"],
                important_events,
                start,
                end
            )

            comparison_rows.append({

                "phase": phase,

                "algorithm": algorithm,

                "run_id": result["run_id"],

                **measurements

            })

    # --------------------------------------------------------
    # SAVE COMPARISON CSV
    # --------------------------------------------------------

    comparison_path = (
        output_folder
        / "pid_fidelity_comparison.csv"
    )

    save_csv(
        comparison_path,
        comparison_rows
    )

    # --------------------------------------------------------
    # SAVE PER-SAMPLE RECONSTRUCTION DATA
    # --------------------------------------------------------

    fixed = results[
        "Fixed Hybrid"
    ]

    pid = results[
        "PID Hybrid"
    ]

    fixed_set = set(
        fixed["indices"].tolist()
    )

    pid_set = set(
        pid["indices"].tolist()
    )

    event_set = set(
        important_events.tolist()
    )

    sample_rows = []

    for index, value in enumerate(original):

        phase = (
            "Congestion"
            if index < args.phase_split
            else "Recovery"
        )

        sample_rows.append({

            "sample_index": index,

            "phase": phase,

            "original_temperature": float(value),

            "fixed_retained": int(
                index in fixed_set
            ),

            "pid_retained": int(
                index in pid_set
            ),

            "fixed_reconstructed": float(
                fixed["reconstructed"][index]
            ),

            "pid_reconstructed": float(
                pid["reconstructed"][index]
            ),

            "important_change_proxy_event": int(
                index in event_set
            )

        })

    sample_path = (
        output_folder
        / "pid_reconstruction_samples.csv"
    )

    save_csv(
        sample_path,
        sample_rows
    )

    # --------------------------------------------------------
    # GENERATE GRAPH
    # --------------------------------------------------------

    graph_path = (
        output_folder
        / "pid_fidelity_reconstruction.png"
    )

    create_reconstruction_plot(
        graph_path,
        original,
        results,
        args.phase_split
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "PID FIDELITY REPORT CREATED SUCCESSFULLY"
    )

    print("=" * 65)

    print(
        "Original sensor readings:",
        len(original)
    )

    print(
        "Important-change proxy events:",
        len(important_events)
    )

    print(
        "Fixed Hybrid retained:",
        len(fixed["indices"])
    )

    print(
        "PID Hybrid retained:",
        len(pid["indices"])
    )

    print("\nOVERALL FIDELITY COMPARISON")

    print("-" * 65)

    for row in comparison_rows:

        if row["phase"] != "Overall":
            continue

        print(
            f"\nAlgorithm: {row['algorithm']}"
        )

        print(
            f"RMSE: {row['rmse_celsius']:.5f} °C"
        )

        print(
            f"MAE: {row['mae_celsius']:.5f} °C"
        )

        print(
            f"PRDN: {row['prdn_pct']:.3f}%"
        )

        print(
            "Important-change exact recall: "
            f"{row['proxy_event_exact_recall_pct']:.2f}%"
        )

        print(
            "Important-change near recall (+/-1): "
            f"{row['proxy_event_near_recall_pct']:.2f}%"
        )

    print("\n" + "=" * 65)

    print(
        "Comparison CSV:",
        comparison_path.resolve()
    )

    print(
        "Reconstruction CSV:",
        sample_path.resolve()
    )

    print(
        "Graph:",
        graph_path.resolve()
    )

    print("=" * 65)

    print(
        "\nNote: Reconstruction is retrospective."
    )

    print(
        "Important-change proxy events are not "
        "labelled sensor faults."
    )


# ============================================================
# 9. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()