
"""
IoT Downsampling Benchmark - Robustness Testing

Five Algorithms:
1. Uniform Decimation
2. Standard Deviation
3. HP-PE
4. HP-Energy
5. Hybrid HP-PE + HP-Energy

Six Evaluation Metrics:
1. Data Reduction (%)
2. Important Data Retained (%)
3. Event Recall (%)
4. F2-Score
5. PRDN (%)
6. Processing Time (ms)

Robustness Experiments:
- Different event positions
- Different event amplitudes
- Different event durations

This is an offline experiment.
MQTT and PID adaptation are not included.
"""

import argparse
import json
import shutil
import time
import urllib.request

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms import METHODS, choose_samples


# ============================================================
# 1. DATASET CONFIGURATION
# ============================================================

DATA_URL = "https://db.csail.mit.edu/labdata/data.txt.gz"

COLUMNS = [
    "date",
    "time",
    "epoch",
    "moteid",
    "temperature",
    "humidity",
    "light",
    "voltage"
]

METRICS = [
    "data_reduction_pct",
    "important_data_retained_pct",
    "event_recall_pct",
    "F2_score",
    "PRDN_pct",
    "selection_time_ms"
]


# ============================================================
# 2. LOAD INTEL LAB DATASET
# ============================================================

def load_intel(path, sensor, feature, limit):

    path = Path(path)

    if not path.exists():

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        print("Downloading Intel Lab dataset...")

        try:
            request = urllib.request.Request(
                DATA_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(
                request,
                timeout=90
            ) as source, path.open("wb") as target:

                shutil.copyfileobj(source, target)

        except Exception as exc:

            path.unlink(missing_ok=True)

            raise RuntimeError(
                f"Download failed. Save data.txt.gz "
                f"to {path}. Error: {exc}"
            ) from exc

    blocks = []

    for block in pd.read_csv(
        path,
        sep=" ",
        names=COLUMNS,
        header=None,
        chunksize=200_000,
        compression="infer"
    ):

        rows = block.loc[
            pd.to_numeric(
                block["moteid"],
                errors="coerce"
            ) == sensor,
            ["date", "time", feature]
        ]

        if not rows.empty:
            blocks.append(rows.copy())

    if not blocks:
        raise ValueError(
            f"No readings found for sensor {sensor}."
        )

    df = pd.concat(
        blocks,
        ignore_index=True
    )

    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str)
        + " "
        + df["time"].astype(str),
        format="mixed",
        errors="coerce"
    )

    df["value"] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    df = df.loc[
        df["timestamp"].notna()
        & np.isfinite(df["value"]),
        ["timestamp", "value"]
    ]

    df = (
        df.sort_values(
            "timestamp",
            kind="stable"
        )
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    if limit > 0:
        df = df.head(limit).copy()

    if len(df) < 100:
        raise ValueError(
            "Fewer than 100 usable readings."
        )

    return df


# ============================================================
# 3. SYNTHETIC DEMO DATA
# ============================================================

def make_demo(n=2400):

    rng = np.random.default_rng(12)

    t = np.arange(n)

    x = (
        22
        + 1.8 * np.sin(t / 170)
        + 0.4 * np.sin(t / 11)
        + rng.normal(0, 0.11, n)
    )

    return pd.DataFrame({
        "timestamp": (
            pd.Timestamp("2004-03-01")
            + pd.to_timedelta(t * 31, unit="s")
        ),
        "value": x
    })


# ============================================================
# 4. INJECT SYNTHETIC EVENTS
# ============================================================

def inject_events(
    clean,
    count,
    seed=42,
    amplitude_multiplier=1.0,
    width=3
):
    """
    Inject labelled synthetic events.

    seed:
        Controls random event positions.

    amplitude_multiplier:
        Controls event strength.

    width:
        Controls event duration in samples.

    Events are distributed across separate regions
    to prevent them from overlapping.
    """

    values = np.asarray(
        clean,
        dtype=float
    ).copy()

    if count == 0:
        return values, np.array([], dtype=int)

    if (
        width < 3
        or width > 31
        or width % 2 != 1
    ):
        raise ValueError(
            "Event width must be an odd number "
            "between 3 and 31."
        )

    if len(values) < 40 * count:
        raise ValueError(
            "Too many events for this signal."
        )

    rng = np.random.default_rng(seed)

    # Divide the signal into separate event regions.
    edges = np.linspace(
        0,
        len(values),
        count + 1,
        dtype=int
    )

    margin = max(
        10,
        width // 2 + 4
    )

    # Randomly select a position inside each region.
    centers = np.array(
        [
            int(
                rng.integers(
                    start + margin,
                    end - margin
                )
            )
            for start, end in zip(
                edges[:-1],
                edges[1:]
            )
        ],
        dtype=int
    )

    # Base event magnitude.
    base_amplitude = max(
        8 * np.std(np.diff(clean)),
        3 * np.std(clean),
        0.1
    )

    amplitude = (
        base_amplitude * amplitude_multiplier
    )

    half = width // 2

    # Triangular event shape.
    left = np.linspace(
        0.25,
        1.0,
        half + 1
    )

    shape = np.r_[
        left,
        left[-2::-1]
    ]

    # Inject each event.
    for j, center in enumerate(centers):

        polarity = (
            1 if j % 2 == 0 else -1
        )

        values[
            center - half:center + half + 1
        ] += (
            polarity * amplitude * shape
        )

    return values, centers


# ============================================================
# 5. IDENTIFY IMPORTANT SENSOR READINGS
# ============================================================

def important_indices(values, fraction=0.05):
    """
    Important data = top 5% of sensor readings
    ranked by absolute consecutive change.

    This is an importance proxy, not labelled faults.
    """

    changes = np.abs(
        np.diff(values)
    )

    count = max(
        1,
        int(np.ceil(
            fraction * len(changes)
        ))
    )

    indices = np.argsort(
        -changes,
        kind="stable"
    )[:count] + 1

    return indices


# ============================================================
# 6. DETECT EVENTS
# ============================================================

def detect_events(
    reconstruction,
    threshold,
    minimum_gap=7
):
    """
    Detect significant changes in reconstructed data.
    """

    changes = np.abs(
        np.diff(
            reconstruction,
            prepend=reconstruction[0]
        )
    )

    ranking = np.argsort(
        -changes,
        kind="stable"
    )

    selected = []

    for index in ranking:

        if changes[index] <= threshold:
            break

        if all(
            abs(int(index) - previous) > minimum_gap
            for previous in selected
        ):

            selected.append(int(index))

    return np.array(
        sorted(selected),
        dtype=int
    )


# ============================================================
# 7. EVENT RECALL AND F2-SCORE
# ============================================================

def event_recall_and_f2(
    true_centers,
    predictions,
    tolerance=4
):
    """
    Compare detected events against known event centers.

    One prediction may match only one true event.
    """

    matched = set()

    true_positive = 0

    for center in true_centers:

        candidates = sorted(
            (
                abs(int(pred) - int(center)),
                j
            )
            for j, pred in enumerate(predictions)
            if j not in matched
            and abs(int(pred) - int(center)) <= tolerance
        )

        if candidates:

            matched.add(
                candidates[0][1]
            )

            true_positive += 1

    precision = (
        true_positive / len(predictions)
        if len(predictions)
        else 0.0
    )

    recall = (
        true_positive / len(true_centers)
        if len(true_centers)
        else 0.0
    )

    f2 = (
        5 * precision * recall
        / (4 * precision + recall)
        if 4 * precision + recall
        else 0.0
    )

    return 100 * recall, f2


# ============================================================
# 8. VALIDATE ALGORITHM OUTPUT
# ============================================================

def validate_indices(
    indices,
    n,
    keep,
    method
):

    indices = np.asarray(indices)

    expected = max(
        2,
        min(n, round(n * keep))
    )

    if (
        indices.ndim != 1
        or not np.issubdtype(
            indices.dtype,
            np.integer
        )
        or len(indices) != expected
        or np.any(indices < 0)
        or np.any(indices >= n)
        or len(np.unique(indices)) != len(indices)
    ):

        raise ValueError(
            f"{method} returned invalid indices. "
            f"Expected {expected} unique samples."
        )

    return np.sort(
        indices.astype(int)
    )


# ============================================================
# 9. RUN COMPLETE ROBUSTNESS BENCHMARK
# ============================================================

def run_benchmark(df, args, output):

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    clean = df["value"].to_numpy(
        dtype=float
    )

    timestamps = df["timestamp"].reset_index(
        drop=True
    )

    seconds = (
        timestamps - timestamps.iloc[0]
    ).dt.total_seconds().to_numpy(float)

    if np.any(np.diff(seconds) <= 0):
        raise ValueError(
            "Timestamps must be strictly increasing."
        )

    # Use one fixed threshold for all experiments.
    threshold = max(
        2 * np.quantile(
            np.abs(np.diff(clean)),
            0.999
        ),
        1e-4
    )

    # Create all robustness testing combinations.
    if args.synthetic_events:

        conditions = list(
            product(
                args.event_seeds,
                args.event_amplitudes,
                args.event_widths
            )
        )

    else:

        conditions = [
            (42, 1.0, 3)
        ]

    records = []

    # ========================================================
    # RUN EACH ROBUSTNESS TRIAL
    # ========================================================

    for trial_number, (
        seed,
        amplitude,
        width
    ) in enumerate(conditions, start=1):

        signal, true_events = inject_events(
            clean,
            args.synthetic_events,
            seed,
            amplitude,
            width
        )

        important = important_indices(signal)

        print(
            f"Trial {trial_number}/{len(conditions)} | "
            f"Seed={seed} | "
            f"Amplitude={amplitude}x | "
            f"Width={width}"
        )

        # ====================================================
        # RUN ALL FIVE ALGORITHMS
        # ====================================================

        for keep in args.keep:

            for method in METHODS:

                timings = []

                previous = None

                # Repeat timing three times.
                for _ in range(3):

                    start = time.perf_counter()

                    chosen = choose_samples(
                        signal,
                        method,
                        keep,
                        args.window,
                        args.chunk,
                        args.hp_alpha,
                        args.pe_weight
                    )

                    elapsed_ms = (
                        time.perf_counter() - start
                    ) * 1000

                    indices = validate_indices(
                        chosen,
                        len(signal),
                        keep,
                        method
                    )

                    if (
                        previous is not None
                        and not np.array_equal(
                            previous,
                            indices
                        )
                    ):

                        raise ValueError(
                            f"{method} produced inconsistent "
                            "results across identical runs."
                        )

                    previous = indices

                    timings.append(elapsed_ms)

                # ============================================
                # RECONSTRUCT DOWNSAMPLED SIGNAL
                # ============================================

                reconstructed = np.interp(
                    seconds,
                    seconds[indices],
                    signal[indices]
                )

                # ============================================
                # METRIC 1: DATA REDUCTION
                # ============================================

                data_reduction = (
                    100 * (
                        1 - len(indices) / len(signal)
                    )
                )

                # ============================================
                # METRIC 2: IMPORTANT DATA RETAINED
                # ============================================

                important_retained = (
                    100 * np.isin(
                        important,
                        indices
                    ).mean()
                )

                # ============================================
                # METRIC 3 & 4: EVENT RECALL AND F2
                # ============================================

                recall_pct = np.nan
                f2 = np.nan

                if len(true_events):

                    detected = detect_events(
                        reconstructed,
                        threshold
                    )

                    recall_pct, f2 = (
                        event_recall_and_f2(
                            true_events,
                            detected
                        )
                    )

                # ============================================
                # METRIC 5: PRDN
                # ============================================

                denominator = np.sum(
                    (
                        signal - signal.mean()
                    ) ** 2
                )

                prdn = (
                    100 * np.sqrt(
                        np.sum(
                            (
                                signal - reconstructed
                            ) ** 2
                        ) / denominator
                    )
                    if denominator > 1e-12
                    else np.nan
                )

                # ============================================
                # METRIC 6: PROCESSING TIME
                # ============================================

                processing_time_ms = float(
                    np.median(timings)
                )

                # ============================================
                # RECORD TRIAL RESULTS
                # ============================================

                records.append({
                    "trial": trial_number,
                    "event_seed": seed,
                    "event_amplitude_multiplier": (
                        amplitude
                    ),
                    "event_width_samples": width,
                    "algorithm": method,
                    "data_reduction_pct": data_reduction,
                    "important_data_retained_pct": (
                        important_retained
                    ),
                    "event_recall_pct": recall_pct,
                    "F2_score": f2,
                    "PRDN_pct": float(prdn),
                    "selection_time_ms": processing_time_ms
                })

    # ========================================================
    # 10. SAVE INDIVIDUAL TRIAL RESULTS
    # ========================================================

    trials = pd.DataFrame(records)

    trials.to_csv(
        output / "benchmark_trials.csv",
        index=False
    )

    # ========================================================
    # 11. CALCULATE MEAN AND STANDARD DEVIATION
    # ========================================================

    keys = [
        "algorithm",
        "data_reduction_pct"
    ]

    grouped = trials.groupby(
        keys,
        sort=False
    )

    # Mean across all robustness trials.
    summary = grouped[
        METRICS[1:]
    ].mean().reset_index()

    # Standard deviation across trials.
    deviations = grouped[
        METRICS[1:]
    ].std().reset_index()

    deviations = deviations.rename(
        columns={
            metric: metric + "_std"
            for metric in METRICS[1:]
        }
    )

    summary = summary.merge(
        deviations,
        on=keys,
        how="left"
    )

    summary["trial_count"] = len(conditions)

    # Save averages for create_report.py.
    summary.to_csv(
        output / "benchmark_comparison.csv",
        index=False
    )

    # ========================================================
    # 12. SAVE EXPERIMENT CONFIGURATION
    # ========================================================

    settings = {
        "dataset": (
            "Synthetic demo"
            if args.demo
            else "Intel Lab Data"
        ),
        "sensor": args.sensor,
        "feature": args.feature,
        "samples": len(clean),
        "synthetic_event_count": args.synthetic_events,
        "event_seeds": args.event_seeds,
        "event_amplitude_multipliers": (
            args.event_amplitudes
        ),
        "event_width_samples": args.event_widths,
        "trial_count": len(conditions),
        "retention_fractions": args.keep,
        "important_data_definition": (
            "Top 5% absolute changes in tested signal"
        ),
        "reconstruction": (
            "Linear interpolation at original timestamps"
        ),
        "event_detector": (
            "Fixed first-difference threshold; "
            "tolerance +/-4 samples"
        ),
        "timing": (
            "Median of 3 offline selection runs per trial"
        ),
        "network_tested": False
    }

    (
        output / "benchmark_settings.json"
    ).write_text(
        json.dumps(
            settings,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # 13. DISPLAY RESULTS
    # ========================================================

    print(
        "\nAVERAGE BENCHMARK RESULTS "
        f"({len(conditions)} trials)"
    )

    print(
        summary[
            ["algorithm"] + METRICS
        ].round(3).to_string(
            index=False,
            na_rep="N/A"
        )
    )

    print(
        f"\nResults saved to: {output.resolve()}"
    )

    print(
        "\nbenchmark_trials.csv = individual trial results"
    )

    print(
        "benchmark_comparison.csv = average results"
    )

    print(
        "\nNOTE: Important data uses a high-change proxy."
    )

    print(
        "NOTE: Event Recall and F2 use synthetic events."
    )

    print(
        "NOTE: No MQTT, real network bandwidth or "
        "PID measurements are included."
    )


# ============================================================
# 14. MAIN PROGRAM
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Six-metric IoT downsampling robustness benchmark"
        )
    )

    parser.add_argument(
        "--demo",
        action="store_true"
    )

    parser.add_argument(
        "--data",
        default="data/data.txt.gz"
    )

    parser.add_argument(
        "--sensor",
        type=int,
        default=1
    )

    parser.add_argument(
        "--feature",
        choices=COLUMNS[4:],
        default="temperature"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=4000
    )

    parser.add_argument(
        "--keep",
        nargs="+",
        type=float,
        default=[0.5, 0.2, 0.1]
    )

    parser.add_argument(
        "--synthetic-events",
        type=int,
        default=0
    )

    # NEW: Random event positions.
    parser.add_argument(
        "--event-seeds",
        nargs="+",
        type=int,
        default=[42]
    )

    # NEW: Event intensity multipliers.
    parser.add_argument(
        "--event-amplitudes",
        nargs="+",
        type=float,
        default=[1.0]
    )

    # NEW: Event duration in sample readings.
    parser.add_argument(
        "--event-widths",
        nargs="+",
        type=int,
        default=[3]
    )

    parser.add_argument(
        "--window",
        type=int,
        default=48
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=256
    )

    parser.add_argument(
        "--hp-alpha",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--pe-weight",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--output",
        default="benchmark_results"
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE ARGUMENTS
    # ========================================================

    if not 1 <= args.sensor <= 54:
        parser.error(
            "--sensor must be between 1 and 54"
        )

    if (
        args.max_samples < 0
        or args.synthetic_events < 0
    ):
        parser.error(
            "--max-samples and --synthetic-events "
            "must be nonnegative"
        )

    if any(
        not 0 < keep <= 1
        for keep in args.keep
    ):
        parser.error(
            "Every --keep must be in (0, 1]"
        )

    if len({
        round(100 * keep)
        for keep in args.keep
    }) != len(args.keep):
        parser.error(
            "Retention levels must be unique "
            "after rounding to a percentage."
        )

    if args.window < 6 or args.chunk < 8:
        parser.error(
            "--window must be >= 6 and --chunk >= 8"
        )

    if (
        not 0 < args.hp_alpha <= 1
        or not 0 <= args.pe_weight <= 1
    ):
        parser.error(
            "Invalid high-pass alpha or hybrid weight."
        )

    if any(
        seed < 0
        for seed in args.event_seeds
    ):
        parser.error(
            "Event seeds must be nonnegative."
        )

    if any(
        amplitude <= 0
        for amplitude in args.event_amplitudes
    ):
        parser.error(
            "Event amplitudes must be positive."
        )

    if any(
        width < 3
        or width > 31
        or width % 2 == 0
        for width in args.event_widths
    ):
        parser.error(
            "Event widths must be odd numbers "
            "between 3 and 31."
        )

    if (
        len(set(args.event_seeds))
        != len(args.event_seeds)
        or len(set(args.event_amplitudes))
        != len(args.event_amplitudes)
        or len(set(args.event_widths))
        != len(args.event_widths)
    ):
        parser.error(
            "Do not repeat seeds, amplitudes or widths."
        )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    if args.demo:

        df = make_demo()
        folder = "SYNTHETIC_DEMO"

    else:

        df = load_intel(
            args.data,
            args.sensor,
            args.feature,
            args.max_samples
        )

        folder = (
            f"intel_mote_{args.sensor}_{args.feature}"
        )

    # ========================================================
    # SET OUTPUT FOLDER
    # ========================================================

    trial_count = (
        len(args.event_seeds)
        * len(args.event_amplitudes)
        * len(args.event_widths)
        if args.synthetic_events
        else 1
    )

    if args.synthetic_events:

        folder += (
            f"_synthetic_events_{args.synthetic_events}"
        )

    if trial_count > 1:
        folder += "_robustness"

    # ========================================================
    # START EXPERIMENT
    # ========================================================

    run_benchmark(
        df,
        args,
        Path(args.output) / folder
    )


if __name__ == "__main__":
    main()