
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from algorithms import METHODS, choose_samples
from signal_quality import evaluate_fidelity
from intel_common import load_data, get_windows, FEATURES, OUT

parser = argparse.ArgumentParser()
parser.add_argument("--max-windows", type=int, default=0)
parser.add_argument("--output", type=Path, default=OUT / "full_dataset")
args = parser.parse_args()

if args.max_windows < 0:
    parser.error("--max-windows cannot be negative")
if args.max_windows and args.output == OUT / "full_dataset":
    parser.error("Use --output benchmark_results/intel/demo for a test run")

args.output.mkdir(parents=True, exist_ok=True)
settings = {"max_windows": args.max_windows, "reductions": [50, 60, 70, 80],
            "methods": METHODS, "window": 1000, "minimum": 256, "gap": 300}
config = args.output / "benchmark_settings.json"
if config.exists() and json.loads(config.read_text()) != settings:
    parser.error("Different settings found. Choose another output folder.")
config.write_text(json.dumps(settings, indent=2))

csv_file = args.output / "benchmark_trials.csv"
keys = ["sensor_type", "mote", "source_start_row", "algorithm",
        "target_reduction_pct"]
done = set()

if csv_file.exists():
    old = pd.read_csv(csv_file)
    done = set(map(tuple, old[keys].astype(str).to_numpy()))
    print(f"Resuming {len(done)} completed comparisons")

df = load_data()

for feature in FEATURES:
    windows = get_windows(df, feature)
    if args.max_windows:
        windows = windows[:args.max_windows]

    print(f"\n{feature}: {len(windows)} windows")
    rows = []

    for mote, start, x in windows:
        for reduction in [50, 60, 70, 80]:
            keep = (100 - reduction) / 100

            for method in METHODS:
                key = (str(feature), str(mote), str(start),
                       str(method), str(reduction))
                if key in done:
                    continue

                begin = time.perf_counter()
                idx = np.asarray(choose_samples(x, method, keep))
                elapsed = (time.perf_counter() - begin) * 1000

                expected = max(2, min(len(x), round(len(x) * keep)))
                if (len(idx) != expected or len(np.unique(idx)) != expected
                        or np.any(idx < 0) or np.any(idx >= len(x))):
                    raise ValueError(f"Invalid selection: {feature}, {method}")

                metrics = evaluate_fidelity(x, idx)
                rows.append({
                    "sensor_type": feature, "mote": mote,
                    "source_start_row": start,
                    "original_samples": len(x), "algorithm": method,
                    "target_reduction_pct": reduction,
                    "data_reduction_pct": 100 * (1 - len(idx) / len(x)),
                    "retained_samples": len(idx),
                    **metrics, "selection_time_ms": elapsed
                })

    if rows:
        pd.DataFrame(rows).to_csv(
            csv_file, mode="a", index=False, header=not csv_file.exists()
        )
        print(f"Saved {len(rows)} new comparisons")

if not csv_file.exists():
    raise ValueError("No benchmark results produced")

results = pd.read_csv(csv_file)
metrics = ["RMSE", "MAE", "PRDN_pct", "P95_retained_pct",
           "P99_retained_pct", "selection_time_ms"]

groups = ["sensor_type", "target_reduction_pct", "algorithm"]
summary = results.groupby(groups)[metrics].agg(["mean", "std"])
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary["window_count"] = results.groupby(groups).size()
summary.reset_index().to_csv(args.output / "benchmark_summary.csv", index=False)

print("\nIntel fixed-budget benchmark complete.")
print("Results:", args.output)