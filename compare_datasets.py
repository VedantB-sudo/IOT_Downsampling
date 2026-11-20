
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("benchmark_results")
OUTPUT = BASE / "Dataset_Comparison.xlsx"

files = {
    "PAMAP2": BASE / "pamap2/six_methods_full/benchmark_summary.csv",
    "Intel": BASE / "intel/full_dataset/benchmark_summary.csv"
}

results = []

for dataset, path in files.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path)
    df["Dataset"] = dataset
    results.append(df)

data = pd.concat(results, ignore_index=True)

metrics = {
    "Best PRDN": ("PRDN_pct_mean", "min"),
    "Best RMSE": ("RMSE_mean", "min"),
    "Best MAE": ("MAE_mean", "min"),
    "Best P95": ("P95_retained_pct_mean", "max"),
    "Fastest": ("selection_time_ms_mean", "min")
}

winners = []

for (dataset, sensor, reduction), group in data.groupby(
    ["Dataset", "sensor_type", "target_reduction_pct"]
):
    row = {
        "Dataset": dataset,
        "Sensor": sensor,
        "Reduction %": reduction
    }

    for name, (metric, direction) in metrics.items():
        valid = group.dropna(subset=[metric])

        if valid.empty:
            row[name] = "N/A"
            continue

        value = valid[metric].min() if direction == "min" else valid[metric].max()
        row[name] = ", ".join(
            valid.loc[np.isclose(valid[metric], value), "algorithm"]
        )

    valid = group.dropna(
        subset=["PRDN_pct_mean", "P95_retained_pct_mean"]
    )

    pareto = []

    for _, candidate in valid.iterrows():
        dominated = (
            (valid["PRDN_pct_mean"] <= candidate["PRDN_pct_mean"])
            & (valid["P95_retained_pct_mean"] >= candidate["P95_retained_pct_mean"])
            & (
                (valid["PRDN_pct_mean"] < candidate["PRDN_pct_mean"])
                | (valid["P95_retained_pct_mean"] > candidate["P95_retained_pct_mean"])
            )
        ).any()

        if not dominated:
            pareto.append(candidate["algorithm"])

    row["Pareto Trade-off"] = ", ".join(pareto) or "N/A"
    winners.append(row)

winners = pd.DataFrame(winners)

with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    data.to_excel(writer, sheet_name="All Results", index=False)
    winners.to_excel(writer, sheet_name="Algorithm Winners", index=False)

    for dataset in files:
        for label, filename in [
            ("PID", "pid_comparison.csv"),
            ("Delivery", "delivery_comparison.csv")
        ]:
            path = BASE / dataset.lower() / filename

            if path.exists():
                pd.read_csv(path).to_excel(
                    writer,
                    sheet_name=f"{dataset} {label}",
                    index=False
                )

print("\nALGORITHM WINNERS")
print(winners.to_string(index=False))
print(f"\nSaved: {OUTPUT}")