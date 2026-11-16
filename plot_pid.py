
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

file = Path("benchmark_results/pamap2/pid_comparison.csv")

if not file.exists():
    raise SystemExit("Run benchmark_pid.py first")

df = pd.read_csv(file)
results = df.pivot(index="Algorithm", columns="Mode", values="PRDN %")

results.plot(kind="bar", figsize=(10, 5))
plt.title("PAMAP2: PID vs Fixed Retention")
plt.ylabel("PRDN (%) - Lower is better")
plt.xlabel("")
plt.xticks(rotation=25, ha="right")
plt.legend(title="Mode")
plt.tight_layout()

plt.savefig(file.with_name("pid_comparison.png"), dpi=200)
plt.show()

print("Comparison graph saved successfully")