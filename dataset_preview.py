
import pandas as pd

path = "data/PAMAP2_Dataset/Protocol/subject101.dat"

df = pd.read_csv(path, sep=r"\s+", header=None)

preview = df.iloc[:10, [0, 1, 4, 5, 6, 10]]

preview.columns = [
    "Timestamp",
    "Activity ID",
    "Hand Acc X",
    "Hand Acc Y",
    "Hand Acc Z",
    "Hand Gyro X"
]

print("\nPAMAP2 DATASET PREVIEW\n")
print(preview.to_string(index=False))

print("\nTotal rows:", len(df))
print("Total columns:", len(df.columns))