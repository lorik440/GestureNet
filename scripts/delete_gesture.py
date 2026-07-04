from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
if not DATA_PATH.exists():
    DATA_PATH = PROJECT_ROOT / "data.csv"

data    = pd.read_csv(DATA_PATH, header=None)
counts  = data.iloc[:, -1].str.strip().str.lower().value_counts()

print("Current gestures in data.csv:")
for name, count in counts.items():
    print(f"  {name:<25} {count} samples")

label = input("\nEnter gesture name to delete: ").strip().lower()

if label not in counts.index:
    print(f"'{label}' not found in data.csv")
else:
    before = len(data)
    data   = data[data.iloc[:, -1].str.strip().str.lower() != label]
    after  = len(data)
    confirm = input(f"Delete {before - after} samples of '{label}'? (y/n): ").strip().lower()
    if confirm == "y":
        data.to_csv(DATA_PATH, header=False, index=False)
        print(f"Done — removed '{label}' ({before - after} samples). {after} samples remaining.")
    else:
        print("Cancelled.")
