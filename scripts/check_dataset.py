from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
if not DATA_PATH.exists():
	DATA_PATH = PROJECT_ROOT / "data.csv"

data = pd.read_csv(DATA_PATH, header=None)
print(data.iloc[:, -1].value_counts())
