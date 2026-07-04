from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, top_k_accuracy_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
if not DATA_PATH.exists():
    # fallback to legacy root
    DATA_PATH = PROJECT_ROOT / "data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = PROJECT_ROOT

with open(MODELS_DIR / "model_config.pkl", "rb") as f:
    cfg = pickle.load(f)
with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


class GestureNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 128, bias=False),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 64),                    nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


def load_model():
    m = GestureNet(cfg["input_dim"], cfg["num_classes"])
    m.load_state_dict(torch.load(MODELS_DIR / "model.pth", map_location="cpu", weights_only=False))
    m.eval()
    return m


def predict(model, X):
    with torch.no_grad():
        logits = model(torch.tensor(X))
        probs  = torch.softmax(logits, dim=1).numpy()
        preds  = probs.argmax(axis=1)
    return preds, probs


def print_section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# Load data
data = pd.read_csv(DATA_PATH, header=None)
X    = data.iloc[:, :-1].values.astype(np.float32)
y    = data.iloc[:, -1].str.strip().str.lower().values
X, y = shuffle(X, y, random_state=42)
y_enc = le.transform(y).astype(np.int64)
classes = le.classes_

print_section("DATASET SUMMARY")
print(f"  Total samples : {len(X)}")
print(f"  Features      : {X.shape[1]}")
print(f"  Classes ({len(classes)}): {list(classes)}")

# Held-out test set
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
model = load_model()
preds, probs = predict(model, X_test)

acc = (preds == y_test).mean()
print(f"\n  Overall accuracy : {acc:.4f}  ({acc:.2%})")

if len(classes) > 2:
    top2 = top_k_accuracy_score(y_test, probs, k=2)
    print(f"  Top-2 accuracy   : {top2:.4f}  ({top2:.2%})")

print(f"\n  Per-class report:\n")
print(classification_report(y_test, preds, target_names=classes, digits=3))

cm = confusion_matrix(y_test, preds)
col_w = max(len(c) for c in classes) + 2
header = " " * col_w + "".join(f"{c:>{col_w}}" for c in classes)
print(f"\n  Predicted →")
print(f"  {header}")
for i, row in enumerate(cm):
    row_str = "".join(f"{v:>{col_w}}" for v in row)
    print(f"  {classes[i]:<{col_w}}{row_str}")
