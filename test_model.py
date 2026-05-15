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

with open("model_config.pkl", "rb") as f:
    cfg = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
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
    m.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=False))
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


# ── Load data ─────────────────────────────────────────────────────────────────
data = pd.read_csv("data.csv", header=None)
X    = data.iloc[:, :-1].values.astype(np.float32)
y    = data.iloc[:, -1].str.strip().str.lower().values
X, y = shuffle(X, y, random_state=42)
y_enc = le.transform(y).astype(np.int64)
classes = le.classes_

print_section("DATASET SUMMARY")
print(f"  Total samples : {len(X)}")
print(f"  Features      : {X.shape[1]}")
print(f"  Classes ({len(classes)}): {list(classes)}")
print("\n  Samples per class:")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    bar = "█" * (cnt // 10)
    print(f"    {cls:<20} {cnt:>4}  {bar}")

# ── 1. Held-out test set (80/20 stratified) ───────────────────────────────────
print_section("1. HELD-OUT TEST SET  (80/20 stratified split)")
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

# ── 2. Confusion matrix ───────────────────────────────────────────────────────
print_section("2. CONFUSION MATRIX")
cm = confusion_matrix(y_test, preds)
col_w = max(len(c) for c in classes) + 2
header = " " * col_w + "".join(f"{c:>{col_w}}" for c in classes)
print(f"\n  Predicted →")
print(f"  {header}")
for i, row in enumerate(cm):
    row_str = "".join(f"{v:>{col_w}}" for v in row)
    print(f"  {classes[i]:<{col_w}}{row_str}")

# Most confused pairs
print("\n  Most confused pairs:")
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
for _ in range(min(3, len(classes))):
    i, j = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
    if cm_no_diag[i, j] == 0:
        break
    print(f"    '{classes[i]}' predicted as '{classes[j]}': {cm_no_diag[i,j]} times")
    cm_no_diag[i, j] = 0

# ── 3. 5-fold cross-validation ────────────────────────────────────────────────
print_section("3. 5-FOLD CROSS-VALIDATION  (most reliable estimate)")
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc), 1):
    m = load_model()
    p, _ = predict(m, X[val_idx])
    fold_acc = (p == y_enc[val_idx]).mean()
    fold_accs.append(fold_acc)
    print(f"  Fold {fold}: {fold_acc:.4f}")

mean_acc = np.mean(fold_accs)
std_acc  = np.std(fold_accs)
print(f"\n  Mean accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
print(f"  Min / Max     : {min(fold_accs):.4f} / {max(fold_accs):.4f}")

# ── 4. Confidence analysis ────────────────────────────────────────────────────
print_section("4. CONFIDENCE ANALYSIS")
_, all_probs = predict(model, X_test)
top_conf     = all_probs.max(axis=1)
correct_mask = preds == y_test

print(f"\n  Avg confidence when CORRECT : {top_conf[correct_mask].mean():.3f}")
print(f"  Avg confidence when WRONG   : {top_conf[~correct_mask].mean():.3f}" if (~correct_mask).any() else "  No wrong predictions!")

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
print(f"\n  Coverage vs accuracy at confidence thresholds:")
print(f"  {'Threshold':>10}  {'Coverage':>10}  {'Accuracy':>10}")
for t in thresholds:
    mask     = top_conf >= t
    coverage = mask.mean()
    t_acc    = (preds[mask] == y_test[mask]).mean() if mask.any() else 0.0
    print(f"  {t:>10.0%}  {coverage:>10.1%}  {t_acc:>10.2%}")

# ── 5. Per-class confidence ───────────────────────────────────────────────────
print_section("5. PER-CLASS CONFIDENCE")
print(f"\n  {'Class':<20} {'Avg conf':>10}  {'Accuracy':>10}  {'Samples':>8}")
for i, cls in enumerate(classes):
    mask = y_test == i
    if not mask.any():
        continue
    avg_conf = top_conf[mask].mean()
    cls_acc  = (preds[mask] == y_test[mask]).mean()
    print(f"  {cls:<20} {avg_conf:>10.3f}  {cls_acc:>10.2%}  {mask.sum():>8}")

print(f"\n{'─'*60}")
print(f"  SUMMARY")
print(f"{'─'*60}")
print(f"  Held-out accuracy  : {acc:.2%}")
print(f"  Cross-val accuracy : {mean_acc:.2%} ± {std_acc:.2%}")
print(f"  Recommended threshold for deployment: 70%")
print(f"{'─'*60}\n")
