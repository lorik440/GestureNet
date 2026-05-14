import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data.csv", header=None)
X    = data.iloc[:, :-1].values.astype(np.float32)
y    = data.iloc[:, -1].str.strip().str.lower().values

if X.shape[1] != 126:
    raise ValueError(f"Expected 126 features (63×2 hands), got {X.shape[1]}")

X, y = shuffle(X, y, random_state=42)
le   = LabelEncoder()
y_enc = le.fit_transform(y).astype(np.int64)
print("Classes:", list(le.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
X_test_t     = torch.tensor(X_test)
y_test_t     = torch.tensor(y_test)


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
        return self.net(x)


model     = GestureNet(X.shape[1], len(le.classes_))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
criterion = nn.CrossEntropyLoss()
best_acc, no_improve = 0, 0

for epoch in range(1, 121):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        acc = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()

    print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc, no_improve = acc, 0
        torch.save(model.state_dict(), "model.pth")
        print(f"  Saved best: {best_acc:.4f}")
    else:
        no_improve += 1
        if no_improve >= 15:
            print(f"Early stopping at epoch {epoch}")
            break

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("model_config.pkl", "wb") as f:
    pickle.dump({"input_dim": X.shape[1], "num_classes": len(le.classes_)}, f)

print(f"Done. Best accuracy: {best_acc:.4f}")
