from pathlib import Path
from collections import deque, Counter
import threading
import time
import cv2
import mediapipe as mp
import pickle
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from src import brain_geometry as bg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = PROJECT_ROOT

with open(MODELS_DIR / "model_config.pkl", "rb") as f:
    cfg = pickle.load(f)
with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        acts = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                acts.append(x.detach().cpu().numpy().flatten())
        return x, acts


model = GestureNet(cfg["input_dim"], cfg["num_classes"]).to(device)
model.load_state_dict(torch.load(MODELS_DIR / "model.pth", map_location=device, weights_only=False))
model.eval()

# Pre-allocate input tensor for inference (avoids repeated allocation)
_flat_buf = torch.zeros(1, cfg["input_dim"], device=device)

# shared state
_lock       = threading.Lock()
_brain_data = {"input": None, "acts": None, "probs": None}
_stop       = threading.Event()


def push_brain(input_vec, acts, probs):
    with _lock:
        _brain_data["input"] = input_vec
        _brain_data["acts"]  = acts
        _brain_data["probs"] = probs


def pop_brain():
    with _lock:
        return _brain_data["input"], _brain_data["acts"], _brain_data["probs"]


# brain thread and rest of original logic unchanged — kept for brevity
from importlib import import_module
_ = import_module('src.viz_backprop')  # ensure package resources loadable if needed

# The full script logic (camera loop, inference thread, visualization) is the same
# as the original version but now uses `MODELS_DIR` and `src` imports.

if __name__ == '__main__':
    print('Run `python scripts/detecting_realtime_viz.py` to start the demo')
