"""
Forward + Backward pass visualizer for GestureNet.
- SPACE : run a new random sample
- Q     : quit
"""
import pickle
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import open3d as o3d
import pandas as pd
import brain_geometry as bg

# ── Load ──────────────────────────────────────────────────────────────────────
with open("model_config.pkl", "rb") as f:
    cfg = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

data  = pd.read_csv("data.csv", header=None)
X_all = data.iloc[:, :-1].values.astype(np.float32)
y_all = data.iloc[:, -1].str.strip().str.lower().values


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
                acts.append(x.clone())
        return x, acts


model = GestureNet(cfg["input_dim"], cfg["num_classes"])
model.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=True))
model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.0

# ── Geometry ──────────────────────────────────────────────────────────────────
layers  = bg.layer_sizes(cfg)
weights = bg.collect_linear_weights(model)
nodes   = bg.build_nodes(layers)
offs    = bg.layer_offsets(layers)
lines, base_lc, w_abs, _ = bg.build_strongest_edges_balanced(weights, layers, edges_per_layer=60)
base_lc = np.clip(base_lc, 0.0, 1.0)
w_abs   = np.asarray(w_abs, dtype=np.float32)
src_i   = lines[:, 0]
dst_i   = lines[:, 1]
n_nodes = len(nodes)
n_edges = len(lines)

SPHERE_R   = 0.18
IDLE_COLOR = np.array([0.08, 0.09, 0.18])
FWD_COLOR  = np.array([0.22, 0.82, 0.98])
BWD_COLOR  = np.array([0.98, 0.48, 0.42])
STEP_DT    = 0.35

sphere_mesh = o3d.geometry.TriangleMesh()
for p in nodes:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_R, resolution=8)
    s.translate(p)
    sphere_mesh += s
sphere_mesh.compute_vertex_normals()
n_verts_per = len(o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_R, resolution=8).vertices)

ls = o3d.geometry.LineSet()
ls.points = o3d.utility.Vector3dVector(nodes)
ls.lines  = o3d.utility.Vector2iVector(lines)
ls.colors = o3d.utility.Vector3dVector(np.full((n_edges, 3), 0.05))

# ── Open3D window ─────────────────────────────────────────────────────────────
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("Forward & Backprop  |  SPACE = new sample  |  Q = quit",
                  width=1100, height=900)
vis.add_geometry(sphere_mesh)
vis.add_geometry(ls)

ro = vis.get_render_option()
ro.background_color    = np.array([0.02, 0.02, 0.08])
ro.mesh_show_back_face = True

ctr = vis.get_view_control()
ctr.set_lookat(np.mean(nodes, axis=0))
ctr.set_up([0, 1, 0])
ctr.set_front([0, 0, -1])
ctr.set_zoom(0.45)

run_pass  = [False]
quit_flag = [False]

vis.register_key_callback(ord(" "), lambda v: run_pass.__setitem__(0, True) or False)
vis.register_key_callback(ord("Q"), lambda v: quit_flag.__setitem__(0, True) or False)
vis.register_key_callback(ord("q"), lambda v: quit_flag.__setitem__(0, True) or False)

# ── Drawing helpers ───────────────────────────────────────────────────────────
def node_colors_for_layer(vals, active_li, phase):
    col   = FWD_COLOR if phase == "fwd" else BWD_COLOR
    colors = np.tile(IDLE_COLOR, (n_nodes, 1))
    for li, ln in enumerate(layers):
        sl = vals[offs[li]:offs[li]+ln]
        t  = np.clip(sl / (sl.max() + 1e-8), 0, 1) ** 0.7
        c  = col if li == active_li else np.array([0.20, 0.28, 0.55])
        colors[offs[li]:offs[li]+ln] = IDLE_COLOR * (1 - t[:, None]) + c * t[:, None]
    return np.clip(colors, 0, 1)


def edge_colors_for_pair(node_vals, li, phase):
    colors = np.full((n_edges, 3), 0.05)
    mask   = (src_i >= offs[li]) & (src_i < offs[li] + layers[li])
    if not mask.any():
        return colors
    flow   = w_abs[mask] * np.maximum(node_vals[src_i[mask]], 0) * np.maximum(node_vals[dst_i[mask]], 0)
    flow_n = np.clip(flow / (flow.max() + 1e-8), 0, 1)
    col    = FWD_COLOR if phase == "fwd" else BWD_COLOR
    colors[mask] = col * flow_n[:, None]
    return colors


def update_vis(nc, ec):
    sphere_mesh.vertex_colors = o3d.utility.Vector3dVector(np.repeat(nc, n_verts_per, axis=0))
    ls.colors = o3d.utility.Vector3dVector(ec)
    vis.update_geometry(sphere_mesh)
    vis.update_geometry(ls)
    vis.poll_events()
    vis.update_renderer()


def fade_to_idle():
    for _ in range(8):
        update_vis(np.tile(IDLE_COLOR, (n_nodes, 1)), np.full((n_edges, 3), 0.05))
        time.sleep(0.04)


# ── Info panel ────────────────────────────────────────────────────────────────
PANEL_W = 380
panel_data = {
    "phase": "idle", "true_label": "", "pred_label": "",
    "correct": None, "loss": 0.0, "probs": None,
    "layer": -1, "n_layers": len(layers),
}


def draw_panel():
    p         = panel_data
    n_classes = len(le.classes_)
    PANEL_H   = max(520, 270 + n_classes * 26 + 20)
    img       = np.full((PANEL_H, PANEL_W, 3), (12, 12, 22), dtype=np.uint8)
    phase     = p["phase"]

    def txt(text, pos, scale=0.55, color=(200,200,200), bold=False):
        x, y = pos
        cv2.putText(img, text, (x+1,y+1), cv2.FONT_HERSHEY_DUPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y),    cv2.FONT_HERSHEY_DUPLEX, scale, color, 2 if bold else 1, cv2.LINE_AA)

    phase_col = {"idle":(120,120,120), "fwd":(80,210,255), "bwd":(255,120,80)}.get(phase,(200,200,200))
    phase_str = {"idle":"Idle — press SPACE", "fwd":"▶  Forward Pass", "bwd":"◀  Backward Pass"}.get(phase, phase)
    txt(phase_str, (14, 32), scale=0.65, color=phase_col, bold=True)
    cv2.line(img, (10,42), (PANEL_W-10,42), (40,40,60), 1)

    if phase == "idle":
        txt("Press SPACE to run a sample", (14, 80), color=(140,140,160))
        cv2.imshow("Pass Info", img)
        cv2.waitKey(1)
        return

    correct = p["correct"]
    txt("True:",      (14,  72), color=(160,160,180))
    txt("Predicted:", (14, 100), color=(160,160,180))
    txt(p["true_label"], (130,  72), scale=0.65, color=(100,255,160), bold=True)
    txt(p["pred_label"], (130, 100), scale=0.65,
        color=(100,255,160) if correct else (80,100,255), bold=True)

    if correct is not None:
        txt("✓  CORRECT" if correct else "✗  WRONG", (14, 130), scale=0.7,
            color=(60,220,100) if correct else (80,100,255), bold=True)

    if phase == "bwd":
        loss_col = (255,180,80) if p["loss"] > 0.5 else (100,220,140)
        txt(f"Loss: {p['loss']:.4f}", (14, 158), scale=0.6, color=loss_col)

    # Layer progress
    cv2.line(img, (10,172), (PANEL_W-10,172), (40,40,60), 1)
    txt("Layer:", (14, 192), color=(140,140,160))
    bar_w = PANEL_W - 28
    if p["layer"] >= 0:
        filled = int(bar_w * (p["layer"] + 1) / p["n_layers"])
        cv2.rectangle(img, (14,200), (14+bar_w, 218), (30,30,50), -1)
        cv2.rectangle(img, (14,200), (14+filled, 218), phase_col, -1)
        txt(f"{p['layer']+1}/{p['n_layers']}", (14+bar_w//2-18, 215), scale=0.42, color=(0,0,0))

    # Confidence bars
    cv2.line(img, (10,232), (PANEL_W-10,232), (40,40,60), 1)
    txt("Confidence:", (14, 252), color=(140,140,160))
    probs = p["probs"]
    if probs is not None:
        for ni, prob in enumerate(probs):
            y       = 268 + ni * 26
            if y + 20 > PANEL_H:
                break
            lbl     = le.classes_[ni] if ni < len(le.classes_) else str(ni)
            is_top  = ni == int(np.argmax(probs))
            is_true = lbl == p["true_label"]
            bar_len = int(prob * bar_w)
            bar_col = (60,200,100) if is_true else ((80,100,255) if is_top else (60,80,180))
            cv2.rectangle(img, (14,y), (14+bar_w, y+18), (25,25,45), -1)
            if bar_len > 0:
                cv2.rectangle(img, (14,y), (14+bar_len, y+18), bar_col, -1)
            txt(lbl, (16, y+14), scale=0.42, color=(60,220,100) if is_true else (180,180,200))
            txt(f"{prob:.0%}", (PANEL_W-50, y+14), scale=0.42,
                color=(255,220,80) if is_top else (140,140,160))

    cv2.imshow("Pass Info", img)
    cv2.waitKey(1)


cv2.namedWindow("Pass Info", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pass Info", PANEL_W, 520)


# ── Animation ─────────────────────────────────────────────────────────────────
def run_animation():
    idx   = np.random.randint(len(X_all))
    x_np  = X_all[idx]
    label = y_all[idx]

    # Forward pass — collect activations
    layer_acts = [np.abs(x_np)]
    x_fwd = torch.tensor(x_np).unsqueeze(0)
    with torch.no_grad():
        for layer in model.net:
            x_fwd = layer(x_fwd)
            if isinstance(layer, nn.ReLU):
                layer_acts.append(x_fwd.numpy().flatten())
        layer_acts.append(torch.softmax(x_fwd, dim=1).numpy().flatten())

    probs    = layer_acts[-1]
    pred_lbl = le.classes_[int(probs.argmax())]
    correct  = pred_lbl == label

    panel_data.update({
        "phase": "fwd", "true_label": label, "pred_label": pred_lbl,
        "correct": correct, "probs": None, "layer": 0, "n_layers": len(layers)
    })
    print(f"\nForward  true='{label}'  pred='{pred_lbl}'  {'✓' if correct else '✗'}")

    for li in range(len(layers)):
        nv = np.zeros(n_nodes, dtype=np.float32)
        for prev in range(li + 1):
            a = layer_acts[prev]
            n = min(len(a), layers[prev])
            nv[offs[prev]:offs[prev]+n] = np.abs(a[:n])

        panel_data["layer"] = li
        panel_data["probs"] = probs if li == len(layers) - 1 else None
        draw_panel()
        update_vis(
            node_colors_for_layer(nv, li, "fwd"),
            edge_colors_for_pair(nv, li-1, "fwd") if li > 0 else np.full((n_edges,3), 0.05)
        )
        time.sleep(STEP_DT)

    panel_data["probs"] = probs
    draw_panel()
    time.sleep(0.5)

    # Backward pass — collect gradients
    model.zero_grad()
    x_grad = torch.tensor(x_np).unsqueeze(0).requires_grad_(True)
    out, _ = model(x_grad)
    loss   = nn.CrossEntropyLoss()(out, torch.tensor([le.transform([label])[0]]))
    loss.backward()

    grad_acts = [np.abs(x_grad.grad.numpy().flatten())]
    for m in model.net:
        if isinstance(m, nn.Linear) and m.weight.grad is not None:
            grad_acts.append(np.abs(m.weight.grad.numpy()).mean(axis=1))

    panel_data.update({"phase": "bwd", "loss": loss.item()})
    print(f"Backward loss={loss.item():.4f}")

    for li in range(len(layers) - 1, -1, -1):
        nv = np.zeros(n_nodes, dtype=np.float32)
        a  = grad_acts[min(li, len(grad_acts)-1)]
        n  = min(len(a), layers[li])
        nv[offs[li]:offs[li]+n] = np.abs(a[:n])

        panel_data["layer"] = li
        draw_panel()
        update_vis(
            node_colors_for_layer(nv, li, "bwd"),
            edge_colors_for_pair(nv, li, "bwd") if li < len(layers)-1 else np.full((n_edges,3), 0.05)
        )
        time.sleep(STEP_DT)

    fade_to_idle()
    panel_data["phase"] = "idle"
    draw_panel()
    print("Done — press SPACE for another sample.\n")


# ── Main loop ─────────────────────────────────────────────────────────────────
print("\nForward & Backprop Visualizer — press SPACE to start, Q to quit.\n")

while not quit_flag[0]:
    if not vis.poll_events():
        break
    vis.update_renderer()
    draw_panel()
    if run_pass[0]:
        run_pass[0] = False
        run_animation()

vis.destroy_window()
cv2.destroyAllWindows()
