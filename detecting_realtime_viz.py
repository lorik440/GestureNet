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
import brain_geometry as bg

with open("model_config.pkl", "rb") as f:
    cfg = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GestureNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 64),        nn.ReLU(),
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
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=False))
model.eval()

# Pre-allocate input tensor for inference (avoids repeated allocation)
_flat_buf = torch.zeros(1, cfg["input_dim"], device=device)

# ── Shared state ──────────────────────────────────────────────────────────────
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


# ── 3D brain thread ───────────────────────────────────────────────────────────
def brain_thread():
    layers  = bg.layer_sizes(cfg)
    weights = bg.collect_linear_weights(model)
    nodes   = bg.build_nodes(layers)
    lines, base_lc, w_abs, _ = bg.build_strongest_edges_balanced(
        weights, layers, edges_per_layer=320   # more edges = more visible
    )
    base_lc = np.clip(base_lc, 0.0, 1.0)
    w_abs   = np.asarray(w_abs, dtype=np.float32)
    src_i   = lines[:, 0]
    dst_i   = lines[:, 1]   # precomputed once
    n_nodes, n_edges = len(nodes), len(lines)
    offs    = bg.layer_offsets(layers)

    # Precompute per-layer edge masks so we don't recompute every frame
    layer_masks = [
        (src_i >= offs[li]) & (src_i < offs[li] + layers[li])
        for li in range(len(layers) - 1)
    ]

    IDLE_COLOR = np.array([0.22, 0.35, 0.50])   # bright idle — visible in dark room
    SPHERE_R   = 0.28

    sphere_mesh = o3d.geometry.TriangleMesh()
    for p in nodes:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_R, resolution=6)
        s.translate(p)
        sphere_mesh += s
    sphere_mesh.compute_vertex_normals()
    n_verts_per = len(o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_R, resolution=6).vertices)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(nodes)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.full((n_edges, 3), 0.0))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("GestureNet — Live 3D Brain", width=960, height=900)
    vis.add_geometry(sphere_mesh)
    vis.add_geometry(ls)

    # 3D text labels
    label_meshes = []
    for ni in range(layers[-1]):
        name   = le.classes_[ni] if ni < len(le.classes_) else str(ni)
        p      = nodes[offs[-1] + ni]
        anchor = np.array(p) + np.array([0.0, 0.0, SPHERE_R * 3.0])
        try:
            tm = o3d.t.geometry.TriangleMesh.create_text(name, depth=0.04)
            m  = tm.to_legacy()
            ext = float(np.max(m.get_axis_aligned_bounding_box().get_extent())) + 1e-6
            m.scale(0.55 / ext, center=m.get_center())
            m.translate(anchor - np.array([m.get_center()[0], m.get_center()[1], 0.0]))
            m.compute_vertex_normals()
            m.paint_uniform_color([1.0, 0.55, 0.85])
            label_meshes.append(m)
            vis.add_geometry(m)
        except Exception:
            label_meshes.append(None)

    ro = vis.get_render_option()
    ro.background_color    = np.array([0.02, 0.02, 0.08])
    ro.mesh_show_back_face = True

    ctr = vis.get_view_control()
    ctr.set_lookat(np.mean(nodes, axis=0))
    ctr.set_up([0, 1, 0])
    ctr.set_front([0, 0, -1])
    ctr.set_zoom(0.5)

    smooth_vals = np.zeros(n_nodes, dtype=np.float32)
    ALPHA  = 0.60
    min_dt = 1.0 / 30.0
    last_t = 0.0
    # Pre-allocate arrays reused every frame
    final_colors    = np.empty((n_nodes, 3), dtype=np.float64)
    color_buf       = np.empty((n_nodes, 3), dtype=np.float64)   # for values_to_colors
    all_vert_colors = np.empty((n_nodes * n_verts_per, 3), dtype=np.float64)
    edge_col        = np.full((n_edges, 3), 0.0, dtype=np.float64)
    flow_n          = np.zeros(n_edges, dtype=np.float32)
    # Pre-tile label colors so we don't allocate each frame
    label_n_verts   = [len(m.vertices) if m is not None else 0 for m in label_meshes]
    col_dim         = [np.tile([0.50, 0.55, 0.85], (nv, 1)) for nv in label_n_verts]
    col_win         = [np.tile([0.0,  1.0,  0.5 ], (nv, 1)) for nv in label_n_verts]
    prev_winner     = -2   # force first update

    while not _stop.is_set():
        if not vis.poll_events():
            break

        now = time.perf_counter()
        if now - last_t < min_dt:
            time.sleep(0.001)
            continue
        last_t = now

        input_vec, acts, probs = pop_brain()
        target      = bg.pack_node_values(input_vec, acts, layers, class_probs=probs, offs=offs)
        smooth_vals += ALPHA * (target - smooth_vals)
        active       = float(smooth_vals.max()) > 1e-5

        # Node colors — reuse color_buf and final_colors
        bg.values_to_colors_rgb01(smooth_vals, layers,
                                   output_layer_is_probability=(probs is not None),
                                   offs=offs, out=color_buf)
        # smooth_vals already in [0,1] per-layer — use directly as boost
        b = smooth_vals[:, None].astype(np.float64)
        np.clip(color_buf * b * 2.0, 0, 1, out=final_colors)
        final_colors += IDLE_COLOR * (1.0 - b)
        np.clip(final_colors, 0, 1, out=final_colors)

        winner = int(np.argmax(probs)) if probs is not None and active else -1
        if active and winner >= 0:
            pulse = 0.5 + 0.5 * np.sin(now * 6.0)
            np.clip(final_colors[offs[-1] + winner] * (1.6 + pulse), 0, 1,
                    out=final_colors[offs[-1] + winner])

        # Expand to per-vertex colors using indexing (no temp allocation)
        all_vert_colors[:] = np.repeat(final_colors, n_verts_per, axis=0)
        sphere_mesh.vertex_colors = o3d.utility.Vector3dVector(all_vert_colors)
        vis.update_geometry(sphere_mesh)

        # Label colors — only update when winner changes
        if winner != prev_winner:
            prev_winner = winner
            for ni, m in enumerate(label_meshes):
                if m is None:
                    continue
                m.vertex_colors = o3d.utility.Vector3dVector(
                    col_win[ni] if ni == winner else col_dim[ni]
                )
                vis.update_geometry(m)

        # Edge colors — reuse pre-allocated array
        if n_edges > 0:
            if active:
                flow = w_abs * smooth_vals[src_i] * smooth_vals[dst_i]
                flow_n[:] = 0.0
                last_mask_idx = len(layer_masks) - 1
                for li, mask in enumerate(layer_masks):
                    if not mask.any():
                        continue
                    if li == last_mask_idx:
                        # Last layer: use only src activation × weight
                        # dst (output) brightness already shown by node color
                        f = w_abs[mask] * smooth_vals[src_i[mask]]
                    else:
                        f = flow[mask]
                    flow_n[mask] = f / (float(f.max()) + 1e-8)
                np.sqrt(flow_n, out=flow_n)
                t = flow_n[:, None]
                np.clip(base_lc * (1.0 - t * 0.80) + t * 0.80, 0.0, 1.0, out=edge_col)
                edge_col *= flow_n[:, None]
            else:
                edge_col[:] = 0.04
            ls.colors = o3d.utility.Vector3dVector(edge_col)
            vis.update_geometry(ls)

        vis.update_renderer()

    vis.destroy_window()


# ── Helpers ───────────────────────────────────────────────────────────────────
HAND_CONNS = list(mp.solutions.hands.HAND_CONNECTIONS)   # precomputed once

def put_text(img, text, pos, scale=0.6, color=(255,255,255), thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_DUPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),     cv2.FONT_HERSHEY_DUPLEX, scale, color,   thickness,   cv2.LINE_AA)


def extract_features(detected, handedness):
    def hand_feat(hl):
        # Go straight to numpy — no intermediate Python list of tuples
        lms   = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark], dtype=np.float32)
        wrist = lms[0]
        scale = np.linalg.norm(wrist[:2] - lms[12, :2]) + 1e-6
        return ((lms - wrist) / scale).ravel()

    if len(detected) == 2:
        pairs = sorted(zip(detected, handedness),
                       key=lambda x: 0 if x[1].classification[0].label == "Left" else 1)
        return np.concatenate([hand_feat(pairs[0][0]), hand_feat(pairs[1][0])])
    feat = hand_feat(detected[0])
    return np.concatenate([feat, np.zeros(63, dtype=np.float32)])


# ── Threads ───────────────────────────────────────────────────────────────────
threading.Thread(target=brain_thread, daemon=True).start()

_infer_lock   = threading.Lock()
_infer_input  = {"frame": None}
_infer_result = {"prediction": "No hand", "probs": None}


def inference_thread():
    while not _stop.is_set():
        with _infer_lock:
            flat = _infer_input["frame"]
            _infer_input["frame"] = None
        if flat is None:
            time.sleep(0.002)
            continue
        with torch.inference_mode():   # faster than no_grad
            _flat_buf.copy_(torch.from_numpy(flat).unsqueeze(0))
            out, acts  = model(_flat_buf)
            probs_t    = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx   = int(probs_t.argmax())
        prediction = le.classes_[pred_idx]
        push_brain(flat, acts, probs_t)
        with _infer_lock:
            _infer_result.update({"prediction": prediction, "probs": probs_t})


threading.Thread(target=inference_thread, daemon=True).start()

# ── Camera loop ───────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap      = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

WIN         = "Gesture Recognition"
first_frame = True
vote_buffer = deque(maxlen=6)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Downscale first (cheaper), then convert color once for MediaPipe
        small  = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        small  = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = hands.process(small)

        with _infer_lock:
            prediction = _infer_result["prediction"]
            probs      = _infer_result["probs"]

        confidence = float(probs.max()) if probs is not None else 0.0
        h, w = frame.shape[:2]

        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                for c0, c1 in HAND_CONNS:
                    p1, p2 = hl.landmark[c0], hl.landmark[c1]
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0,200,255), 2)
                for lm in hl.landmark:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (255,255,255), -1)

            flat = extract_features(results.multi_hand_landmarks, results.multi_handedness)
            with _infer_lock:
                if _infer_input["frame"] is None:
                    _infer_input["frame"] = flat
        else:
            push_brain(None, None, None)

        vote_buffer.append(prediction)
        stable = Counter(vote_buffer).most_common(1)[0][0]

        put_text(frame, stable,              (20, 60),  scale=2.0, color=(0,255,80),    thickness=2)
        put_text(frame, f"{confidence:.0%}", (20, 105), scale=1.1, color=(180,255,180), thickness=1)
        put_text(frame, "Q to quit",         (20, h-20), scale=0.6, color=(160,160,160))

        cv2.imshow(WIN, frame)
        if first_frame:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            first_frame = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    _stop.set()
    cap.release()
    cv2.destroyAllWindows()
