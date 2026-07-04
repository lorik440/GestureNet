import numpy as np
import torch.nn as nn


def layer_sizes(cfg):
    return [int(cfg["input_dim"]), 256, 128, 64, int(cfg["num_classes"])]


def collect_linear_weights(model):
    return [m.weight.detach().cpu().numpy() for m in model.net if isinstance(m, nn.Linear)]


def layer_offsets(layers):
    offs, s = [], 0
    for n in layers:
        offs.append(s)
        s += n
    return offs


def build_nodes(layers, z_spacing=4.0):
    nodes = []
    for li, n in enumerate(layers):
        side     = int(np.ceil(np.sqrt(n)))
        x_center = (side - 1) / 2.0
        y_center = (side - 1) / 2.0
        z        = li * z_spacing
        for i in range(n):
            nodes.append([i % side - x_center, i // side - y_center, z])
    return np.asarray(nodes, dtype=np.float64)


def build_strongest_edges_balanced(weights, layers, edges_per_layer=56):
    offs = layer_offsets(layers)
    chunks, w_chunks, sgn_chunks = [], [], []

    for li, w in enumerate(weights):
        wa   = np.abs(w).ravel()
        k    = min(int(edges_per_layer), wa.size)
        pick = np.argpartition(-wa, k-1)[:k] if k < wa.size else np.argsort(-wa)
        pick = pick[np.argsort(-wa[pick])]

        jj, ii = np.unravel_index(pick, w.shape)
        chunks.append(np.column_stack([offs[li] + ii, offs[li+1] + jj]))
        w_chunks.append(wa[pick].astype(np.float32))
        sgn_chunks.append(np.sign(w.ravel()[pick]).astype(np.float32))

    if not chunks:
        z = np.zeros((0, 2), dtype=np.int32)
        return z, np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    lines = np.vstack(chunks).astype(np.int32)
    w_abs = np.concatenate(w_chunks)
    sgn   = np.concatenate(sgn_chunks)
    phase = (w_abs / (w_abs.max() + 1e-8)).astype(np.float32)

    lc = np.zeros((len(lines), 3), dtype=np.float64)
    lc[sgn >= 0] = (0.22, 0.82, 0.98)
    lc[sgn <  0] = (0.98, 0.48, 0.42)

    return lines, lc, w_abs, phase


def pack_node_values(input_vec, activations, layers, class_probs=None, offs=None):
    if offs is None:
        offs = layer_offsets(layers)
    vals = np.zeros(sum(layers), dtype=np.float32)

    if input_vec is not None:
        v = np.asarray(input_vec, dtype=np.float32).ravel()
        n = min(len(v), layers[0])
        raw = np.abs(v[:n])
        # Per-layer normalize input
        vmax = raw.max() + 1e-8
        vals[offs[0]:offs[0]+n] = raw / vmax

    if activations is not None and len(activations) >= len(layers) - 2:
        for li, act in enumerate(activations):
            n, base = layers[li+1], offs[li+1]
            # Always treat activations as hidden layer outputs (post-ReLU)
            a = np.asarray(act).ravel()[:n]
            amax = a.max() + 1e-8
            vals[base:base+n] = (a / amax) ** 0.6

    # Output layer always comes from class_probs — separate from activations
    if class_probs is not None:
        p = np.asarray(class_probs, dtype=np.float32).ravel()
        k = min(layers[-1], len(p))
        pmax = p[:k].max() + 1e-8
        vals[offs[-1]:offs[-1]+k] = p[:k] / pmax

    return vals


def values_to_colors_rgb01(vals, layers, *, output_layer_is_probability=False,
                           prob_gamma=2.15, offs=None, out=None):
    if offs is None:
        offs = layer_offsets(layers)
    n_total = len(vals)
    if out is None:
        out = np.zeros((n_total, 3), dtype=np.float64)
    else:
        out[:] = 0.0

    if vals.max() < 1e-8:
        out[:] = 0.28
        return out

    for li, ln in enumerate(layers):
        sl   = vals[offs[li]:offs[li]+ln].astype(np.float32)
        last = li == len(layers) - 1
        dst  = out[offs[li]:offs[li]+ln]

        if last and output_layer_is_probability:
            t = np.clip(sl / (sl.max() + 1e-8), 0, 1) ** prob_gamma
        else:
            # Single percentile call returning both lo and hi
            lo, hi = np.percentile(sl, [8 if not last else 5, 96 if not last else 95])
            t = np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)
            if not last:
                t **= 0.92

        # Write directly into output slice — no temporary arrays
        np.clip(1.15 * t, 0, 1, out=dst[:, 0])
        np.clip((t - 0.15) * 1.35, 0, 1, out=dst[:, 1])
        np.clip(0.35 * (1 - t) + 0.12 * t, 0, 1, out=dst[:, 2])

    return out
