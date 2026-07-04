import pickle
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from . import brain_geometry as bg

with open("../models/model_config.pkl", "rb") as f:
    cfg = pickle.load(f)


class GestureNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 128, bias=False),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 64),                    nn.ReLU(),
            nn.Linear(64, num_classes)
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureNet(cfg["input_dim"], cfg["num_classes"]).to(device)
model.load_state_dict(torch.load("../models/model.pth", map_location=device, weights_only=False))
model.eval()

layers  = bg.layer_sizes(cfg)
weights = bg.collect_linear_weights(model)
nodes   = bg.build_nodes(layers)
lines, line_colors, _, _ = bg.build_strongest_edges_balanced(weights, layers, edges_per_layer=320)
line_colors = np.clip(line_colors * 0.55, 0.0, 1.0)

node_colors = []
for li, n in enumerate(layers):
    for _ in range(n):
        node_colors.append([0.2 + li * 0.15, 0.5, 1.0 - li * 0.12])
node_colors = np.array(node_colors)

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(nodes)
line_set.lines  = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(line_colors)

spheres = []
for i, p in enumerate(nodes):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=0.22)
    s.compute_vertex_normals()
    s.translate(p)
    s.paint_uniform_color(node_colors[i])
    spheres.append(s)

o3d.visualization.draw_geometries(spheres + [line_set])
