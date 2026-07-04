# Hand Gesture Recognition — Project Report

## Overview

This project is a real-time hand gesture recognition system built with Python. It uses a webcam to detect hand positions, extracts mathematical features from the hand, feeds them into a neural network, and displays the predicted gesture. Alongside the detection, a live 3D visualization shows the neural network's internal state as it processes each frame — every neuron and connection lights up in real time based on how strongly it fires.

---

## How the System Works — Step by Step

### Step 1: The Camera Captures a Frame

The webcam captures a video frame at 640×480 pixels, 30 frames per second. The frame is resized to 320×240 before being sent to MediaPipe — this halves the processing time with no meaningful loss in detection quality since MediaPipe only needs to find the hand outline, not fine detail.

### Step 2: MediaPipe Detects the Hand

Google's MediaPipe library analyzes the frame and locates the hand. It identifies **21 specific points** on the hand called **landmarks**. These are not pixels — they are precise 3D coordinates of anatomical points on the hand.

**The 21 MediaPipe hand landmarks:**

| Index | Point |
|---|---|
| 0 | Wrist |
| 1 | Thumb CMC (base of thumb) |
| 2 | Thumb MCP |
| 3 | Thumb IP |
| 4 | Thumb tip |
| 5 | Index finger MCP (knuckle) |
| 6 | Index finger PIP |
| 7 | Index finger DIP |
| 8 | Index finger tip |
| 9 | Middle finger MCP |
| 10 | Middle finger PIP |
| 11 | Middle finger DIP |
| 12 | Middle finger tip |
| 13 | Ring finger MCP |
| 14 | Ring finger PIP |
| 15 | Ring finger DIP |
| 16 | Ring finger tip |
| 17 | Pinky MCP |
| 18 | Pinky PIP |
| 19 | Pinky DIP |
| 20 | Pinky tip |

**What each landmark records:**
Each landmark stores three values:
- `x` — horizontal position (0.0 = left edge of frame, 1.0 = right edge)
- `y` — vertical position (0.0 = top of frame, 1.0 = bottom)
- `z` — depth estimate (negative = closer to camera, positive = further)

So each hand produces 21 × 3 = **63 raw values**.

### Step 3: Feature Extraction — Normalization

The raw x/y/z values are position-dependent — if you move your hand left or right, all the numbers change even though the gesture is the same. To fix this, the values are normalized:

1. **Subtract the wrist position** from every landmark — this makes all coordinates relative to the wrist, so the gesture looks the same regardless of where on screen the hand is
2. **Divide by hand scale** — the scale is the distance from the wrist (landmark 0) to the middle finger MCP (landmark 12). This makes the gesture scale-invariant — the same gesture at different distances from the camera produces the same numbers

The formula for each coordinate `c` of each landmark:
```
normalized_c = (c - wrist_c) / scale
```

This produces **63 normalized values per hand**.

**For two-hand gestures:** the left hand features come first (63 values), right hand second (63 values), giving **126 total features**. For single-hand gestures, the second 63 values are filled with zeros.

### Step 4: The Neural Network Makes a Prediction

The 126 normalized values are fed into GestureNet, a feedforward neural network.

---

## The Neural Network — GestureNet

### Architecture

```
Input Layer:     126 neurons  (one per normalized landmark coordinate)
                      ↓
Hidden Layer 1:  256 neurons  + BatchNorm + ReLU + Dropout(35%)
                      ↓
Hidden Layer 2:  128 neurons  + BatchNorm + ReLU + Dropout(25%)
                      ↓
Hidden Layer 3:   64 neurons  + ReLU
                      ↓
Output Layer:      N neurons  (one per gesture class)
```

### Why These Layer Sizes?

The network follows a **funnel pattern** — each layer is smaller than the previous one. This forces the network to compress the 126 input values into increasingly abstract representations:

- **256 neurons** in the first hidden layer gives the network enough capacity to detect many different combinations of landmark positions
- **128 neurons** in the second layer combines those patterns into higher-level features (e.g. "fingers spread" or "thumb extended")
- **64 neurons** in the third layer distills everything into the most essential features for distinguishing gestures
- **N output neurons** — one per gesture. The highest-scoring output is the predicted gesture

### What Each Component Does

**Linear layer** — a matrix multiplication. Each neuron computes a weighted sum of all inputs from the previous layer. The weights are what the network learns during training.

**BatchNorm (Batch Normalization)** — normalizes the outputs of a layer so they have consistent scale. Without this, the values flowing through the network can become very large or very small, making training unstable. Particularly important here because hand landmark coordinates can vary in scale.

**ReLU (Rectified Linear Unit)** — the activation function. It sets any negative value to zero: `output = max(0, input)`. This introduces non-linearity — without it, the entire network would just be one big linear equation and couldn't learn complex patterns. ReLU also creates sparsity — roughly half the neurons output zero on any given input, which is why only some nodes light up in the visualization.

**Dropout** — during training only, randomly sets a percentage of neurons to zero each batch. This prevents the network from relying too heavily on any single neuron and forces it to learn more robust features. 35% dropout on layer 1 means 35% of the 256 neurons are randomly disabled each training step.

**Output layer** — no activation function. The raw scores (logits) are passed through **softmax** at inference time, which converts them to probabilities that sum to 1.0. The gesture with the highest probability is the prediction.

### How the Network Learns — Training

Training runs for up to 120 epochs with early stopping (stops if accuracy doesn't improve for 15 consecutive epochs).

Each epoch:
1. The training data is split into batches of 32 samples
2. For each batch: feed forward → compute loss → backpropagate → update weights
3. After all batches: evaluate accuracy on the held-out test set
4. If accuracy improved: save the model weights to `model.pth`

**Loss function:** CrossEntropyLoss — measures how wrong the prediction was. If the model predicted "peace" with 90% confidence but the true label was "fist", the loss is high. If it predicted "fist" with 90% confidence and the label was "fist", the loss is low.

**Optimizer:** Adam with learning rate 0.0008 — adjusts each weight by a small amount in the direction that reduces the loss.

---

## The 3D Neural Network Visualization

### What You See

The 3D brain window shows the actual structure of GestureNet rendered as a 3D graph:

- **Spheres (nodes)** — each sphere represents one neuron. The network has 126 + 256 + 128 + 64 + N neurons total, all visible
- **Lines (edges)** — each line represents a connection (weight) between two neurons. Only the strongest connections are shown (up to 320 per layer pair)
- **Colors** — cyan edges have positive weights (excitatory), coral/red edges have negative weights (inhibitory)

### How Nodes Light Up

When a hand is detected, the visualization shows the actual activation values flowing through the network:

1. **Input layer** — brightness proportional to the normalized landmark values. Landmarks that moved far from the wrist position glow brighter
2. **Hidden layers** — brightness proportional to the post-ReLU activation. Neurons that fired (positive output after ReLU) glow, neurons that were suppressed (zeroed by ReLU) stay dark
3. **Output layer** — brightness proportional to the softmax probability. The winning gesture's output neuron glows brightest (normalized to 1.0), others show their relative probability

Each layer is normalized independently — the brightest node in each layer always reaches full brightness. This is intentional: it shows the relative importance of neurons within each layer, not absolute values (which would make deeper layers look dim due to BatchNorm).

### How Edges Light Up

Edge brightness = `|weight| × src_activation × dst_activation`

For the last layer (64 → output), only `|weight| × src_activation` is used — because the output node brightness already represents the class probability, multiplying again would make the edges too dim.

Each layer's edges are normalized to their own maximum, so all layers show comparable brightness regardless of the absolute scale of activations.

### Smoothing

The visualization uses exponential smoothing with α=0.5:
```
smooth_value = smooth_value + 0.5 × (new_value - smooth_value)
```
This prevents the visualization from jumping instantly between states, making it easier to follow the flow of activation through the network.

### The Winning Node Pulse

The output node corresponding to the predicted gesture pulses at 3Hz using a sine wave. This makes it immediately obvious which gesture the network chose, even when multiple output nodes are partially lit.

---

## Files in the Project

| File | Purpose |
|---|---|
| `scripts/collect_data.py` | Record gesture samples from webcam to `data/data.csv` (or appends to legacy `data.csv`). |
| `scripts/training_model.py` | Train `GestureNet` on collected data and save artifacts to `models/` (or repo root). |
| `scripts/test_model.py` | Evaluate model accuracy with detailed statistics using `models/` artifacts. |
| `scripts/detecting_realtime_viz.py` | Live detection + 3D neural network visualization (uses `src/` visualization modules). |
| `src/neural_brain_viz.py` | Static 3D view of the trained network structure (library module). |
| `src/viz_backprop.py` | Animated forward + backward pass visualizer (library module). |
| `src/brain_geometry.py` | Shared math for 3D node layout and edge selection (library module). |
| `scripts/check_dataset.py` | Show sample counts per gesture in `data/data.csv` or legacy `data.csv`. |
| `scripts/delete_gesture.py` | Remove a gesture class from the dataset by name. |

### Saved Files

| File | Contents |
|---|---|
| `data/data.csv` (or `data.csv`) | All collected training samples (126 features + label per row) |
| `models/model.pth` (or `model.pth`) | Trained neural network weights |
| `models/label_encoder.pkl` (or `label_encoder.pkl`) | Maps gesture names ↔ integer class indices |
| `models/model_config.pkl` (or `model_config.pkl`) | Stores input size and number of classes |

---

## Data Collection Process

Run `collect_data.py` and enter:
1. The gesture name (e.g. "fist", "peace", "thumbs_up")
2. Whether it's one-hand or two-hand

The script collects 400 samples with a 0.12 second delay between each to ensure variety. A 2-second countdown gives you time to position your hand before collection starts. Press ESC to stop early.

**Important:** collect all gestures in similar lighting conditions. The model learns from the data you give it — if "fist" was collected in bright light and "peace" in dim light, the model may use lighting artifacts as a feature rather than the actual hand shape.

---

## Testing the Model

Run `test_model.py` for a full evaluation report including:

- **Overall accuracy** on the held-out 20% test set
- **Per-class precision, recall, F1** — shows which gestures are most reliable
- **Confusion matrix** — shows which gestures get confused with each other
- **5-fold cross-validation** — the most reliable accuracy estimate, tests on 5 different splits
- **Confidence threshold analysis** — shows what accuracy you get if you only accept predictions above 50%, 60%, 70%, 80%, 90% confidence
- **Per-class confidence** — shows which gestures the model is uncertain about

---

## Forward and Backward Pass Visualizer

Run `viz_backprop.py` to see a step-by-step animation of how the network processes a single sample:

**Forward pass (cyan):** Each layer lights up left to right as the signal propagates from input to output. You can see which neurons fire at each stage.

**Backward pass (coral):** After the forward pass, the network computes how wrong it was (the loss) and propagates gradients right to left. Neurons with large gradients are the ones the optimizer would adjust most — they're the neurons that contributed most to the error.

Press SPACE to run a new random sample, Q to quit. An info panel shows the true label, predicted label, whether it was correct, the loss value, and confidence bars for every class.

---

## Performance Notes

The system runs three threads simultaneously:
- **Main thread** — camera capture and MediaPipe hand detection (runs every frame)
- **Inference thread** — PyTorch model prediction (runs when new features are available, skips frames if busy)
- **Brain thread** — Open3D 3D visualization update (runs at 20fps)

MediaPipe processes a 320×240 downscaled frame for speed. The model uses `torch.inference_mode()` which is faster than `no_grad()`. The brain thread pre-allocates all numpy arrays to avoid per-frame memory allocation.
