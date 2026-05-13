# Project Packages — What They Do

This document explains the key packages used in the Gesture Recognition project.

---

## Core AI / Machine Learning

### `torch` (PyTorch)
The main deep learning framework. Used to define, train, and run the `GestureNet` neural network that classifies hand gestures. Handles tensor operations, backpropagation, and model saving/loading.

### `scikit-learn`
Used during training and testing for:
- Splitting data into train/test sets (`train_test_split`)
- Encoding gesture labels as numbers (`LabelEncoder`)
- Evaluating the model (`classification_report`, `confusion_matrix`, `StratifiedKFold`)

### `numpy`
Fundamental numerical computing library. Used everywhere for array operations — normalizing hand landmarks, computing distances, building feature vectors.

### `pandas`
Used to read and manage the `data.csv` dataset. Loads rows of hand landmark features and their gesture labels.

### `tensorflow` / `keras`
TensorFlow is installed as a dependency of **MediaPipe**. It is not directly used in this project's code, but MediaPipe requires it internally.

---

## Computer Vision

### `mediapipe`
Google's hand tracking library. Detects 21 hand landmarks (x, y, z coordinates) per hand in real time from the camera feed. This is the input source for all gesture features.

### `opencv-contrib-python` (cv2)
Handles everything camera-related:
- Capturing frames from the webcam
- Drawing hand skeleton overlays on the video
- Displaying the live camera window
- Converting between color formats (BGR ↔ RGB)

---

## 3D Visualization

### `open3d`
Used to render the live 3D brain visualization of the neural network. Draws the nodes (neurons) as spheres, edges (weights) as lines, and animates them based on the model's activations during inference.

---

## Web / Dashboard

### `dash` + `Flask` + `plotly`
- **Plotly** — interactive charting library
- **Flask** — lightweight web server (used by Dash internally)
- **Dash** — builds interactive web dashboards. Present in the project for potential data visualization of training results.

---

## Data & Utilities

### `scipy`
Scientific computing utilities. Used indirectly by scikit-learn and numpy for statistical operations.

### `matplotlib`
Plotting library. Useful for visualizing training curves or data distributions during development.

### `pickle` (built-in)
Saves and loads Python objects to disk — used to store the `LabelEncoder` (`label_encoder.pkl`) and model config (`model_config.pkl`).

### `sympy` / `mpmath`
Symbolic and arbitrary-precision mathematics. Installed as dependencies, not directly used in the project code.

---

## Networking & Auth (indirect dependencies)

### `requests`, `urllib3`, `certifi`
HTTP libraries used internally by TensorFlow/MediaPipe for downloading models or checking updates. Not called directly in project code.

### `google-auth`, `grpcio`, `protobuf`
Google's communication and serialization libraries — required by TensorFlow and MediaPipe internally.

---

## Dev / Notebook Tools

### `ipython`, `jupyter_core`, `ipywidgets`
Jupyter notebook support. Not used at runtime but installed in the environment for interactive development.

### `rich`
Pretty terminal output with colors and formatting. Used optionally for nicer console logs.

### `colorama`
Makes terminal color codes work on Windows. Used by other packages internally.

---

## Summary Table

| Package | Role | Used Directly |
|---|---|---|
| `torch` | Neural network (GestureNet) | Yes |
| `mediapipe` | Hand landmark detection | Yes |
| `opencv-contrib-python` | Camera & image processing | Yes |
| `open3d` | 3D brain visualization | Yes |
| `scikit-learn` | Training utilities & evaluation | Yes |
| `numpy` | Array/math operations | Yes |
| `pandas` | Dataset loading | Yes |
| `tensorflow` / `keras` | MediaPipe dependency | No (indirect) |
| `dash` / `plotly` / `flask` | Web dashboard | No (available) |
| `matplotlib` | Plotting | No (available) |
| `scipy` | Math utilities | No (indirect) |
