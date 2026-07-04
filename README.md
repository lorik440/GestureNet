# GestureNet

> Lightweight gestures detection and visualization toolkit (PyTorch).

## Overview

GestureNet provides scripts for collecting gesture data, training a PyTorch model, and running real-time detection and visualization. This repository has been reorganized to separate source modules, runnable scripts, datasets, and model artifacts.

## Requirements

- Create and activate a Python virtual environment.
- Install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Layout after reorganization

- `src/` — library modules used by scripts (brain geometry, visualization helpers).
- `scripts/` — runnable scripts: `collect_data.py`, `training_model.py`, `detecting_realtime_viz.py`, `test_model.py`, `check_dataset.py`, `delete_gesture.py`.
- `data/` — dataset files such as `data.csv` (moved or created here).
- `models/` — model artifacts: `model.pth`, `label_encoder.pkl`, `model_config.pkl`.
- `PACKAGES.md`, `PROJECT_REPORT.md` — documentation and notes.

## Quick Start

1. Collect data (example):

```bash
python -m scripts.collect_data
```

2. Train model:

```bash
python -m scripts.training_model
```

3. Run real-time detection + viz:

```bash
python -m scripts.detecting_realtime_viz
```

4. Test trained model:

```bash
python -m scripts.test_model
```

## Notes

- Adjust script parameters inside the files or via command-line options if available.
- Keep `requirements.txt` up to date when adding new packages.
- The repo comes with a already trained model with some     gestures like; opened, closed, etc,

## Contributing

Feel free to open issues or submit pull requests. For questions, run the detection script and paste relevant logs.


 
