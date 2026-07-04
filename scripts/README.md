Scripts to run common tasks. Run from the repository root using the `-m` module flag so `src/` imports resolve:

```bash
python -m scripts.collect_data
python -m scripts.training_model
python -m scripts.test_model
python -m scripts.detecting_realtime_viz
```

Files created by scripts:
- `data/data.csv` — dataset (or legacy `data.csv` at repo root)
- `models/` — model artifacts (`model.pth`, `label_encoder.pkl`, `model_config.pkl`)
