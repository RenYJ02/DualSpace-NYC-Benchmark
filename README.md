This repository contains the data, data loading and preprocessing code, model definitions, and training scripts for spatio-temporal hurdle models (Hurdle-Gamma and Hurdle-LogNormal) for road-level traffic accident risk forecasting on New York City's road network.

The models combine **Temporal Convolutional Networks (TCN)** with **Diffusion Graph Convolution** to jointly predict:
- **Occurrence probability**: whether an accident will happen on a road segment
- **Severity intensity**: the expected risk score given an accident occurs

## Data

Path: `data/`

| Directory / File | Description |
|---|---|
| `line_graph/` | Road network graph: edges (`edge_index.npy`), node mapping (`node_id_map.parquet`), borough masks (`mask_borough_*.npz`), borough ID mappings |
| `crashes_daily_labels.parquet` | Daily crash labels (crash count, injuries, fatalities, risk score) |
| `links_with_features.parquet` | Road segment static features (width, lanes, speed, POI, population, etc.) |
| `links_features_scaler.json` | Feature standardization statistics |
| `calendar/calendar_day.parquet` | Daily calendar features (day-of-week, holiday, etc.) |
| `weather/nyc_weather_daily.parquet` | Daily weather features per borough |
| `train_keys_all.parquet` | Road segment geometry hashes |

- **Time range**: 2021-01-01 ~ 2024-12-31
- **Spatial coverage**: 86,982 road segments across 5 NYC boroughs
- **Risk score**: `crash_count + 2 × injured + 3 × killed`

## Dependencies

Python 3.10+, PyTorch 2.0+ with CUDA (optional). Install:

```bash
pip install -r requirements.txt
```

Core packages:

| Package | Version |
|---|---|
| numpy | 2.1.0 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| torch | ≥2.0.0 |
| pyarrow | ≥10.0.0 |
| scikit-learn | ≥1.0.0 |
| joblib | ≥1.0.0 |
| matplotlib | ≥3.5.0 |
| networkx | ≥3.0.0 |
| tqdm | ≥4.0.0 |

## Quick Start

### Quick test (500 road segments, CPU, 1 epoch)

```bash
python train.py --max_nodes 500 --epochs 1 --device cpu
```

### Full training (all 87K roads, GPU recommended)

```bash
python train.py --model st_hurdle_lognormal --epochs 12 --device cuda
```

### Choose model

```bash
python train.py --model st_hurdle_gamma      # Hurdle-Gamma
python train.py --model st_hurdle_lognormal   # Hurdle-LogNormal (default)
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `st_hurdle_lognormal` | `st_hurdle_gamma` or `st_hurdle_lognormal` |
| `--epochs` | `12` | Training epochs |
| `--history_len` | `14` | Input history window (days) |
| `--horizon` | `3` | Forecast horizon (days) |
| `--hidden_dim` | `64` | Hidden dimension |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `1` | Batch size (graph models use batch=1) |
| `--max_nodes` | `0` | Subset nodes for quick testing (0 = all) |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |
| `--data_dir` | `data` | Path to data directory |
| `--out_dir` | `output` | Output directory for checkpoints and summaries |

## Output

Training produces two files in `--out_dir`:

- `{model}_best.pt` — Model checkpoint (PyTorch state dict + config)
- `{model}_summary.json` — Metrics summary (MAE, RMSE, Recall@20%, NDCG@20%, PICP, MPIW, Interval Score)

## Model Architecture

```
Input [N, T, F]  →  TCN (causal temporal conv × L_t)
                 →  DiffusionConv (Chebyshev graph conv × L_s, order K)
                 →  LayerNorm
                 →  Hurdle Head ──┬── q_logits (occurrence logits)
                                  ├── mu / mu_log (intensity mean)
                                  └── alpha / sigma_log (dispersion)
```

- **Spatial**: Diffusion graph convolution with Chebyshev polynomial approximation
- **Temporal**: Gated TCN (causal dilated convolution with gate mechanism)
- **Loss**: Binary cross-entropy (occurrence) + Gamma/LogNormal NLL (positive part)
- **Risk diffusion**: EMA temporal smoothing + spatial random walk to reduce zero-inflation
- **Priors**: Historical cumulative crash frequency + structural features (degree centrality, PageRank)

## Evaluation Metrics

| Metric | Description |
|---|---|
| MAE / RMSE | Point prediction error (all samples & positive-only) |
| Recall@20% | Recall of top-20% highest-risk roads (target-space & event-space) |
| NDCG@20% | Normalized DCG for top-20% ranking |
| PICP | Prediction Interval Coverage Probability |
| MPIW | Mean Prediction Interval Width |
| IS | Interval Score (width + penalty for misses) |

## File Structure

```
├── data/                  # NYC road network & accident dataset
├── model.py               # Model definitions (encoder, heads, loss functions)
├── utils.py               # Data loading, graph ops, metrics, risk diffusion
├── train.py               # Training / evaluation script
├── requirements.txt       # Python dependencies
└── README.md
```
