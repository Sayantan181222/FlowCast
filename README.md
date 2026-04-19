# 🌊 FlowCast — AI Traffic & Demand Intelligence

> A modular, AI-powered framework for NYC Taxi demand forecasting, trip duration prediction, and route optimization — built on real Kaggle competition data with LightGBM and Transformer models compared side-by-side.

---

## 📂 Dataset

This project uses the **[NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)** Kaggle competition dataset.

Place the files in the `nyc-taxi-trip-duration/` folder:

```
AIDSTL/
└── nyc-taxi-trip-duration/
    ├── train.csv          # 1,458,644 trips — Jan to Jun 2016 (with trip_duration)
    └── test.csv           # 625,134 trips — same period (predict trip_duration)
```

### Data Schema

**train.csv**
| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique trip ID |
| `vendor_id` | int | Taxi vendor (1 or 2) |
| `pickup_datetime` | datetime | Trip start time |
| `dropoff_datetime` | datetime | Trip end time |
| `passenger_count` | int | Number of passengers |
| `pickup_longitude` | float | Pickup GPS longitude |
| `pickup_latitude` | float | Pickup GPS latitude |
| `dropoff_longitude` | float | Dropoff GPS longitude |
| `dropoff_latitude` | float | Dropoff GPS latitude |
| `store_and_fwd_flag` | string | Trip stored in memory before send |
| `trip_duration` | int | **Target**: trip duration in **seconds** |

**test.csv** — Same columns except `trip_duration` and `dropoff_datetime` (these are what we predict).

---

## 🏗️ System Architecture

```
nyc-taxi-trip-duration/train.csv  (1.46M trips)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Module 1 — Data Pipeline                                                │
│  • Parse datetimes · Haversine distance · Assign nearest of 67 NYC zones │
│  • Clean outliers · Engineer 12 temporal features (sin/cos encodings)    │
│  • Aggregate to 4,358 hourly demand steps · Z-score normalize            │
└──────────────┬───────────────────────────────────────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌─────────────────────────────────────┐
│ Module 2a   │  │ Module 2b — Trip Duration Predictor  │
│ Demand      │  │ GBM (200 trees, 13 features)         │
│ Forecasting │  │ log1p(trip_duration) target          │
│             │  │ → Kaggle submission.csv              │
│ ┌─────────┐ │  └─────────────────────────────────────┘
│ │LightGBM │ │
│ │+Lag 168h│ │  test.csv → predict → download CSV
│ │39 feat  │ │
│ │6 models │ │
│ └─────────┘ │
│ ┌─────────┐ │
│ │Transform│ │
│ │Encoder  │ │
│ │217K par │ │
│ └─────────┘ │
│  Compare ↕  │
└──────┬──────┘
       │
       ▼
┌────────────────────┐   ┌────────────────────────────────┐
│ Module 3 — Routing │   │ Module 4 — Evaluation          │
│ 67-node NYC graph  │   │ Metrics: MAE, RMSE, MAPE, R²   │
│ Dijkstra / A*      │   │ LGBM vs Transformer comparison │
│ Bellman-Ford       │   │ Per-horizon step analysis      │
│ TSP 2-opt          │   │ Feature importance charts      │
└────────────────────┘   └────────────────────────────────┘
```

---

## 📦 Modules

### Module 1 — Data Pipeline (`module1_data/`)

Loads and preprocesses real NYC Taxi data for all downstream tasks.

**Key steps:**
1. **Schema normalization** — parse datetimes, compute haversine distance (km → miles), assign each trip to nearest of 67 NYC zones (vectorized NumPy, ~5s for 1.46M rows)
2. **Data cleaning** — remove outliers (duration 30s–2h, distance 0.1–100mi, NYC coordinate bounding box)
3. **Feature engineering** — 12 temporal features: hour, day_of_week, month, is_rush_hour, is_weekend + 6 cyclical sin/cos encodings
4. **Demand aggregation** — aggregate trips to city-wide hourly time series (4,358 hourly steps)
5. **Normalization** — Z-score scaling per feature
6. **Temporal split** — 70% train / 15% val / 15% test (time-ordered)

**Performance:** ~40 seconds for all 1.46M records on CPU.

---

### Module 2a — Demand Forecasting (Two Models Compared)

#### ⚡ LightGBM + Lag Features (`module2_forecasting/lgbm_forecaster.py`)

Converts the hourly demand series into a supervised learning problem with 39 hand-crafted lag features.

**Features per sample:**
- `lag_1h` to `lag_24h` — last 24 hours
- `lag_48h`, `lag_72h`, `lag_168h` — 2 days, 3 days, **1 week ago (same hour)**
- `rolling_mean_3h`, `rolling_mean_6h`, `rolling_mean_24h` — short-term trend
- `rolling_std_24h` — daily demand variance
- `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` — cyclical encodings
- `is_rush_hour`, `is_weekend`

**Strategy:** Direct multi-step — one independent LightGBM model trained per horizon step (no error accumulation). Early stopping on 50 rounds.

| Metric | Value |
|--------|-------|
| Val MAE | 0.1906 |
| Val RMSE | 0.2604 |
| Val R² | **0.916** |
| Training Time | **6 seconds** |

---

#### 🧠 Transformer Encoder (`module2_forecasting/transformer_model.py`)

Custom Pre-LayerNorm Transformer trained on the normalized demand sequence.

**Architecture:**
```
Input [24 timesteps × 11 features]
  → Linear Projection (11 → 64, LayerNorm, GELU)
  → Sinusoidal Positional Encoding
  → 4 × Transformer Encoder Layers
      Pre-LN MultiHead Self-Attention (4 heads)
      Residual + Pre-LN Feed-Forward (64 → 256 → 64)
  → Global Average Pooling
  → MLP Head (64 → 128 → 64 → 6)
Output [6 hour-ahead demand forecast]
```

| Metric | Value |
|--------|-------|
| Test MAE | 0.2877 |
| Test RMSE | 0.3826 |
| Parameters | 217,926 |
| Training Time | ~35 seconds (30 epochs) |

**Comparison:**

| Model | Test MAE | Test RMSE | R² | Time |
|-------|----------|-----------|-----|------|
| ⚡ **LightGBM** | **0.1906** | **0.2604** | **0.916** | **6s** |
| 🧠 Transformer | 0.2877 | 0.3826 | ~0.75 | ~35s |

> **LightGBM wins by 34% on MAE.** The `lag_1h`, `lag_168h` (same-hour last week), and `rolling_std_24h` are the most predictive features. For structured time series with known seasonality, gradient boosting consistently outperforms vanilla Transformers.

---

### Module 2b — Trip Duration Predictor (`module2_forecasting/trip_predictor.py`)

A **Gradient Boosting Regressor** trained on per-trip features from `train.csv`, predicting `trip_duration` for each `test.csv` trip in Kaggle submission format.

**Features (13 total):**
| Feature | Description |
|---------|-------------|
| `distance_km` | Haversine distance pickup → dropoff |
| `bearing` | Direction angle (0–360°) |
| `pickup_hour` | Hour of day (0–23) |
| `pickup_dow` | Day of week (0=Mon) |
| `pickup_month` | Month (1–6) |
| `is_rush_hour` | 1 if 7–9am or 5–7pm |
| `is_weekend` | 1 if Sat/Sun |
| `passenger_count` | Number of passengers |
| `vendor_id` | Taxi vendor (1 or 2) |
| `center_lat/lon` | Midpoint coordinates |
| `delta_lat/lon` | Coordinate differences |

**Target:** `log1p(trip_duration_seconds)` — inverse: `expm1`
**Kaggle Metric:** RMSLE (Root Mean Squared Logarithmic Error)

**Output:** `submission.csv` with columns `[id, trip_duration]` ready for Kaggle upload.

---

### Module 3 — Route Optimization (`module3_routing/`)

Builds a transportation network graph from real trip data.

- **Graph:** G = (67 NYC zone nodes, ~1,528 edges)
- **Edge weights:** Haversine distance (km) + estimated duration (min)
- **Algorithms:** Dijkstra, A\* (haversine heuristic), Bellman-Ford, Multi-stop TSP (Nearest Neighbor + 2-opt)

---

### Module 4 — Evaluation (`module4_evaluation/`)

Automated experiment pipeline:
- **Forecasting:** MAE, RMSE, MAPE, R² per model per horizon step
- **Routing:** Path efficiency, algorithm computation time comparison
- **Reports:** Plotly-generated charts saved to `reports/`

---

## 🖥️ Streamlit Dashboard (6 Pages)

| Page | Description |
|------|-------------|
| 🏠 **Dashboard** | System status, dataset KPIs, architecture diagram, model results table |
| 📊 **Data Explorer** | Load & preprocess train.csv, temporal/spatial/distribution visualizations |
| 🧠 **Demand Forecasting** | Train LightGBM (⚡ Tab 1) + Transformer (🧠 Tab 2) + Side-by-Side Comparison (📊 Tab 3) |
| 🎯 **Test Predictions** | Train GBM trip predictor → Upload test.csv → Predict → Download Kaggle submission |
| 🗺️ **Route Optimizer** | Interactive pathfinding on 67-zone NYC network |
| 📈 **Experiments** | Automated evaluation suite, algorithm comparison |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd AIDSTL
pip3 install -r requirements.txt
```

### 2. Add Dataset

```
AIDSTL/nyc-taxi-trip-duration/train.csv   # ~200 MB
AIDSTL/nyc-taxi-trip-duration/test.csv    # ~70 MB
```

### 3. Launch Dashboard

```bash
streamlit run main.py
```

### 4. Recommended Workflow

```
1. Data Explorer      → Load & preprocess train.csv (1.46M records, ~40s)
2. Demand Forecasting → ⚡ Train LightGBM (6s) → 🧠 Train Transformer (35s) → 📊 Compare
3. Test Predictions   → Train GBM → Upload test.csv → Download submission.csv
4. Route Optimizer    → Explore NYC transportation network (67 zones)
5. Experiments        → Run automated evaluation suite
```

---

## 📁 Project Structure

```
AIDSTL/
├── main.py                              # Streamlit entry point (6 pages)
├── config.py                            # Paths, hyperparameters, NYC zone map
├── requirements.txt
├── README.md
│
├── nyc-taxi-trip-duration/              # ← PUT KAGGLE DATA HERE
│   ├── train.csv                        # 1.46M trips (with trip_duration)
│   └── test.csv                         # 625K trips (predict trip_duration)
│
├── module1_data/
│   ├── data_preprocessor.py             # Full preprocessing pipeline
│   └── data_loader.py                   # PyTorch Dataset + DataLoader
│
├── module2_forecasting/
│   ├── lgbm_forecaster.py               # ⚡ LightGBM with lag features (NEW)
│   ├── transformer_model.py             # 🧠 Custom Transformer encoder
│   ├── trainer.py                       # Transformer training loop
│   ├── trip_predictor.py                # GBM trip duration predictor
│   └── positional_encoding.py           # Sinusoidal positional encoding
│
├── module3_routing/
│   ├── graph_builder.py                 # NetworkX graph from trip data
│   ├── route_optimizer.py               # Dijkstra, A*, Bellman-Ford, TSP
│   └── network_visualizer.py            # Plotly network maps
│
├── module4_evaluation/
│   ├── metrics.py                       # MAE, RMSE, MAPE, R², RMSLE
│   ├── experiment_runner.py             # Automated evaluation pipeline
│   └── report_generator.py             # Plotly report generation
│
├── ui/
│   ├── styles.py                        # Light-themed CSS, glassmorphism cards
│   ├── components.py                    # Metric rows, headers
│   └── pages/
│       ├── dashboard.py                 # System overview + results table
│       ├── data_explorer.py             # Real data loading & visualization
│       ├── demand_forecast.py           # LGBM + Transformer + Comparison tabs
│       ├── test_predictions.py          # Trip predictor + test.csv upload
│       ├── route_optimizer.py           # Routing UI
│       └── experiments.py              # Experiment runner UI
│
├── data/                                # Processed data (auto-generated)
├── models/                              # Saved model checkpoints
└── reports/                             # Experiment results
```

---

## ⚙️ Requirements

```
torch>=2.0
lightgbm>=4.0          # New — LightGBM demand forecaster
networkx>=3.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
plotly>=5.15
streamlit>=1.28
scipy>=1.10
```

---

## 📊 Key Results Summary

### Demand Forecasting (6-step ahead, normalized units)

| Model | Strategy | Test MAE | Test RMSE | R² | Train Time |
|-------|----------|----------|-----------|-----|------------|
| ⚡ **LightGBM** | Direct multi-step | **0.1906** | **0.2604** | **0.916** | **6s** |
| 🧠 Transformer | Seq-to-seq | 0.2877 | 0.3826 | ~0.75 | ~35s |

### Trip Duration Prediction (val set, 500K sample)

| Model | Val RMSLE | Val MAE | Kaggle Sub |
|-------|-----------|---------|-----------|
| ⚡ GBM (200 trees) | ~0.38 | ~4.5 min | ✅ Ready |

### Route Optimization

| Metric | Value |
|--------|-------|
| NYC Zones (nodes) | 67 |
| Road Connections (edges) | ~1,528 |
| Algorithms | Dijkstra, A\*, Bellman-Ford, TSP 2-opt |

---

## 🏆 Kaggle Competition Context

Built around the **NYC Taxi Trip Duration** competition (RMSLE evaluation). The system extends beyond competition requirements with:
- City-wide **demand forecasting** (LightGBM + Transformer comparison)
- **Graph-based route optimization** (67-zone NYC network)
- **Interactive 6-page Streamlit dashboard** with real-time training
