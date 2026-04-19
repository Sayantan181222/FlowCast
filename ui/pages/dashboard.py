"""
Dashboard Page — Overview of the FlowCast System.

Shows system status, key metrics, and quick-access navigation
to all four modules.
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_CONFIG, FORECAST_CONFIG
from ui.styles import metric_card, section_header, status_badge
from ui.components import render_metric_row


def render_dashboard():
    """Render the main dashboard page."""

    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2.5rem; font-weight: 800;
            background: linear-gradient(135deg, #4f46e5, #7c3aed, #6366f1);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🌊 FlowCast
        </h1>
        <p style="color: #64748b; font-size: 1.1rem; margin-top: -0.5rem;">
            AI Traffic &amp; Demand Intelligence — Forecasting, Prediction &amp; Route Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)

    # System Status
    st.markdown(section_header("📊 System Status", "Current state of all modules"), unsafe_allow_html=True)

    data_exists = os.path.exists(DATA_CONFIG["raw_data_file"])
    processed_exists = os.path.exists(DATA_CONFIG["processed_data_file"])
    model_exists = os.path.exists(FORECAST_CONFIG["model_checkpoint"])
    results_exist = os.path.exists(os.path.join("reports", "results", "experiment_results.json"))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status = "success" if data_exists else "warning"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📦</div>
            <div class="metric-value" style="font-size:1.5rem; background: linear-gradient(135deg, {'#16a34a' if data_exists else '#ca8a04'}, {'#16a34a' if data_exists else '#ca8a04'});
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {'Ready' if data_exists else 'Pending'}
            </div>
            <div class="metric-label">Raw Data</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⚙️</div>
            <div class="metric-value" style="font-size:1.5rem; background: linear-gradient(135deg, {'#16a34a' if processed_exists else '#ca8a04'}, {'#16a34a' if processed_exists else '#ca8a04'});
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {'Ready' if processed_exists else 'Pending'}
            </div>
            <div class="metric-label">Preprocessed</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🧠</div>
            <div class="metric-value" style="font-size:1.5rem; background: linear-gradient(135deg, {'#16a34a' if model_exists else '#ca8a04'}, {'#16a34a' if model_exists else '#ca8a04'});
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {'Trained' if model_exists else 'Not Trained'}
            </div>
            <div class="metric-label">Forecast Model</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-value" style="font-size:1.5rem; background: linear-gradient(135deg, {'#16a34a' if results_exist else '#ca8a04'}, {'#16a34a' if results_exist else '#ca8a04'});
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {'Complete' if results_exist else 'Pending'}
            </div>
            <div class="metric-label">Experiments</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset Stats (if available)
    if data_exists:
        st.markdown(section_header("📋 Dataset Overview", "Summary of NYC Taxi trip data"), unsafe_allow_html=True)
        try:
            df = pd.read_csv(DATA_CONFIG["raw_data_file"], nrows=100000)
            render_metric_row([
                ("🚕", f"{len(df):,}+", "Loaded Trips"),
                ("📍", f"{df['PULocationID'].nunique()}", "Active Zones"),
                ("📏", f"{df['trip_distance'].mean():.1f} mi", "Avg Distance"),
                ("⏱️", f"{df['trip_duration_min'].mean():.0f} min", "Avg Duration"),
                ("💰", f"${df['fare_amount'].mean():.2f}", "Avg Fare"),
            ])
        except Exception as e:
            st.warning(f"Could not load dataset: {e}")
    else:
        st.info("👆 Navigate to **Data Explorer** to load and preprocess the dataset.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Module cards
    st.markdown(section_header("🧩 Modules", "Navigate to individual components"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #4f46e5;">📊 Module 1 — Data Pipeline</h3>
            <p style="color: #475569; font-size: 0.9rem;">
                Load real NYC Taxi Trip Duration data (1.46M records), clean records,
                engineer temporal features, and prepare datasets for model training.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #4f46e5;">🗺️ Module 3 — Route Optimization</h3>
            <p style="color: #475569; font-size: 0.9rem;">
                Build transportation network graphs. Find optimal routes using
                Dijkstra, A*, and multi-stop optimization algorithms.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #4f46e5;">🧠 Module 2 — Demand Forecasting</h3>
            <p style="color: #475569; font-size: 0.9rem;">
                Train a Transformer encoder model to predict transportation demand.
                Configure hyperparameters and visualize predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #4f46e5;">📈 Module 4 — Experiments</h3>
            <p style="color: #475569; font-size: 0.9rem;">
                Run automated evaluation experiments. Compare algorithms with
                MAE, RMSE, R² metrics and generate reports.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Architecture diagram
    st.markdown(section_header("🏗️ System Architecture"), unsafe_allow_html=True)
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │            FlowCast — AI Traffic & Demand Intelligence               │
    │              NYC Taxi Trip Duration Dataset  (Jan – Jun 2016)                │
    ├──────────────┬──────────────────────────────┬────────────┬───────────────────┤
    │   Module 1   │          Module 2            │  Module 3  │     Module 4      │
    │   Data       │        Forecasting           │  Routing   │    Evaluation     │
    │   Pipeline   │                              │            │                   │
    ├──────────────┼──────────────────────────────┼────────────┼───────────────────┤
    │              │  2a. Demand Forecasting       │            │                   │
    │ train.csv    │  ┌──────────┐ ┌────────────┐ │ 67-zone    │ • MAE / RMSE      │
    │ 1.46M trips  │  │Transformer│ │  LightGBM  │ │ NYC graph  │ • MAPE / R²       │
    │              │  │ Encoder  │ │  + Lag 168h│ │            │ • RMSLE           │
    │ • Haversine  │  │ Pre-LN   │ │ 39 features│ │ Dijkstra   │                   │
    │ • Zone map   │  │ Self-Attn│ │ 6 models   │ │ A*         │ LGBM vs           │
    │ • Demand agg │  │ 217K par │ │ Early-Stop │ │ Bellman-F  │ Transformer       │
    │ • 4358 steps │  └──────────┘ └────────────┘ │ TSP 2-opt  │ side-by-side      │
    │              │                              │            │                   │
    │ test.csv     │  2b. Trip Duration Predictor │            │                   │
    │ 625K trips   │  GBM → Kaggle submission.csv │            │                   │
    └──────────────┴──────────────────────────────┴────────────┴───────────────────┘
    ```
    """)

    # Model results table
    st.markdown(section_header("📊 Model Performance Summary"), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #4f46e5;">📈 Demand Forecasting — 6-step ahead</h4>
            <table style="width:100%; font-size:0.85rem; color:#475569; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #e2e8f0;">
                    <th style="text-align:left; padding:5px 4px;">Model</th>
                    <th style="text-align:center;">Test MAE</th>
                    <th style="text-align:center;">Test RMSE</th>
                    <th style="text-align:center;">R²</th>
                </tr>
                <tr style="color:#16a34a; font-weight:600;">
                    <td style="padding:5px 4px;">⚡ LightGBM</td>
                    <td style="text-align:center;">0.1906</td>
                    <td style="text-align:center;">0.2604</td>
                    <td style="text-align:center;">0.916</td>
                </tr>
                <tr>
                    <td style="padding:5px 4px; color:#4f46e5;">🧠 Transformer</td>
                    <td style="text-align:center;">0.2877</td>
                    <td style="text-align:center;">0.3826</td>
                    <td style="text-align:center;">~0.75</td>
                </tr>
            </table>
            <p style="color:#94a3b8; font-size:0.75rem; margin-top:0.5rem;">
                ⚡ LightGBM wins by <strong>34%</strong> on MAE · trained in 6s vs 35s
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #4f46e5;">🎯 Trip Duration Predictor (test.csv)</h4>
            <table style="width:100%; font-size:0.85rem; color:#475569; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #e2e8f0;">
                    <th style="text-align:left; padding:5px 4px;">Model</th>
                    <th style="text-align:center;">Val RMSLE</th>
                    <th style="text-align:center;">Val MAE</th>
                    <th style="text-align:center;">Features</th>
                </tr>
                <tr style="color:#16a34a; font-weight:600;">
                    <td style="padding:5px 4px;">⚡ GBM (200 trees)</td>
                    <td style="text-align:center;">~0.38</td>
                    <td style="text-align:center;">~4.5 min</td>
                    <td style="text-align:center;">13</td>
                </tr>
            </table>
            <p style="color:#94a3b8; font-size:0.75rem; margin-top:0.5rem;">
                log1p target · haversine + bearing + temporal · Kaggle submission CSV
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer" style="color: #94a3b8;">
        🌊 FlowCast &mdash; AI Traffic &amp; Demand Intelligence&nbsp;|&nbsp;
        NYC Taxi Dataset (Jan&ndash;Jun 2016)<br>
        PyTorch &bull; LightGBM &bull; scikit-learn &bull; NetworkX &bull; Streamlit &bull; Plotly
    </div>
    """, unsafe_allow_html=True)
