"""
Demand Forecast Page — Module 2 UI.

Two forecasting models side-by-side:
  - Transformer Encoder (deep learning)
  - LightGBM with Lag Features (gradient boosting)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_CONFIG, FORECAST_CONFIG, MODELS_DIR
from ui.styles import section_header, metric_card
from ui.components import render_metric_row, render_header

LGBM_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_demand_model.pkl")
LGBM_HISTORY_PATH = os.path.join(MODELS_DIR, "lgbm_training_history.json")


def render_demand_forecast():
    """Render the demand forecasting page."""
    render_header("Demand Forecasting", "🧠",
                  "Module 2 — Train & compare Transformer vs LightGBM on hourly demand")

    if not os.path.exists(DATA_CONFIG["demand_data_file"]):
        st.warning("⚠️ No demand data found. Please go to **Data Explorer** and load the dataset first.")
        return

    # ── Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "⚡ LightGBM", "🧠 Transformer", "📊 Model Comparison"
    ])

    with tab1:
        _render_lgbm_tab()
    with tab2:
        _render_transformer_tab()
    with tab3:
        _render_comparison_tab()


# ══════════════════════════════════════════════════════════════════
# TAB 1 — LightGBM
# ══════════════════════════════════════════════════════════════════

def _render_lgbm_tab():
    st.markdown(section_header("⚡ LightGBM + Lag Features",
                               "Gradient Boosting with up to 168-hour history window"), unsafe_allow_html=True)

    lgbm_trained = os.path.exists(LGBM_MODEL_PATH)
    color = "#16a34a" if lgbm_trained else "#64748b"
    icon = "✅" if lgbm_trained else "⭕"
    st.markdown(f"""
    <div class="glass-card">
        <h4 style="color: {color};">{icon} Model Status: {'Trained — ready to use' if lgbm_trained else 'Not yet trained'}</h4>
        <p style="color: #475569; font-size: 0.85rem;">
            <strong>Strategy:</strong> Direct multi-step (one model per horizon step, no error accumulation)<br>
            <strong>Lags:</strong> 1–24h, 48h, 72h, 168h (1 week same-hour) + rolling mean/std<br>
            <strong>Features:</strong> {168 + 7 + 8} total (lags + rolling stats + temporal encodings)<br>
            <strong>Early stopping:</strong> 50 rounds on validation RMSE
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        lookback = st.select_slider("Lookback Window (hrs)", options=[12, 24, 48], value=24,
                                    key="lgbm_lookback")
    with col2:
        horizon = st.select_slider("Forecast Horizon (hrs)", options=[1, 3, 6, 12], value=6,
                                   key="lgbm_horizon")
    with col3:
        n_estimators = st.slider("Max Trees (per step)", 100, 1000, 500, 100,
                                 key="lgbm_trees")

    col4, col5 = st.columns(2)
    with col4:
        lr = st.select_slider("Learning Rate", options=[0.01, 0.03, 0.05, 0.1], value=0.05,
                              key="lgbm_lr")
    with col5:
        num_leaves = st.select_slider("Num Leaves", options=[31, 63, 127, 255], value=63,
                                      key="lgbm_leaves")

    if st.button("⚡ Train LightGBM", width="stretch"):
        _run_lgbm_training(lookback, horizon, n_estimators, lr, num_leaves)

    st.markdown("---")
    _display_lgbm_results()


def _run_lgbm_training(lookback, horizon, n_estimators, lr, num_leaves):
    from module2_forecasting.lgbm_forecaster import LGBMDemandForecaster

    progress = st.progress(0)
    status = st.empty()
    metrics_box = st.empty()

    status.text("📦 Loading demand time series...")
    progress.progress(5)

    demand_df = pd.read_csv(DATA_CONFIG["demand_data_file"])
    demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
    status.text(f"  {len(demand_df)} hourly steps loaded")
    progress.progress(10)

    forecaster = LGBMDemandForecaster(
        lookback=lookback,
        forecast_horizon=horizon,
        n_estimators=n_estimators,
        learning_rate=lr,
        num_leaves=num_leaves,
    )

    def step_cb(step, total):
        pct = 10 + int(70 * step / total)
        progress.progress(pct)
        status.text(f"⚡ Training horizon step {step}/{total}...")

    history = forecaster.train(demand_df, verbose=True, callback=step_cb)
    progress.progress(85)

    status.text("📊 Evaluating on test set...")
    test_res = forecaster.evaluate(demand_df)
    progress.progress(95)

    forecaster.save()
    st.session_state["lgbm_forecaster"] = forecaster
    st.session_state["lgbm_test_results"] = test_res
    st.session_state["lgbm_history"] = history
    progress.progress(100)
    status.text("✅ Done!")

    st.success(
        f"✅ LightGBM trained! "
        f"Test MAE: {test_res['test_mae']:.4f} | "
        f"RMSE: {test_res['test_rmse']:.4f} | "
        f"R²: {test_res['test_r2']:.4f}"
    )


def _display_lgbm_results():
    history = st.session_state.get("lgbm_history")
    test_res = st.session_state.get("lgbm_test_results")

    if history is None and os.path.exists(LGBM_HISTORY_PATH):
        with open(LGBM_HISTORY_PATH) as f:
            history = json.load(f)
        st.session_state["lgbm_history"] = history

    if history is None:
        st.info("👆 Configure and click **Train LightGBM** to start.")
        return

    st.markdown(section_header("📊 LightGBM Results"), unsafe_allow_html=True)

    render_metric_row([
        ("📉", f"{history.get('val_mae', 0):.4f}", "Val MAE"),
        ("📐", f"{history.get('val_rmse', 0):.4f}", "Val RMSE"),
        ("🎯", f"{history.get('val_r2', 0):.4f}", "Val R²"),
        ("⏱️", f"{history.get('training_time_s', 0):.1f}s", "Train Time"),
        ("🌳", f"{history.get('n_features', 0)}", "Features"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2 = st.tabs(["📈 Per-Horizon MAE", "🔍 Feature Importance"])

    with r1:
        steps = history.get("step_metrics", [])
        if steps:
            df_steps = pd.DataFrame(steps)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_steps["horizon_step"], y=df_steps["val_mae"],
                name="Val MAE", marker_color="#6366f1",
            ))
            fig.add_trace(go.Bar(
                x=df_steps["horizon_step"], y=df_steps["val_rmse"],
                name="Val RMSE", marker_color="#a78bfa",
            ))
            fig.update_layout(
                title="MAE & RMSE per Forecast Horizon Step",
                xaxis_title="Horizon Step (h ahead)",
                yaxis_title="Error (normalized)",
                template="plotly_dark", height=350, barmode="group",
            )
            st.plotly_chart(fig, width="stretch")

            # Best iterations per step
            fig2 = go.Figure(go.Bar(
                x=df_steps["horizon_step"],
                y=df_steps["best_iteration"],
                marker_color="#22c55e",
            ))
            fig2.update_layout(
                title="Best # Trees per Horizon Step (early stopping)",
                xaxis_title="Horizon Step", yaxis_title="Trees Used",
                template="plotly_dark", height=300,
            )
            st.plotly_chart(fig2, width="stretch")

    with r2:
        forecaster = st.session_state.get("lgbm_forecaster")
        if forecaster is None and os.path.exists(LGBM_MODEL_PATH):
            from module2_forecasting.lgbm_forecaster import LGBMDemandForecaster
            try:
                forecaster = LGBMDemandForecaster.load()
                st.session_state["lgbm_forecaster"] = forecaster
            except Exception:
                pass

        if forecaster:
            fi = forecaster.feature_importance()
            fig = px.bar(fi.head(20), x="importance", y="feature", orientation="h",
                         title="Top 20 Feature Importances (avg across horizon steps)",
                         color="importance", color_continuous_scale="Viridis",
                         template="plotly_dark")
            fig.update_layout(height=500, yaxis=dict(autorange="reversed"),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Train the model to see feature importances.")

    if test_res:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(section_header("🔮 Test Set Predictions"), unsafe_allow_html=True)
        preds = np.array(test_res["predictions"])
        targets = np.array(test_res["targets"])
        _plot_predictions(preds, targets, title="LightGBM")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Transformer
# ══════════════════════════════════════════════════════════════════

def _render_transformer_tab():
    st.markdown(section_header("🧠 Transformer Encoder",
                               "Custom Pre-LayerNorm Transformer for sequence-to-sequence forecasting"), unsafe_allow_html=True)

    model_exists = os.path.exists(FORECAST_CONFIG["model_checkpoint"])
    color = "#16a34a" if model_exists else "#64748b"
    icon = "✅" if model_exists else "⭕"
    st.markdown(f"""
    <div class="glass-card">
        <h4 style="color: {color};">{icon} Model Status: {'Trained — checkpoint found' if model_exists else 'Not yet trained'}</h4>
        <p style="color: #475569; font-size: 0.85rem;">
            <strong>Architecture:</strong> Encoder-only Transformer with positional encoding<br>
            <strong>Input:</strong> [lookback × 11 features] → Multi-head Self-Attention → Global Avg Pool → MLP Head<br>
            <strong>Parameters:</strong> ~217K<br>
            <strong>Training:</strong> AdamW + ReduceLROnPlateau + early stopping
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        d_model = st.select_slider("Embedding Dim (d_model)", options=[32, 64, 128, 256], value=64)
    with col2:
        n_layers = st.select_slider("Encoder Layers", options=[2, 3, 4, 6, 8], value=4)
    with col3:
        n_heads = st.select_slider("Attention Heads", options=[2, 4, 8], value=4)
    with col4:
        max_epochs = st.slider("Max Epochs", 5, 100, 30, 5)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        lr = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
    with col6:
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=64)
    with col7:
        lookback = st.select_slider("Lookback Window (hrs)", options=[6, 12, 24, 48], value=24)
    with col8:
        horizon = st.select_slider("Forecast Horizon (hrs)", options=[1, 3, 6, 12], value=6)

    st.markdown(f"""
    <div class="glass-card">
        <h4 style="color: #4f46e5; margin-bottom: 0.5rem;">📐 Architecture Summary</h4>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; font-size: 0.85rem;">
            <div><span style="color: #64748b;">Input:</span> <strong style="color: #1e293b;">[{lookback}, 11]</strong></div>
            <div><span style="color: #64748b;">d_model:</span> <strong style="color: #1e293b;">{d_model}</strong></div>
            <div><span style="color: #64748b;">Layers:</span> <strong style="color: #1e293b;">{n_layers}</strong></div>
            <div><span style="color: #64748b;">Output:</span> <strong style="color: #1e293b;">[{horizon}]</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Train Transformer", width="stretch"):
        _run_transformer_training(d_model, n_layers, n_heads, max_epochs, lr, batch_size, lookback, horizon)

    st.markdown("---")
    _display_transformer_results(lookback, horizon)


def _run_transformer_training(d_model, n_layers, n_heads, max_epochs, lr, batch_size, lookback, horizon):
    import torch
    from module1_data.data_preprocessor import DataPreprocessor
    from module1_data.data_loader import create_data_loaders
    from module2_forecasting.transformer_model import DemandTransformer
    from module2_forecasting.trainer import TransformerTrainer

    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_placeholder = st.empty()

    status_text.text("📦 Loading and preprocessing demand data...")
    progress_bar.progress(5)

    demand_df = pd.read_csv(DATA_CONFIG["demand_data_file"])
    demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])

    preprocessor = DataPreprocessor()
    feature_cols = ["demand", "hour_of_day", "day_of_week", "is_weekend",
                    "is_rush_hour", "hour_sin", "hour_cos",
                    "dow_sin", "dow_cos", "month_sin", "month_cos"]
    feature_cols = [c for c in feature_cols if c in demand_df.columns]
    demand_norm = preprocessor.normalize(demand_df.copy(), feature_cols, verbose=False)
    train_df, val_df, test_df = preprocessor.split_temporal(demand_norm, verbose=False)
    progress_bar.progress(15)

    loaders = create_data_loaders(
        train_df, val_df, test_df, feature_cols,
        lookback_window=lookback, forecast_horizon=horizon,
        batch_size=batch_size, verbose=False,
    )
    progress_bar.progress(20)

    model = DemandTransformer(
        num_features=loaders["num_features"],
        d_model=d_model, n_heads=n_heads,
        n_encoder_layers=n_layers, forecast_horizon=horizon,
    )
    trainer = TransformerTrainer(model)
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    train_losses, val_losses = [], []

    def epoch_cb(epoch, history):
        pct = 20 + int(75 * epoch / max_epochs)
        progress_bar.progress(min(pct, 94))
        status_text.text(f"🧠 Epoch {epoch}/{max_epochs} | "
                         f"Train: {history['train_loss'][-1]:.4f} | "
                         f"Val: {history['val_loss'][-1]:.4f}")
        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"],
                                 mode="lines", name="Train Loss",
                                 line=dict(color="#4ecdc4", width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
                                 mode="lines", name="Val Loss",
                                 line=dict(color="#ff6b6b", width=2)))
        fig.update_layout(template="plotly_dark", height=280,
                          title="Training Progress (Live)",
                          xaxis_title="Epoch", yaxis_title="Loss",
                          margin=dict(l=20, r=20, t=40, b=20))
        chart_placeholder.plotly_chart(fig, width="stretch")

    history = trainer.train(
        loaders["train"], loaders["val"],
        max_epochs=max_epochs, verbose=False, callback=epoch_cb,
    )
    trainer.save_checkpoint()
    progress_bar.progress(99)

    test_results = trainer.evaluate(loaders["test"], verbose=False)
    st.session_state["transformer_history"] = history
    st.session_state["transformer_test_results"] = test_results
    st.session_state["transformer_params"] = model.count_parameters()

    progress_bar.progress(100)
    status_text.text("✅ Done!")
    st.success(f"✅ Transformer trained! "
               f"Test MAE: {test_results['test_mae']:.4f} | "
               f"RMSE: {test_results['test_rmse']:.4f}")


def _display_transformer_results(lookback, horizon):
    history = st.session_state.get("transformer_history")
    history_path = FORECAST_CONFIG["training_history"]

    if history is None and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        st.session_state["transformer_history"] = history

    if history is None:
        st.info("👆 Configure and click **Train Transformer** to start.")
        return

    st.markdown(section_header("📊 Transformer Results"), unsafe_allow_html=True)

    test_results = st.session_state.get("transformer_test_results")
    if test_results:
        render_metric_row([
            ("📉", f"{test_results['test_loss']:.4f}", "Test Loss (MSE)"),
            ("📏", f"{test_results['test_mae']:.4f}", "Test MAE"),
            ("📐", f"{test_results['test_rmse']:.4f}", "Test RMSE"),
            ("🏆", f"{min(history['val_loss']):.4f}", "Best Val Loss"),
            ("🔄", f"{len(history['val_loss'])}", "Epochs Trained"),
        ])

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2 = st.tabs(["📈 Training Curves", "🔮 Predictions"])

    with r1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history["train_loss"], mode="lines",
                                 name="Train Loss", line=dict(color="#4ecdc4", width=2)))
        fig.add_trace(go.Scatter(y=history["val_loss"], mode="lines",
                                 name="Val Loss", line=dict(color="#ff6b6b", width=2)))
        fig.update_layout(template="plotly_dark", height=350,
                          title="Transformer Training Curves",
                          xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, width="stretch")

    with r2:
        if test_results and "predictions" in test_results:
            preds = np.array(test_results["predictions"])
            targets = np.array(test_results["targets"])
            _plot_predictions(preds, targets, title="Transformer")
        else:
            st.info("Train the model to see predictions.")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — Side-by-Side Comparison
# ══════════════════════════════════════════════════════════════════

def _render_comparison_tab():
    st.markdown(section_header("📊 Model Comparison",
                               "LightGBM vs Transformer — side-by-side on the same test set"), unsafe_allow_html=True)

    lgbm_res = st.session_state.get("lgbm_test_results")
    trans_res = st.session_state.get("transformer_test_results")

    # Try loading from files if not in session
    lgbm_history = st.session_state.get("lgbm_history")
    if lgbm_history is None and os.path.exists(LGBM_HISTORY_PATH):
        with open(LGBM_HISTORY_PATH) as f:
            lgbm_history = json.load(f)
        st.session_state["lgbm_history"] = lgbm_history

    trans_history = st.session_state.get("transformer_history")
    if trans_history is None and os.path.exists(FORECAST_CONFIG["training_history"]):
        with open(FORECAST_CONFIG["training_history"]) as f:
            trans_history = json.load(f)
        st.session_state["transformer_history"] = trans_history

    n_trained = sum([lgbm_res is not None, trans_res is not None])
    if n_trained == 0:
        st.info("👆 Train at least one model (LightGBM or Transformer) to see results here. "
                "Train both to compare!")
        return

    # ── Metrics Side-by-Side ─────────────────────────────────────
    st.markdown("### 📋 Metrics Comparison")

    rows = []
    if lgbm_res:
        rows.append({
            "Model": "⚡ LightGBM",
            "Test MAE": f"{lgbm_res['test_mae']:.4f}",
            "Test RMSE": f"{lgbm_res['test_rmse']:.4f}",
            "Test R²": f"{lgbm_res['test_r2']:.4f}",
            "Test MAPE": f"{lgbm_res.get('test_mape', 0):.2f}%",
            "Train Time": f"{lgbm_history.get('training_time_s', 0):.1f}s" if lgbm_history else "–",
        })
    if trans_res:
        rows.append({
            "Model": "🧠 Transformer",
            "Test MAE": f"{trans_res['test_mae']:.4f}",
            "Test RMSE": f"{trans_res['test_rmse']:.4f}",
            "Test R²": "–",
            "Test MAPE": "–",
            "Train Time": f"{len(trans_history.get('val_loss', []))} epochs" if trans_history else "–",
        })

    st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")

    # ── Bar chart comparison ──────────────────────────────────────
    if lgbm_res and trans_res:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Visual Comparison")

        metrics_names = ["Test MAE", "Test RMSE"]
        lgbm_vals = [lgbm_res["test_mae"], lgbm_res["test_rmse"]]
        trans_vals = [trans_res["test_mae"], trans_res["test_rmse"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="⚡ LightGBM", x=metrics_names, y=lgbm_vals,
                             marker_color="#22c55e", text=[f"{v:.4f}" for v in lgbm_vals],
                             textposition="outside"))
        fig.add_trace(go.Bar(name="🧠 Transformer", x=metrics_names, y=trans_vals,
                             marker_color="#6366f1", text=[f"{v:.4f}" for v in trans_vals],
                             textposition="outside"))
        fig.update_layout(
            title="LightGBM vs Transformer — Test Set Metrics",
            yaxis_title="Error (normalized units)",
            template="plotly_dark", height=380, barmode="group",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, width="stretch")

        # ── Winner badge ─────────────────────────────────────────
        lgbm_wins = sum([
            lgbm_res["test_mae"] < trans_res["test_mae"],
            lgbm_res["test_rmse"] < trans_res["test_rmse"],
        ])
        winner = "⚡ LightGBM" if lgbm_wins >= 1 else "🧠 Transformer"
        win_color = "#16a34a" if lgbm_wins >= 1 else "#4f46e5"
        mae_diff = abs(lgbm_res["test_mae"] - trans_res["test_mae"])
        pct_diff = mae_diff / max(lgbm_res["test_mae"], trans_res["test_mae"]) * 100

        st.markdown(f"""
        <div class="glass-card" style="border: 2px solid {win_color}; text-align: center;">
            <h3 style="color: {win_color}; margin: 0;">🏆 {winner} wins on MAE</h3>
            <p style="color: #475569; margin: 0.5rem 0 0 0;">
                Difference: {mae_diff:.4f} ({pct_diff:.1f}% better)
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Prediction overlay ───────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔮 Prediction Overlay (step 1 forecast)")

        lgbm_preds = np.array(lgbm_res["predictions"])[:, 0]
        lgbm_targets = np.array(lgbm_res["targets"])[:, 0]
        trans_preds = np.array(trans_res["predictions"])[::-1]    # last samples
        trans_targets = np.array(trans_res["targets"])[::-1]

        # Align lengths
        n = min(200, len(lgbm_preds), len(trans_preds))
        x = list(range(n))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=lgbm_targets[:n], mode="lines",
                                  name="Actual", line=dict(color="white", width=2)))
        fig2.add_trace(go.Scatter(x=x, y=lgbm_preds[:n], mode="lines",
                                  name="⚡ LightGBM", line=dict(color="#22c55e", width=1.5, dash="dot")))
        fig2.add_trace(go.Scatter(x=x, y=trans_preds[:n], mode="lines",
                                  name="🧠 Transformer", line=dict(color="#818cf8", width=1.5, dash="dash")))
        fig2.update_layout(
            title=f"Predictions vs Actual — Last {n} Test Steps (1h-ahead)",
            xaxis_title="Time Step", yaxis_title="Normalized Demand",
            template="plotly_dark", height=400,
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig2, width="stretch")

    elif lgbm_res or trans_res:
        st.info("💡 Train **both** models to unlock the full side-by-side comparison and prediction overlay.")
        if lgbm_res:
            st.markdown("**⚡ LightGBM** — trained ✅")
        if trans_res:
            st.markdown("**🧠 Transformer** — trained ✅")


# ══════════════════════════════════════════════════════════════════
# Shared Helpers
# ══════════════════════════════════════════════════════════════════

def _plot_predictions(preds: np.ndarray, targets: np.ndarray, title: str = ""):
    """Plot predicted vs actual for first horizon step."""
    preds_flat = np.array(preds)[:, 0] if preds.ndim == 2 else preds.flatten()
    targets_flat = np.array(targets)[:, 0] if targets.ndim == 2 else targets.flatten()
    n = min(300, len(preds_flat))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=targets_flat[:n], mode="lines", name="Actual",
                             line=dict(color="white", width=2)))
    fig.add_trace(go.Scatter(y=preds_flat[:n], mode="lines", name=f"{title} Prediction",
                             line=dict(color="#6366f1", width=1.5, dash="dot")))
    fig.update_layout(
        title=f"{title} — Predicted vs Actual (1h-ahead, last {n} steps)",
        xaxis_title="Time Step", yaxis_title="Normalized Demand",
        template="plotly_dark", height=380,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, width="stretch")
