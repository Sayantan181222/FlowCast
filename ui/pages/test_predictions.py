"""
Test Predictions Page — NYC Taxi Trip Duration Prediction.

Allows the user to:
1. Train a Gradient Boosting model on train.csv
2. Upload test.csv and predict trip durations
3. Download Kaggle submission CSV
4. View exploratory analysis of predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_CONFIG, MODELS_DIR
from ui.styles import section_header, metric_card
from ui.components import render_metric_row, render_header

TRIP_MODEL_PATH = os.path.join(MODELS_DIR, "trip_duration_model.pkl")
TRIP_SCALER_PATH = os.path.join(MODELS_DIR, "trip_duration_scaler.pkl")


def render_test_predictions():
    """Render the test predictions page."""
    render_header("Test Predictions", "🎯",
                  "Train trip duration model on train.csv · Predict on test.csv · Download submission")

    tab1, tab2, tab3 = st.tabs(["🏋️ Train Model", "🔮 Predict & Submit", "📊 Analysis"])

    with tab1:
        _render_training_tab()

    with tab2:
        _render_prediction_tab()

    with tab3:
        _render_analysis_tab()


# ─────────────────────────────────────────────
# TAB 1: Train Model
# ─────────────────────────────────────────────

def _render_training_tab():
    """Train the GBM trip duration model."""
    st.markdown(section_header("🏋️ Train Trip Duration Model",
                               "Gradient Boosting Regressor on NYC Taxi train.csv"), unsafe_allow_html=True)

    real_data_exists = os.path.exists(DATA_CONFIG["real_train_file"])
    model_exists = os.path.exists(TRIP_MODEL_PATH)

    # Status
    col1, col2 = st.columns(2)
    with col1:
        color = "#16a34a" if real_data_exists else "#ca8a04"
        icon = "✅" if real_data_exists else "⚠️"
        msg = "train.csv found (~1.46M records)" if real_data_exists else "train.csv not found"
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: {color};">{icon} Training Data</h4>
            <p style="color: #475569; font-size: 0.85rem;">{msg}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        color2 = "#16a34a" if model_exists else "#64748b"
        icon2 = "✅" if model_exists else "⭕"
        msg2 = "Trained model found — ready to predict" if model_exists else "Model not yet trained"
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: {color2};">{icon2} Model Status</h4>
            <p style="color: #475569; font-size: 0.85rem;">{msg2}</p>
        </div>
        """, unsafe_allow_html=True)

    if not real_data_exists:
        st.error("❌ train.csv not found at `nyc-taxi-trip-duration/train.csv`")
        return

    st.markdown("<br>", unsafe_allow_html=True)

    # Hyperparameters
    st.markdown(section_header("⚙️ Model Hyperparameters"), unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.slider("Number of Trees", 50, 500, 200, 50)
    with col2:
        max_depth = st.slider("Max Tree Depth", 3, 8, 5)
    with col3:
        sample_size = st.select_slider(
            "Training Sample Size",
            options=[100000, 250000, 500000, 1000000, 1458644],
            value=500000,
            format_func=lambda x: f"{x:,}" if x < 1458644 else "All (1.46M)",
        )

    st.markdown(f"""
    <div class="glass-card">
        <h4 style="color: #4f46e5;">📐 Model Info</h4>
        <p style="color: #475569; font-size: 0.85rem;">
            <strong>Algorithm:</strong> Gradient Boosting Regressor (scikit-learn)<br>
            <strong>Target:</strong> log₁p(trip_duration_seconds) — inverse: exp(·)−1<br>
            <strong>Metric:</strong> RMSLE (Kaggle competition standard)<br>
            <strong>Features:</strong> Haversine distance, bearing, hour/day/month, rush hour, vendor, passenger count
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Train Model", use_container_width=True):
        _train_model(n_estimators, max_depth, sample_size)


def _train_model(n_estimators, max_depth, sample_size):
    """Execute model training."""
    from module2_forecasting.trip_predictor import TripDurationPredictor

    progress = st.progress(0)
    status = st.empty()

    # Load training data
    status.text(f"📦 Loading {sample_size:,} records from train.csv...")
    progress.progress(5)

    nrows = sample_size if sample_size < 1458644 else None
    df = pd.read_csv(DATA_CONFIG["real_train_file"], nrows=nrows)
    progress.progress(20)

    # Train/val split (80/20 by time)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df = df.sort_values("pickup_datetime").reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    status.text(f"🧠 Training GBM on {len(df_train):,} samples...")
    progress.progress(30)

    predictor = TripDurationPredictor(n_estimators=n_estimators, max_depth=max_depth)
    train_metrics = predictor.train(df_train, verbose=False)
    progress.progress(75)

    status.text("📊 Evaluating on validation set...")
    val_metrics = predictor.evaluate(df_val, verbose=False)
    progress.progress(90)

    # Save model
    predictor.save()
    st.session_state["trip_predictor"] = predictor
    progress.progress(100)
    status.text("✅ Training complete!")

    # Show metrics
    st.success(f"✅ Model trained and saved!")
    render_metric_row([
        ("📉", f"{val_metrics['rmsle']:.4f}", "Val RMSLE"),
        ("📏", f"{val_metrics['mae_min']:.2f} min", "Val MAE"),
        ("📊", f"{val_metrics['mape']:.1f}%", "Val MAPE"),
        ("🎯", f"{val_metrics['r2']:.4f}", "Val R²"),
        ("🌳", f"{n_estimators}", "Trees"),
    ])

    # Feature importance chart
    st.markdown("<br>", unsafe_allow_html=True)
    fi = predictor.feature_importance()
    fig = px.bar(fi, x="importance", y="feature", orientation="h",
                 title="Feature Importance",
                 color="importance", color_continuous_scale="Viridis",
                 template="plotly_dark")
    fig.update_layout(height=400, yaxis=dict(autorange="reversed"),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Prediction vs actual scatter (on validation set sample)
    st.markdown("**Predicted vs Actual Duration (Validation Sample)**")
    n_show = min(2000, len(val_metrics["predictions_sec"]))
    idx = np.random.choice(len(val_metrics["predictions_sec"]), n_show, replace=False)
    pred_min = val_metrics["predictions_sec"][idx] / 60
    true_min = val_metrics["targets_sec"][idx] / 60

    fig2 = go.Figure()
    fig2.add_trace(go.Scattergl(
        x=true_min, y=pred_min,
        mode="markers",
        marker=dict(size=3, color="#6366f1", opacity=0.4),
        name="Predictions",
    ))
    max_val = min(120, max(true_min.max(), pred_min.max()))
    fig2.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(color="#ef4444", dash="dash", width=1.5),
        name="Perfect Prediction",
    ))
    fig2.update_layout(
        title=f"Predicted vs Actual (n={n_show:,})",
        xaxis_title="Actual Duration (min)",
        yaxis_title="Predicted Duration (min)",
        template="plotly_dark", height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 2: Predict & Submit
# ─────────────────────────────────────────────

def _render_prediction_tab():
    """Predict on uploaded test.csv and generate submission."""
    st.markdown(section_header("🔮 Predict on Test Data",
                               "Upload test.csv to generate trip duration predictions"), unsafe_allow_html=True)

    model_exists = os.path.exists(TRIP_MODEL_PATH)
    if not model_exists:
        st.warning("⚠️ No trained model found. Please train the model in the **Train Model** tab first.")
        return

    # Load model if not in session
    if "trip_predictor" not in st.session_state:
        from module2_forecasting.trip_predictor import TripDurationPredictor
        try:
            st.session_state["trip_predictor"] = TripDurationPredictor.load()
            st.success("✅ Loaded saved model from disk.")
        except Exception as e:
            st.error(f"Could not load model: {e}")
            return

    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #4f46e5;">📁 Upload Options</h4>
        <p style="color: #475569; font-size: 0.85rem;">
            Upload the Kaggle <code>test.csv</code> file, or use the bundled test file if available.<br>
            Expected columns: <code>id, vendor_id, pickup_datetime, passenger_count,
            pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        uploaded = st.file_uploader("📂 Upload test.csv", type=["csv"])

    with col2:
        bundled_exists = os.path.exists(DATA_CONFIG["real_test_file"])
        if bundled_exists:
            st.markdown("<br>", unsafe_allow_html=True)
            use_bundled = st.button("📦 Use Bundled test.csv (~625K records)",
                                    use_container_width=True)
        else:
            use_bundled = False

    df_test = None

    if uploaded is not None:
        with st.spinner("Reading uploaded file..."):
            df_test = pd.read_csv(uploaded)
        st.success(f"✅ Uploaded: {len(df_test):,} records")

    elif use_bundled and bundled_exists:
        with st.spinner("Loading bundled test.csv..."):
            df_test = pd.read_csv(DATA_CONFIG["real_test_file"])
        st.success(f"✅ Loaded bundled test.csv: {len(df_test):,} records")

    if df_test is not None:
        # Validate columns
        required = ["id", "pickup_datetime", "pickup_longitude", "pickup_latitude",
                    "dropoff_longitude", "dropoff_latitude"]
        missing = [c for c in required if c not in df_test.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
            return

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Test Data Preview (first 10 rows):**")
        st.dataframe(df_test.head(10), use_container_width=True)

        # Predict button
        if st.button("🚀 Run Predictions", use_container_width=True):
            _run_predictions(df_test)


def _run_predictions(df_test):
    """Execute predictions and show/download results."""
    predictor = st.session_state["trip_predictor"]

    progress = st.progress(0)
    status = st.empty()

    status.text(f"🔮 Predicting {len(df_test):,} trips...")
    progress.progress(20)

    submission = predictor.predict_with_id(df_test)
    submission["trip_duration_min"] = (submission["trip_duration"] / 60).round(2)

    progress.progress(80)
    st.session_state["submission"] = submission
    st.session_state["test_df"] = df_test
    progress.progress(100)
    status.text("✅ Predictions complete!")

    st.success(f"✅ Predicted {len(submission):,} trips!")

    # Key stats
    dur = submission["trip_duration"]
    render_metric_row([
        ("🔢", f"{len(submission):,}", "Trips Predicted"),
        ("📏", f"{dur.mean()/60:.1f} min", "Avg Duration"),
        ("⬇️", f"{dur.min()/60:.1f} min", "Min Duration"),
        ("⬆️", f"{dur.quantile(0.95)/60:.1f} min", "95th Pct"),
        ("⬆️⬆️", f"{dur.max()/60:.1f} min", "Max Duration"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # Preview results table
    st.markdown("**Prediction Results (first 20 rows):**")
    st.dataframe(submission.head(20), use_container_width=True)

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        # Kaggle submission format (id, trip_duration)
        kaggle_csv = submission[["id", "trip_duration"]].to_csv(index=False)
        st.download_button(
            "⬇️ Download Kaggle Submission CSV",
            data=kaggle_csv,
            file_name="nyc_taxi_submission.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        # Full results with minutes
        full_csv = submission.to_csv(index=False)
        st.download_button(
            "⬇️ Download Full Results CSV",
            data=full_csv,
            file_name="nyc_taxi_predictions_full.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Duration histogram
    fig = px.histogram(submission, x="trip_duration_min", nbins=80,
                       title="Predicted Trip Duration Distribution",
                       template="plotly_dark",
                       color_discrete_sequence=["#6366f1"],
                       range_x=[0, 90])
    fig.update_layout(height=350, xaxis_title="Predicted Duration (min)",
                      yaxis_title="Number of Trips")
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 3: Analysis
# ─────────────────────────────────────────────

def _render_analysis_tab():
    """Detailed analysis of predictions."""
    st.markdown(section_header("📊 Prediction Analysis",
                               "Explore patterns in the predicted trip durations"), unsafe_allow_html=True)

    submission = st.session_state.get("submission")
    df_test = st.session_state.get("test_df")

    if submission is None or df_test is None:
        st.info("👆 Run predictions in the **Predict & Submit** tab first.")
        return

    # Merge for analysis
    df = df_test.copy()
    df["predicted_duration_min"] = submission["trip_duration_min"].values
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.day_name()

    col1, col2 = st.columns(2)

    with col1:
        # Avg predicted duration by hour
        hourly = df.groupby("hour")["predicted_duration_min"].mean().reset_index()
        fig = px.bar(hourly, x="hour", y="predicted_duration_min",
                     title="Avg Predicted Duration by Hour",
                     color="predicted_duration_min",
                     color_continuous_scale="Viridis",
                     template="plotly_dark")
        fig.update_layout(height=350, xaxis_title="Hour of Day",
                          yaxis_title="Avg Duration (min)",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # By day of week
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily = df.groupby("day_of_week")["predicted_duration_min"].mean().reset_index()
        daily["day_of_week"] = pd.Categorical(daily["day_of_week"], categories=day_order, ordered=True)
        daily = daily.sort_values("day_of_week")

        fig = px.bar(daily, x="day_of_week", y="predicted_duration_min",
                     title="Avg Predicted Duration by Day",
                     color="predicted_duration_min",
                     color_continuous_scale="Plasma",
                     template="plotly_dark")
        fig.update_layout(height=350, xaxis_title="Day",
                          yaxis_title="Avg Duration (min)",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Pickup location scatter colored by duration
    if "pickup_latitude" in df.columns:
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig = go.Figure(go.Scattergl(
            x=sample["pickup_longitude"], y=sample["pickup_latitude"],
            mode="markers",
            marker=dict(
                size=3,
                color=sample["predicted_duration_min"],
                colorscale="RdYlGn_r",
                colorbar=dict(title="Pred. Duration (min)"),
                opacity=0.6,
            ),
            text=[f"Pred: {d:.1f} min" for d in sample["predicted_duration_min"]],
            hoverinfo="text",
        ))
        fig.update_layout(
            title=f"Pickup Locations Colored by Predicted Duration (sample {len(sample):,})",
            xaxis=dict(title="Longitude", showgrid=False),
            yaxis=dict(title="Latitude", showgrid=False, scaleanchor="x"),
            template="plotly_dark", height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Passenger count vs duration
    if "passenger_count" in df.columns:
        pax = df[df["passenger_count"].between(1, 6)].groupby(
            "passenger_count")["predicted_duration_min"].mean().reset_index()
        fig = px.bar(pax, x="passenger_count", y="predicted_duration_min",
                     title="Avg Predicted Duration by Passenger Count",
                     color="predicted_duration_min",
                     color_continuous_scale="Blues",
                     template="plotly_dark")
        fig.update_layout(height=320, xaxis_title="Passenger Count",
                          yaxis_title="Avg Duration (min)",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
