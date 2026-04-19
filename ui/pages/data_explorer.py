"""
Data Explorer Page — Module 1 UI.

Interactive data loading, preprocessing, and visualization of real NYC Taxi data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_CONFIG, NYC_ZONES
from ui.styles import section_header, metric_card
from ui.components import render_metric_row, render_header


def render_data_explorer():
    """Render the data exploration page."""
    render_header("Data Explorer", "📊", "Module 1 — Load, preprocess, and explore NYC Taxi trip data")

    # Data Loading Section
    st.markdown(section_header("🔧 Data Loading", "Load and preprocess real NYC Taxi Trip Duration data"), unsafe_allow_html=True)

    # ── Data source: file on disk OR upload ───────────────────────────
    real_data_exists = os.path.exists(DATA_CONFIG["real_train_file"])
    uploaded_file = None

    if real_data_exists:

        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: #16a34a;">✅ Dataset Found Locally</h4>
            <p style="color: #475569; font-size: 0.85rem;">
                <strong>train.csv</strong> detected at
                <code>nyc-taxi-trip-duration/train.csv</code><br>
                ~1.46M trip records · Jan–Jun 2016
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="border: 2px dashed #6366f1;">
            <h4 style="color: #4f46e5;">📂 Upload NYC Taxi Dataset</h4>
            <p style="color: #475569; font-size: 0.85rem;">
                <code>train.csv</code> not found locally.
                Download it from
                <a href="https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data"
                   target="_blank" style="color:#6366f1;">
                   Kaggle NYC Taxi Trip Duration
                </a>
                and upload below.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload train.csv or train.csv.gz (Kaggle NYC Taxi Trip Duration)",
            type=["csv", "gz"],
            help="train.csv is ~200MB. To upload faster, compress it: gzip train.csv → train.csv.gz (~50MB). "
                 "Pandas reads both formats automatically.",
            key="train_csv_upload",
        )

        if uploaded_file is None:
            st.info("👆 Upload **train.csv** to continue. You can still use **Demand Forecasting** "
                    "and **Route Optimizer** with the pre-loaded data.")
            st.markdown("---")
            # Show demand summary even without upload
            demand_file = DATA_CONFIG.get("demand_data_file", "data/demand_aggregated.csv")
            if os.path.exists(demand_file):
                st.markdown(section_header("📊 Pre-loaded Demand Summary",
                                           "Aggregated hourly demand is available for forecasting"),
                            unsafe_allow_html=True)
                demand_df = pd.read_csv(demand_file)
                render_metric_row([
                    ("📅", f"{len(demand_df):,}", "Hourly Steps"),
                    ("📈", f"{demand_df['demand'].mean():.0f}", "Avg Hourly Demand"),
                    ("🔝", f"{demand_df['demand'].max():.0f}", "Peak Demand"),
                    ("📉", f"{demand_df['demand'].min():.0f}", "Min Demand"),
                    ("📏", f"{demand_df['demand'].std():.0f}", "Std Dev"),
                ])
            return

    # ── Record count slider  ───────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.select_slider(
            "Records to Load (use full data for training)",
            options=[50000, 100000, 250000, 500000, 1000000, 1458644],
            value=250000 if uploaded_file else 1458644,
            format_func=lambda x: f"{x:,}" if x < 1458644 else "All (~1.46M)",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_btn = st.button("📦 Load & Preprocess Data", use_container_width=True)

    if load_btn:
        from module1_data.data_preprocessor import DataPreprocessor

        progress = st.progress(0)
        status = st.empty()

        status.text("📦 Reading CSV...")
        progress.progress(5)

        try:
            if uploaded_file is not None:
                # Auto-detect gzip vs plain CSV
                nrows = sample_size if sample_size < 1458644 else None
                is_gz = uploaded_file.name.endswith(".gz")
                status.text(f"📦 Reading {'compressed' if is_gz else ''} CSV...")
                df_raw = pd.read_csv(
                    uploaded_file,
                    nrows=nrows,
                    compression="gzip" if is_gz else None,
                )
                # Save to disk so other modules can access it
                os.makedirs("nyc-taxi-trip-duration", exist_ok=True)
                df_raw.to_csv(DATA_CONFIG["real_train_file"], index=False)
                st.info(f"💾 Saved {len(df_raw):,} rows to `nyc-taxi-trip-duration/train.csv`")

            else:
                nrows = sample_size if sample_size < 1458644 else None
                df_raw = pd.read_csv(DATA_CONFIG["real_train_file"], nrows=nrows)

            progress.progress(20)
            status.text(f"⚙️ Preprocessing {len(df_raw):,} records...")

            preprocessor = DataPreprocessor()
            results = preprocessor.run_pipeline(df_raw, save=True, verbose=True)
            progress.progress(90)

            results["processed_trips"].to_csv(DATA_CONFIG["raw_data_file"], index=False)
            st.session_state["raw_data"] = results["processed_trips"]
            st.session_state["preprocessed"] = results
            if "transport_graph" in st.session_state:
                del st.session_state["transport_graph"]

            progress.progress(100)
            status.text("✅ Done!")
            st.success(f"✅ Loaded and preprocessed {len(results['processed_trips']):,} records!")

        except Exception as e:
            st.error(f"❌ Error during preprocessing: {e}")
            return

    st.markdown("---")

    # ── Display section ───────────────────────────────────────────────
    df = None
    if "raw_data" in st.session_state:
        df = st.session_state["raw_data"]
    elif os.path.exists(DATA_CONFIG["raw_data_file"]):
        df = pd.read_csv(DATA_CONFIG["raw_data_file"], nrows=200000)
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        st.session_state["raw_data"] = df


    if df is not None:
        # Overview Metrics
        st.markdown(section_header("📋 Dataset Overview"), unsafe_allow_html=True)
        render_metric_row([
            ("🚕", f"{len(df):,}", "Total Trips"),
            ("📍", f"{df['PULocationID'].nunique()}", "Pickup Zones"),
            ("📏", f"{df['trip_distance'].mean():.2f} mi", "Avg Distance"),
            ("⏱️", f"{df['trip_duration_min'].mean():.1f} min", "Avg Duration"),
            ("💰", f"${df['fare_amount'].mean():.2f}", "Avg Fare"),
        ])

        st.markdown("<br>", unsafe_allow_html=True)

        # Data Preview
        st.markdown(section_header("🔍 Data Preview", "First 20 rows of processed data"), unsafe_allow_html=True)
        display_cols = [c for c in ["pickup_datetime", "PULocationID", "DOLocationID",
                                     "trip_distance", "trip_duration_min", "passenger_count",
                                     "fare_amount", "pickup_latitude", "pickup_longitude"] if c in df.columns]
        st.dataframe(df[display_cols].head(20), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualization Tabs
        st.markdown(section_header("📈 Data Visualizations"), unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["⏰ Temporal", "🗺️ Spatial", "📊 Distributions", "🔥 Heatmap"])

        with tab1:
            _render_temporal_charts(df)
        with tab2:
            _render_spatial_charts(df)
        with tab3:
            _render_distribution_charts(df)
        with tab4:
            _render_heatmap(df)
    else:
        st.info("👆 Click **Load & Preprocess Data** to load the real NYC Taxi dataset.")


def _render_temporal_charts(df):
    """Render temporal pattern visualizations."""
    df = df.copy()
    if df["pickup_datetime"].dtype == object:
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.day_name()
    df["date"] = df["pickup_datetime"].dt.date

    col1, col2 = st.columns(2)

    with col1:
        hourly = df.groupby("hour").size().reset_index(name="trips")
        fig = px.bar(hourly, x="hour", y="trips",
                     title="Trip Volume by Hour of Day",
                     color="trips", color_continuous_scale="Viridis",
                     template="plotly_dark")
        fig.update_layout(height=350, xaxis_title="Hour", yaxis_title="Number of Trips",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily = df.groupby("day_of_week").size().reset_index(name="trips")
        daily["day_of_week"] = pd.Categorical(daily["day_of_week"], categories=day_order, ordered=True)
        daily = daily.sort_values("day_of_week")

        fig = px.bar(daily, x="day_of_week", y="trips",
                     title="Trip Volume by Day of Week",
                     color="trips", color_continuous_scale="Plasma",
                     template="plotly_dark")
        fig.update_layout(height=350, xaxis_title="Day", yaxis_title="Number of Trips",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Daily trend
    daily_trend = df.groupby("date").size().reset_index(name="trips")
    fig = px.line(daily_trend, x="date", y="trips",
                  title="Daily Trip Volume Over Time",
                  template="plotly_dark")
    fig.update_traces(line=dict(color="#6366f1", width=1.5))
    fig.update_layout(height=300, xaxis_title="Date", yaxis_title="Trips")
    st.plotly_chart(fig, use_container_width=True)


def _render_spatial_charts(df):
    """Render spatial distribution visualizations."""
    # Top pickup zones
    top_pu = df["PULocationID"].value_counts().head(15).reset_index()
    top_pu.columns = ["zone_id", "trips"]
    top_pu["zone_name"] = top_pu["zone_id"].map(
        {zid: info[0] for zid, info in NYC_ZONES.items()}
    )

    fig = px.bar(top_pu, x="trips", y="zone_name", orientation="h",
                 title="Top 15 Pickup Zones",
                 color="trips", color_continuous_scale="Viridis",
                 template="plotly_dark")
    fig.update_layout(height=450, yaxis=dict(autorange="reversed"),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter map of pickup locations (sample for performance)
    if "pickup_latitude" in df.columns:
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig = go.Figure(go.Scattergl(
            x=sample["pickup_longitude"], y=sample["pickup_latitude"],
            mode="markers",
            marker=dict(size=2, color="#6366f1", opacity=0.3),
        ))
        fig.update_layout(
            title=f"Pickup Locations (sample of {len(sample):,})",
            xaxis=dict(title="Longitude", showgrid=False),
            yaxis=dict(title="Latitude", showgrid=False, scaleanchor="x"),
            template="plotly_dark", height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Zone-based map
        zone_trips = df.groupby("PULocationID").size().reset_index(name="trips")
        lats, lons, trips, names = [], [], [], []
        for _, row in zone_trips.iterrows():
            zid = int(row["PULocationID"])
            if zid in NYC_ZONES:
                name, lat, lon = NYC_ZONES[zid]
                lats.append(lat); lons.append(lon)
                trips.append(row["trips"]); names.append(name)

        fig = go.Figure(go.Scattergl(
            x=lons, y=lats, mode="markers",
            marker=dict(
                size=[max(5, min(35, t / max(trips) * 35)) for t in trips],
                color=trips, colorscale="Hot",
                colorbar=dict(title="Trips"), opacity=0.8,
            ),
            text=[f"{n}<br>Trips: {t:,}" for n, t in zip(names, trips)],
            hoverinfo="text",
        ))
        fig.update_layout(
            title="Pickup Demand by Zone",
            xaxis=dict(title="Longitude"), yaxis=dict(title="Latitude", scaleanchor="x"),
            template="plotly_dark", height=500,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_distribution_charts(df):
    """Render feature distribution plots."""
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="trip_distance", nbins=80,
                           title="Trip Distance Distribution",
                           template="plotly_dark",
                           color_discrete_sequence=["#6366f1"],
                           range_x=[0, 20])
        fig.update_layout(height=350, xaxis_title="Distance (miles)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="trip_duration_min", nbins=80,
                           title="Trip Duration Distribution",
                           template="plotly_dark",
                           color_discrete_sequence=["#a78bfa"],
                           range_x=[0, 60])
        fig.update_layout(height=350, xaxis_title="Duration (minutes)")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(df, x="fare_amount", nbins=80,
                           title="Fare Amount Distribution",
                           template="plotly_dark",
                           color_discrete_sequence=["#22c55e"],
                           range_x=[0, 50])
        fig.update_layout(height=350, xaxis_title="Fare ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(df, x="passenger_count",
                           title="Passenger Count Distribution",
                           template="plotly_dark",
                           color_discrete_sequence=["#f59e0b"])
        fig.update_layout(height=350, xaxis_title="Passengers")
        st.plotly_chart(fig, use_container_width=True)


def _render_heatmap(df):
    """Render hour × day demand heatmap."""
    df = df.copy()
    if df["pickup_datetime"].dtype == object:
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = df.groupby(["day", "hour"]).size().reset_index(name="trips")
    pivot_table = pivot.pivot(index="day", columns="hour", values="trips").fillna(0)
    pivot_table = pivot_table.reindex(day_order)

    fig = px.imshow(pivot_table,
                    title="Demand Heatmap — Hour × Day of Week",
                    color_continuous_scale="Viridis",
                    template="plotly_dark",
                    labels=dict(x="Hour of Day", y="Day of Week", color="Trips"))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
