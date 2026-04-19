"""
FlowCast — AI Traffic & Demand Intelligence

Main Streamlit application entry point.
Run with: streamlit run main.py
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.styles import get_main_css
from ui.pages.dashboard import render_dashboard
from ui.pages.data_explorer import render_data_explorer
from ui.pages.demand_forecast import render_demand_forecast
from ui.pages.route_optimizer import render_route_optimizer
from ui.pages.experiments import render_experiments
from ui.pages.test_predictions import render_test_predictions

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="FlowCast — AI Traffic Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Apply Custom CSS
# ============================================================================
st.markdown(get_main_css(), unsafe_allow_html=True)

# ============================================================================
# Sidebar Navigation
# ============================================================================
with st.sidebar:
    st.markdown("# 🌊 FlowCast")
    st.caption("AI Traffic & Demand Intelligence")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📊 Data Explorer",
            "🧠 Demand Forecasting",
            "🎯 Test Predictions",
            "🗺️ Route Optimizer",
            "📈 Experiments",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("""
    <div style="padding: 0.8rem; border-radius: 10px;
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        font-size: 0.8rem; color: #475569;">
        <strong style="color: #4f46e5;">Modules</strong><br>
        1. Data Pipeline<br>
        2. Demand Forecasting<br>
        3. Trip Duration Predictor<br>
        4. Route Optimization<br>
        5. Evaluation & Reports
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #6366f1;
        font-size: 0.7rem; margin-top: 0.5rem;">
        NYC Taxi Dataset (Jan–Jun 2016)<br>
        PyTorch • NetworkX • Plotly<br>
        Streamlit Dashboard v2.0
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# Page Routing
# ============================================================================
if page == "🏠 Dashboard":
    render_dashboard()
elif page == "📊 Data Explorer":
    render_data_explorer()
elif page == "🧠 Demand Forecasting":
    render_demand_forecast()
elif page == "🎯 Test Predictions":
    render_test_predictions()
elif page == "🗺️ Route Optimizer":
    render_route_optimizer()
elif page == "📈 Experiments":
    render_experiments()
