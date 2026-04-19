"""
Experiments Page — Module 4 UI.

Run automated evaluation experiments and view comprehensive reports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_CONFIG, FORECAST_CONFIG, EVAL_CONFIG
from ui.styles import section_header
from ui.components import render_metric_row, render_header


def render_experiments():
    """Render the experiments and evaluation page."""
    render_header("Experiments & Evaluation", "📈", "Module 4 — Run experiments and analyze model performance")

    tab1, tab2, tab3 = st.tabs(["🧪 Run Experiments", "📊 Forecast Results", "🗺️ Routing Results"])

    with tab1:
        _render_experiment_runner()

    with tab2:
        _render_forecast_results()

    with tab3:
        _render_routing_results()


def _render_experiment_runner():
    """Experiment execution interface."""
    st.markdown(section_header("🧪 Experiment Suite", "Run comprehensive evaluation experiments"), unsafe_allow_html=True)

    # Prerequisites check
    data_ready = os.path.exists(DATA_CONFIG["raw_data_file"])
    model_ready = os.path.exists(FORECAST_CONFIG["model_checkpoint"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: {'#16a34a' if data_ready else '#ca8a04'};">
                {'✅' if data_ready else '⚠️'} Data Pipeline
            </h4>
            <p style="color: #475569; font-size: 0.85rem;">
                {'Data is ready for experiments' if data_ready else 'Generate data in Data Explorer first'}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: {'#16a34a' if model_ready else '#ca8a04'};">
                {'✅' if model_ready else '⚠️'} Trained Model
            </h4>
            <p style="color: #475569; font-size: 0.85rem;">
                {'Model checkpoint found' if model_ready else 'Train model in Demand Forecasting first'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Experiment config
    col1, col2 = st.columns(2)
    with col1:
        run_forecast = st.checkbox("Run Forecasting Evaluation", value=model_ready, disabled=not model_ready)
    with col2:
        num_route_pairs = st.slider("Route Test Pairs", 5, 50, 20)
        run_routing = st.checkbox("Run Route Optimization Experiments", value=data_ready, disabled=not data_ready)

    if st.button("🚀 Run Full Experiment Suite", use_container_width=True):
        from module4_evaluation.experiment_runner import ExperimentRunner
        runner = ExperimentRunner()

        progress = st.progress(0)
        status = st.empty()

        try:
            # Forecasting evaluation
            if run_forecast and model_ready:
                status.text("🧠 Evaluating forecasting model on test set...")
                progress.progress(10)

                import torch
                from module1_data.data_preprocessor import DataPreprocessor
                from module1_data.data_loader import create_data_loaders
                from module2_forecasting.transformer_model import DemandTransformer
                from module2_forecasting.trainer import TransformerTrainer

                # Load model
                checkpoint = torch.load(FORECAST_CONFIG["model_checkpoint"],
                                        map_location="cpu", weights_only=False)
                config = checkpoint["model_config"]

                model = DemandTransformer(
                    num_features=config["num_features"],
                    d_model=config["d_model"],
                    n_heads=config["n_heads"],
                    n_encoder_layers=config["n_encoder_layers"],
                    dim_feedforward=config["dim_feedforward"],
                    dropout=config["dropout"],
                    forecast_horizon=config["forecast_horizon"],
                )
                model.load_state_dict(checkpoint["model_state_dict"])

                trainer = TransformerTrainer(model)
                trainer.best_val_loss = checkpoint["best_val_loss"]

                # Load history
                if os.path.exists(FORECAST_CONFIG["training_history"]):
                    with open(FORECAST_CONFIG["training_history"]) as f:
                        trainer.history = json.load(f)

                # Prepare test data
                preprocessor = DataPreprocessor()
                df = pd.read_csv(DATA_CONFIG["raw_data_file"])
                results = preprocessor.run_pipeline(df, save=False, verbose=False)

                loaders = create_data_loaders(
                    results["train"], results["val"], results["test"],
                    results["feature_columns"],
                    lookback_window=FORECAST_CONFIG["lookback_window"],
                    forecast_horizon=config["forecast_horizon"],
                    verbose=False,
                )

                progress.progress(30)
                forecast_eval = runner.run_forecast_evaluation(trainer, loaders["test"], verbose=False)
                st.session_state["forecast_eval"] = forecast_eval
                progress.progress(50)

            # Route optimization experiments
            if run_routing and data_ready:
                status.text(f"🗺️ Running {num_route_pairs} route optimization experiments...")

                from module3_routing.graph_builder import TransportationGraphBuilder
                from module3_routing.route_optimizer import RouteOptimizer

                builder = TransportationGraphBuilder()
                trip_df = pd.read_csv(DATA_CONFIG["raw_data_file"])
                graph = builder.build_from_trip_data(trip_df, verbose=False)
                graph = builder.get_largest_component_subgraph()
                opt = RouteOptimizer(graph)

                progress.progress(60)
                route_eval = runner.run_routing_experiments(
                    opt, graph, num_pairs=num_route_pairs, verbose=False
                )
                st.session_state["route_eval"] = route_eval
                progress.progress(90)

            # Save results
            runner.save_results()
            st.session_state["experiment_results"] = runner.results
            progress.progress(100)
            status.text("✅ All experiments complete!")
            st.success("✅ Experiment suite finished! Check the results tabs.")

        except Exception as e:
            st.error(f"❌ Experiment failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def _render_forecast_results():
    """Display forecasting evaluation results."""
    st.markdown(section_header("🧠 Forecasting Evaluation Results"), unsafe_allow_html=True)

    # Try to load from session or file
    forecast_eval = st.session_state.get("forecast_eval")
    results_path = os.path.join(EVAL_CONFIG["results_dir"], "experiment_results.json")

    if forecast_eval is None and os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        if all_results.get("forecasting"):
            forecast_eval = all_results["forecasting"][-1]

    if forecast_eval is None:
        st.info("👆 Run the experiment suite first to generate forecasting evaluation results.")
        return

    # Overall metrics
    metrics = forecast_eval.get("overall_metrics", {})
    render_metric_row([
        ("📏", f"{metrics.get('MAE', 0):.4f}", "MAE"),
        ("📐", f"{metrics.get('RMSE', 0):.4f}", "RMSE"),
        ("📊", f"{metrics.get('MAPE', 0):.2f}%", "MAPE"),
        ("🎯", f"{metrics.get('R²', 0):.4f}", "R² Score"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    from module4_evaluation.report_generator import ReportGenerator
    reporter = ReportGenerator()

    # Predictions vs Actual
    if "predictions_sample" in forecast_eval and "targets_sample" in forecast_eval:
        predictions = np.array(forecast_eval["predictions_sample"])
        targets = np.array(forecast_eval["targets_sample"])

        fig = reporter.plot_predictions_vs_actual(
            predictions.flatten(), targets.flatten(), n_points=100
        )
        st.plotly_chart(fig, use_container_width=True)

    # Per-horizon metrics
    horizon_metrics = forecast_eval.get("horizon_metrics", [])
    if horizon_metrics:
        fig = reporter.plot_horizon_metrics(horizon_metrics)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Detailed Per-Horizon Metrics:**")
        st.dataframe(pd.DataFrame(horizon_metrics), use_container_width=True)

    # Training info
    if "training_epochs" in forecast_eval:
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: #4f46e5;">Training Summary</h4>
            <p style="color: #475569;">
                Epochs: {forecast_eval.get('training_epochs', 'N/A')} |
                Best Val Loss: {forecast_eval.get('best_val_loss', 'N/A'):.6f} |
                Final Train Loss: {forecast_eval.get('final_train_loss', 'N/A'):.6f}
            </p>
        </div>
        """, unsafe_allow_html=True)


def _render_routing_results():
    """Display routing evaluation results."""
    st.markdown(section_header("🗺️ Route Optimization Results"), unsafe_allow_html=True)

    route_eval = st.session_state.get("route_eval")
    results_path = os.path.join(EVAL_CONFIG["results_dir"], "experiment_results.json")

    if route_eval is None and os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        if all_results.get("routing"):
            route_eval = all_results["routing"][-1]

    if route_eval is None:
        st.info("👆 Run the experiment suite first to generate routing evaluation results.")
        return

    # Aggregated metrics
    aggregated = route_eval.get("aggregated_metrics", {})
    if aggregated:
        from module4_evaluation.report_generator import ReportGenerator
        reporter = ReportGenerator()

        fig = reporter.plot_routing_comparison(aggregated)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.markdown("**Aggregated Algorithm Performance:**")
        agg_data = []
        for algo, data in aggregated.items():
            data_copy = data.copy()
            data_copy["Algorithm"] = algo
            agg_data.append(data_copy)
        st.dataframe(pd.DataFrame(agg_data), use_container_width=True)

    # Individual test results
    individual = route_eval.get("individual_comparisons", [])
    if individual:
        st.markdown(f"**Individual Test Results** ({len(individual)} route evaluations):")
        with st.expander("View All Results"):
            st.dataframe(pd.DataFrame(individual), use_container_width=True)

    num_tested = route_eval.get("num_pairs_tested", 0)
    st.markdown(f"""
    <div class="glass-card">
        <h4 style="color: #4f46e5;">Experiment Summary</h4>
        <p style="color: #475569;">
            Tested {num_tested} random source-destination pairs across all algorithms.
            Results show average distance, duration, and computation time.
        </p>
    </div>
    """, unsafe_allow_html=True)
