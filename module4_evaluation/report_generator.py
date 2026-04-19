"""
Report Generator for Experiment Results.

Generates publication-quality visualizations and summary reports:
- Training loss curves
- Predicted vs actual demand comparisons
- Per-horizon error analysis
- Algorithm performance bar charts
- Comprehensive summary tables
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EVAL_CONFIG


class ReportGenerator:
    """
    Generate visualizations and reports from experiment results.

    Creates Plotly figures suitable for both Streamlit display and
    static export.
    """

    def __init__(self, results=None):
        """
        Initialize with experiment results.

        Args:
            results: Dict from ExperimentRunner, or load from file.
        """
        self.results = results

    def load_results(self, filepath=None):
        """Load results from JSON file."""
        filepath = filepath or os.path.join(EVAL_CONFIG["results_dir"], "experiment_results.json")
        with open(filepath, "r") as f:
            self.results = json.load(f)
        return self

    def plot_training_history(self, history):
        """
        Plot training and validation loss curves.

        Args:
            history: Dict with 'train_loss' and 'val_loss' lists.

        Returns:
            Plotly Figure.
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss Curves", "Validation Metrics"),
        )

        epochs = list(range(1, len(history["train_loss"]) + 1))

        # Loss curves
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history["train_loss"],
                mode="lines", name="Train Loss",
                line=dict(color="#4ecdc4", width=2),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history["val_loss"],
                mode="lines", name="Val Loss",
                line=dict(color="#ff6b6b", width=2),
            ),
            row=1, col=1,
        )

        # Validation metrics
        if "val_mae" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=history["val_mae"],
                    mode="lines", name="Val MAE",
                    line=dict(color="#ffe66d", width=2),
                ),
                row=1, col=2,
            )
        if "val_rmse" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=history["val_rmse"],
                    mode="lines", name="Val RMSE",
                    line=dict(color="#a8e6cf", width=2),
                ),
                row=1, col=2,
            )

        fig.update_layout(
            title="Model Training History",
            template="plotly_dark",
            height=400,
            showlegend=True,
        )
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Metric Value", row=1, col=2)

        return fig

    def plot_predictions_vs_actual(self, predictions, targets, n_points=200):
        """
        Plot predicted vs actual demand values.

        Args:
            predictions: Predicted values (flattened).
            targets: Actual values (flattened).
            n_points: Number of points to display.

        Returns:
            Plotly Figure.
        """
        n = min(n_points, len(predictions))
        idx = np.arange(n)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Predicted vs Actual Demand", "Prediction Error"),
            row_heights=[0.65, 0.35],
            vertical_spacing=0.12,
        )

        # Time series comparison
        fig.add_trace(
            go.Scatter(
                x=idx, y=targets[:n],
                mode="lines", name="Actual",
                line=dict(color="#4ecdc4", width=2),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=idx, y=predictions[:n],
                mode="lines", name="Predicted",
                line=dict(color="#ff6b6b", width=2, dash="dot"),
            ),
            row=1, col=1,
        )

        # Error plot
        errors = predictions[:n] - targets[:n]
        fig.add_trace(
            go.Bar(
                x=idx, y=errors,
                name="Error",
                marker_color=["#ff6b6b" if e > 0 else "#4ecdc4" for e in errors],
                opacity=0.7,
            ),
            row=2, col=1,
        )

        fig.update_layout(
            title="Demand Prediction Analysis",
            template="plotly_dark",
            height=600,
            showlegend=True,
        )
        fig.update_xaxes(title_text="Time Step", row=2, col=1)
        fig.update_yaxes(title_text="Demand", row=1, col=1)
        fig.update_yaxes(title_text="Error", row=2, col=1)

        return fig

    def plot_horizon_metrics(self, horizon_metrics):
        """
        Plot per-horizon evaluation metrics.

        Shows how prediction accuracy degrades with forecast horizon.

        Args:
            horizon_metrics: List of dicts with per-step metrics.

        Returns:
            Plotly Figure.
        """
        if not horizon_metrics:
            return go.Figure()

        steps = [m["horizon_step"] for m in horizon_metrics]
        mae_vals = [m["MAE"] for m in horizon_metrics]
        rmse_vals = [m["RMSE"] for m in horizon_metrics]
        r2_vals = [m["R²"] for m in horizon_metrics]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Error by Horizon Step", "R² by Horizon Step"),
        )

        fig.add_trace(
            go.Bar(x=steps, y=mae_vals, name="MAE", marker_color="#4ecdc4"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=steps, y=rmse_vals, name="RMSE", marker_color="#ff6b6b"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=steps, y=r2_vals, name="R²", marker_color="#ffe66d"),
            row=1, col=2,
        )

        fig.update_layout(
            title="Forecast Accuracy by Prediction Horizon",
            template="plotly_dark",
            height=400,
            barmode="group",
        )
        fig.update_xaxes(title_text="Horizon Step (hours ahead)")
        fig.update_yaxes(title_text="Error", row=1, col=1)
        fig.update_yaxes(title_text="R² Score", row=1, col=2)

        return fig

    def plot_routing_comparison(self, aggregated_metrics):
        """
        Plot aggregated routing algorithm comparison.

        Args:
            aggregated_metrics: Dict from RouteMetrics.aggregate_route_metrics.

        Returns:
            Plotly Figure.
        """
        if not aggregated_metrics:
            return go.Figure()

        algorithms = list(aggregated_metrics.keys())
        avg_dist = [aggregated_metrics[a]["Avg Distance (km)"] for a in algorithms]
        avg_dur = [aggregated_metrics[a]["Avg Duration (min)"] for a in algorithms]
        avg_time = [aggregated_metrics[a]["Avg Comp. Time (ms)"] for a in algorithms]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Avg Distance (km)", "Avg Duration (min)", "Avg Computation (ms)"),
        )

        fig.add_trace(
            go.Bar(x=algorithms, y=avg_dist, marker_color="#4ecdc4",
                   text=[f"{d:.2f}" for d in avg_dist], textposition="auto"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=algorithms, y=avg_dur, marker_color="#ff6b6b",
                   text=[f"{d:.1f}" for d in avg_dur], textposition="auto"),
            row=1, col=2,
        )
        fig.add_trace(
            go.Bar(x=algorithms, y=avg_time, marker_color="#ffe66d",
                   text=[f"{d:.3f}" for d in avg_time], textposition="auto"),
            row=1, col=3,
        )

        fig.update_layout(
            title="Route Optimization — Algorithm Comparison (Aggregated)",
            template="plotly_dark",
            height=400,
            showlegend=False,
        )

        return fig

    def plot_scatter_actual_vs_predicted(self, predictions, targets):
        """
        Scatter plot of actual vs predicted values with ideal line.

        Args:
            predictions: Predicted values.
            targets: Actual values.

        Returns:
            Plotly Figure.
        """
        fig = go.Figure()

        fig.add_trace(go.Scattergl(
            x=targets, y=predictions,
            mode="markers",
            marker=dict(size=3, color="#4ecdc4", opacity=0.5),
            name="Predictions",
        ))

        # Ideal line
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines",
            line=dict(color="#ff6b6b", dash="dash", width=2),
            name="Ideal (y=x)",
        ))

        fig.update_layout(
            title="Actual vs Predicted Demand",
            xaxis_title="Actual Demand",
            yaxis_title="Predicted Demand",
            template="plotly_dark",
            height=500,
        )

        return fig
