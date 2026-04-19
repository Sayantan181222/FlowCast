"""
Performance Metrics for Forecasting and Route Optimization.

Implements standard evaluation metrics:

Forecasting Metrics:
- MAE (Mean Absolute Error): Average absolute prediction error
- RMSE (Root Mean Squared Error): Penalizes large errors more than MAE
- MAPE (Mean Absolute Percentage Error): Scale-independent error measure
- R² (Coefficient of Determination): Proportion of variance explained

Route Metrics:
- Total path distance (km)
- Total travel time (minutes)
- Path efficiency ratio (straight-line / actual distance)
- Computational performance (time and nodes explored)
"""

import numpy as np
from typing import Dict, List
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NYC_ZONES


class ForecastMetrics:
    """
    Evaluation metrics for demand forecasting models.

    All metrics compare predicted values against ground truth targets.
    """

    @staticmethod
    def mae(y_true, y_pred):
        """
        Mean Absolute Error.

        MAE = (1/n) × Σ|yᵢ - ŷᵢ|

        Interpretation: Average magnitude of errors in original units.
        """
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true, y_pred):
        """
        Root Mean Squared Error.

        RMSE = √((1/n) × Σ(yᵢ - ŷᵢ)²)

        Interpretation: Standard deviation of prediction errors.
        Penalizes large errors more heavily than MAE.
        """
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mape(y_true, y_pred, epsilon=1e-8):
        """
        Mean Absolute Percentage Error.

        MAPE = (100/n) × Σ|yᵢ - ŷᵢ|/|yᵢ|

        Interpretation: Average percentage error (scale-independent).
        Note: Undefined for zero targets; epsilon prevents division by zero.
        """
        y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

    @staticmethod
    def r_squared(y_true, y_pred):
        """
        Coefficient of Determination (R²).

        R² = 1 - SS_res / SS_tot
        where SS_res = Σ(yᵢ - ŷᵢ)², SS_tot = Σ(yᵢ - ȳ)²

        Interpretation:
        - R² = 1.0: Perfect prediction
        - R² = 0.0: Model predicts no better than the mean
        - R² < 0.0: Model is worse than predicting the mean
        """
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def compute_all(y_true, y_pred):
        """
        Compute all forecasting metrics at once.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            Dict with all metric values.
        """
        return {
            "MAE": ForecastMetrics.mae(y_true, y_pred),
            "RMSE": ForecastMetrics.rmse(y_true, y_pred),
            "MAPE": ForecastMetrics.mape(y_true, y_pred),
            "R²": ForecastMetrics.r_squared(y_true, y_pred),
        }


class RouteMetrics:
    """
    Evaluation metrics for route optimization algorithms.

    Compares route solutions across algorithms using distance, time,
    computational cost, and efficiency ratios.
    """

    @staticmethod
    def path_efficiency(route_result, graph):
        """
        Compute path efficiency ratio (straight-line / actual distance).

        Efficiency = haversine(start, end) / actual_route_distance

        Higher values indicate more direct routes.
        Values close to 1.0 mean the route follows the straight line.

        Args:
            route_result: RouteResult object.
            graph: Transportation graph.

        Returns:
            Efficiency ratio (0 to 1).
        """
        if not route_result.success or len(route_result.path) < 2:
            return 0.0

        start, end = route_result.path[0], route_result.path[-1]
        start_data = graph.nodes[start]
        end_data = graph.nodes[end]

        # Haversine straight-line distance
        import math
        R = 6371.0
        lat1_r = math.radians(start_data.get("lat", 0))
        lat2_r = math.radians(end_data.get("lat", 0))
        dlat = math.radians(end_data.get("lat", 0) - start_data.get("lat", 0))
        dlon = math.radians(end_data.get("lon", 0) - start_data.get("lon", 0))

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        straight_line = R * c

        if route_result.total_distance_km == 0:
            return 0.0

        return min(straight_line / route_result.total_distance_km, 1.0)

    @staticmethod
    def compare_routes(results_dict, graph):
        """
        Generate a comprehensive comparison table of route results.

        Args:
            results_dict: Dict of algorithm_name → RouteResult.
            graph: Transportation graph for efficiency calculation.

        Returns:
            List of dicts with comparison data.
        """
        comparison = []
        for name, result in results_dict.items():
            if result.success:
                efficiency = RouteMetrics.path_efficiency(result, graph)
                comparison.append({
                    "Algorithm": name,
                    "Distance (km)": round(result.total_distance_km, 3),
                    "Duration (min)": round(result.total_duration_min, 2),
                    "Hops": len(result.path) - 1,
                    "Comp. Time (ms)": round(result.computation_time_ms, 3),
                    "Efficiency": round(efficiency, 4),
                    "Optimal": result.is_optimal,
                })
        return comparison

    @staticmethod
    def aggregate_route_metrics(all_results):
        """
        Aggregate metrics across multiple route experiments.

        Args:
            all_results: List of dicts from compare_routes.

        Returns:
            Dict of algorithm → aggregated metrics.
        """
        from collections import defaultdict
        agg = defaultdict(lambda: {"distances": [], "durations": [], "times": [], "efficiencies": []})

        for result in all_results:
            algo = result["Algorithm"]
            agg[algo]["distances"].append(result["Distance (km)"])
            agg[algo]["durations"].append(result["Duration (min)"])
            agg[algo]["times"].append(result["Comp. Time (ms)"])
            agg[algo]["efficiencies"].append(result["Efficiency"])

        summary = {}
        for algo, data in agg.items():
            summary[algo] = {
                "Avg Distance (km)": round(np.mean(data["distances"]), 3),
                "Avg Duration (min)": round(np.mean(data["durations"]), 2),
                "Avg Comp. Time (ms)": round(np.mean(data["times"]), 3),
                "Avg Efficiency": round(np.mean(data["efficiencies"]), 4),
                "Std Distance (km)": round(np.std(data["distances"]), 3),
                "Experiments": len(data["distances"]),
            }

        return summary
