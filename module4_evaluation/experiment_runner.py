"""
Automated Experiment Runner.

Executes systematic experiments for both demand forecasting and route
optimization models. Supports:

1. Forecasting Experiments:
   - Hyperparameter sweeps (sequence length, model dimension)
   - Cross-validation style temporal evaluation
   - Ablation studies on feature subsets

2. Route Optimization Experiments:
   - Random source-destination pair testing
   - Algorithm comparison across multiple scenarios
   - Scalability analysis

Results are persisted as JSON for later analysis and reporting.
"""

import numpy as np
import pandas as pd
import json
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EVAL_CONFIG, FORECAST_CONFIG, DATA_CONFIG
from module4_evaluation.metrics import ForecastMetrics, RouteMetrics


class ExperimentRunner:
    """
    Automated experiment pipeline for model evaluation.

    Runs systematic experiments, collects results, and persists them
    for analysis and report generation.
    """

    def __init__(self):
        """Initialize experiment runner."""
        self.results = {
            "forecasting": [],
            "routing": [],
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "forecast_config": {k: str(v) for k, v in FORECAST_CONFIG.items()},
                },
            },
        }

    def run_forecast_evaluation(self, trainer, test_loader, scaler=None, verbose=True):
        """
        Evaluate the trained forecasting model on the test set.

        Computes all forecasting metrics and collects predicted vs actual values.

        Args:
            trainer: Trained TransformerTrainer instance.
            test_loader: Test DataLoader.
            scaler: Optional scaler for inverse transforms.
            verbose: Print results.

        Returns:
            Dict with evaluation results.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("FORECASTING MODEL EVALUATION")
            print(f"{'=' * 60}")

        # Get predictions
        test_results = trainer.evaluate(test_loader, verbose=False)
        predictions = test_results["predictions"]
        targets = test_results["targets"]

        # Flatten for overall metrics
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # Compute metrics
        metrics = ForecastMetrics.compute_all(target_flat, pred_flat)

        # Per-horizon metrics (if multi-step)
        horizon_metrics = []
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            for h in range(predictions.shape[1]):
                h_metrics = ForecastMetrics.compute_all(targets[:, h], predictions[:, h])
                h_metrics["horizon_step"] = h + 1
                horizon_metrics.append(h_metrics)

        result = {
            "overall_metrics": metrics,
            "horizon_metrics": horizon_metrics,
            "test_loss": test_results["test_loss"],
            "num_test_samples": len(pred_flat),
            "predictions_sample": predictions[:20].tolist(),
            "targets_sample": targets[:20].tolist(),
        }

        # Add training history if available
        if trainer.history and trainer.history.get("train_loss"):
            result["training_epochs"] = len(trainer.history["train_loss"])
            result["best_val_loss"] = trainer.best_val_loss
            result["final_train_loss"] = trainer.history["train_loss"][-1]
            result["final_val_loss"] = trainer.history["val_loss"][-1]

        self.results["forecasting"].append(result)

        if verbose:
            print(f"\n  Overall Metrics:")
            for name, value in metrics.items():
                print(f"    {name}: {value:.4f}")
            if horizon_metrics:
                print(f"\n  Per-Horizon Metrics:")
                print(f"  {'Step':>6} │ {'MAE':>10} │ {'RMSE':>10} │ {'R²':>10}")
                print(f"  {'─' * 45}")
                for hm in horizon_metrics:
                    print(f"  {hm['horizon_step']:>6} │ {hm['MAE']:>10.4f} │ "
                          f"{hm['RMSE']:>10.4f} │ {hm['R²']:>10.4f}")

        return result

    def run_routing_experiments(self, optimizer, graph, num_pairs=None,
                                seed=42, verbose=True):
        """
        Run route optimization experiments on random source-destination pairs.

        Tests all algorithms on multiple randomly selected node pairs and
        aggregates the results for comparison.

        Args:
            optimizer: RouteOptimizer instance.
            graph: Transportation graph.
            num_pairs: Number of random pairs to test.
            seed: Random seed for reproducibility.
            verbose: Print results.

        Returns:
            Dict with experiment results.
        """
        num_pairs = num_pairs or EVAL_CONFIG["route_test_pairs"]
        rng = np.random.default_rng(seed)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"ROUTE OPTIMIZATION EXPERIMENTS ({num_pairs} pairs)")
            print(f"{'=' * 60}")

        nodes = list(graph.nodes())
        all_comparisons = []

        successful_pairs = 0
        for i in range(num_pairs * 3):  # Try extra pairs in case some fail
            if successful_pairs >= num_pairs:
                break

            src, dst = rng.choice(nodes, size=2, replace=False)

            try:
                results = optimizer.compare_algorithms(int(src), int(dst), verbose=False)

                # Check if at least one algorithm succeeded
                if any(r.success for r in results.values()):
                    comparison = RouteMetrics.compare_routes(results, graph)
                    all_comparisons.extend(comparison)
                    successful_pairs += 1

                    if verbose and successful_pairs <= 5:
                        src_name = graph.nodes[src].get("name", src)
                        dst_name = graph.nodes[dst].get("name", dst)
                        print(f"\n  Pair {successful_pairs}: {src_name} → {dst_name}")
                        for c in comparison:
                            if c["Algorithm"] in results and results[c["Algorithm"]].success:
                                print(f"    {c['Algorithm']:<15} | "
                                      f"Dist: {c['Distance (km)']:>7.2f} km | "
                                      f"Time: {c['Duration (min)']:>7.1f} min | "
                                      f"Comp: {c['Comp. Time (ms)']:>7.3f} ms")
            except Exception:
                continue

        # Aggregate results
        aggregated = RouteMetrics.aggregate_route_metrics(all_comparisons)

        experiment_result = {
            "num_pairs_tested": successful_pairs,
            "individual_comparisons": all_comparisons,
            "aggregated_metrics": aggregated,
        }

        self.results["routing"].append(experiment_result)

        if verbose:
            print(f"\n  {'─' * 50}")
            print(f"  Aggregated Results ({successful_pairs} pairs):")
            print(f"  {'Algorithm':<15} │ {'Avg Dist (km)':>13} │ "
                  f"{'Avg Time (min)':>14} │ {'Avg Comp (ms)':>13}")
            print(f"  {'─' * 65}")
            for algo, data in aggregated.items():
                print(f"  {algo:<15} │ {data['Avg Distance (km)']:>13.3f} │ "
                      f"{data['Avg Duration (min)']:>14.2f} │ "
                      f"{data['Avg Comp. Time (ms)']:>13.3f}")

        return experiment_result

    def save_results(self, filename=None):
        """
        Save all experiment results to JSON.

        Args:
            filename: Output filename (default: experiment_results.json).
        """
        filename = filename or os.path.join(EVAL_CONFIG["results_dir"], "experiment_results.json")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                result = convert(obj)
                if result is not obj:
                    return result
                return super().default(obj)

        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)

        print(f"\nResults saved to: {filename}")

    def load_results(self, filename=None):
        """Load experiment results from JSON."""
        filename = filename or os.path.join(EVAL_CONFIG["results_dir"], "experiment_results.json")
        with open(filename, "r") as f:
            self.results = json.load(f)
        return self.results
