"""
LightGBM Demand Forecaster with Lag Features.

Converts the hourly demand time series into a supervised learning problem
by engineering lag features, then trains a LightGBM gradient boosting model.

Lag features used:
  - demand_lag_1h  to demand_lag_24h   → recent hours
  - demand_lag_48h, demand_lag_72h     → 2 and 3 days ago
  - demand_lag_168h                    → exactly 1 week ago (same day/hour)
  - rolling_mean_3h, rolling_mean_6h   → short-term trend
  - rolling_mean_24h, rolling_std_24h  → daily average and variance
  - Same temporal features as Transformer: hour_sin/cos, dow_sin/cos, etc.

Multi-step forecasting strategy:
  - Recursive (chain): predict step 1, use as lag for step 2, etc.
  - Direct (multi-output): separate model per horizon step (used here,
    more robust to error accumulation)
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle
import time
import json
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FORECAST_CONFIG, MODELS_DIR


LGBM_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_demand_model.pkl")
LGBM_HISTORY_PATH = os.path.join(MODELS_DIR, "lgbm_training_history.json")


def build_lag_features(demand_series: np.ndarray,
                       timestamps: pd.DatetimeIndex,
                       lookback: int = 24,
                       forecast_horizon: int = 6) -> tuple:
    """
    Convert raw demand array into (X, y) supervised learning format.

    Each row of X = lag features for one prediction window.
    Each row of y = next `forecast_horizon` demand values.

    Args:
        demand_series: Raw (unnormalized) demand values, shape (N,)
        timestamps: Datetime index matching demand_series
        lookback: Number of past hours to use as lag features
        forecast_horizon: Number of steps ahead to predict

    Returns:
        X: Feature matrix, shape (N_samples, n_features)
        y: Target matrix, shape (N_samples, forecast_horizon)
        feature_names: List of feature names
    """
    n = len(demand_series)
    max_lag = 168  # 1 week

    # Precompute all lag arrays (vectorized)
    lag_columns = {}

    # Short-range lags: 1h to lookback
    for lag in range(1, lookback + 1):
        lag_columns[f"lag_{lag}h"] = pd.Series(demand_series).shift(lag).values

    # Extended lags: 48h, 72h, 168h
    for lag in [48, 72, 168]:
        lag_columns[f"lag_{lag}h"] = pd.Series(demand_series).shift(lag).values

    # Rolling statistics
    for window in [3, 6, 24]:
        lag_columns[f"rolling_mean_{window}h"] = (
            pd.Series(demand_series).shift(1).rolling(window, min_periods=1).mean().values
        )
    lag_columns["rolling_std_24h"] = (
        pd.Series(demand_series).shift(1).rolling(24, min_periods=1).std().fillna(0).values
    )

    # Temporal features from timestamps
    timestamps = pd.DatetimeIndex(timestamps)
    hour = timestamps.hour.astype(np.float32)
    dow = timestamps.dayofweek.values.astype(np.float32)
    month = timestamps.month.values.astype(np.float32)

    lag_columns["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    lag_columns["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    lag_columns["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    lag_columns["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    lag_columns["month_sin"] = np.sin(2 * np.pi * month / 12)
    lag_columns["month_cos"] = np.cos(2 * np.pi * month / 12)
    lag_columns["is_weekend"] = (dow >= 5).astype(np.float32)
    lag_columns["is_rush_hour"] = (
        ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))
    ).astype(np.float32)

    X_df = pd.DataFrame(lag_columns)
    feature_names = list(X_df.columns)

    # Build target matrix (multi-step)
    y_rows = []
    for step in range(1, forecast_horizon + 1):
        y_rows.append(pd.Series(demand_series).shift(-step).values)
    y_df = pd.DataFrame(np.column_stack(y_rows))

    # Drop rows with any NaN (due to lags or targets)
    full_df = pd.concat([X_df, y_df], axis=1)
    full_df = full_df.iloc[max_lag: n - forecast_horizon].dropna()

    X = full_df.iloc[:, :len(feature_names)].values
    y = full_df.iloc[:, len(feature_names):].values  # (N, horizon)

    return X, y, feature_names


class LGBMDemandForecaster:
    """
    Multi-output LightGBM demand forecaster.

    Trains one LightGBM model per forecast horizon step (Direct strategy).
    Each model is independently optimized — no error accumulation.
    """

    def __init__(self, lookback: int = 24, forecast_horizon: int = 6,
                 n_estimators: int = 500, learning_rate: float = 0.05,
                 max_depth: int = 6, num_leaves: int = 63):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.models = []           # One per horizon step
        self.scaler_mean = None
        self.scaler_std = None
        self.feature_names = None
        self.is_trained = False
        self.training_history = {}

    def _normalize(self, y):
        return (y - self.scaler_mean) / (self.scaler_std + 1e-8)

    def _denormalize(self, y):
        return y * (self.scaler_std + 1e-8) + self.scaler_mean

    def train(self, demand_df: pd.DataFrame, verbose: bool = True,
              callback=None) -> dict:
        """
        Train the LightGBM forecaster.

        Args:
            demand_df: Must have columns ['timestamp', 'demand'] (unnormalized).
            verbose: Print training progress.
            callback: Optional (step, total_steps) callable for UI progress.

        Returns:
            dict with training metrics and history.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("TRAINING LGBM DEMAND FORECASTER")
            print(f"{'=' * 60}")

        t_start = time.time()

        # Ensure sorted
        demand_df = demand_df.sort_values("timestamp").reset_index(drop=True)
        timestamps = pd.to_datetime(demand_df["timestamp"])
        raw_demand = demand_df["demand"].values.astype(np.float64)

        # Fit scaler on raw demand
        self.scaler_mean = raw_demand.mean()
        self.scaler_std = raw_demand.std()
        demand_norm = self._normalize(raw_demand)

        if verbose:
            print(f"  Demand: mean={self.scaler_mean:.1f}, std={self.scaler_std:.1f}")
            print(f"  Time steps: {len(raw_demand)}")

        # Build lag features
        X, y, feature_names = build_lag_features(
            demand_norm, timestamps, self.lookback, self.forecast_horizon
        )
        self.feature_names = feature_names

        if verbose:
            print(f"  Samples: {len(X):,} | Features: {X.shape[1]}")

        # Temporal train/val split (80/20)
        split = int(len(X) * 0.80)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        if verbose:
            print(f"  Split: {len(X_train)} train / {len(X_val)} val")

        self.models = []
        val_maes = []
        val_rmses = []
        step_metrics = []

        total_steps = self.forecast_horizon

        for step in range(self.forecast_horizon):
            y_tr = y_train[:, step]
            y_vl = y_val[:, step]

            lgb_params = {
                "objective": "regression",
                "metric": "rmse",
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "num_leaves": self.num_leaves,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "n_jobs": -1,
                "random_state": 42,
                "verbose": -1,
            }

            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(
                X_train, y_tr,
                eval_set=[(X_val, y_vl)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )

            pred_val = model.predict(X_val)
            mae = mean_absolute_error(y_vl, pred_val)
            rmse = np.sqrt(mean_squared_error(y_vl, pred_val))
            val_maes.append(mae)
            val_rmses.append(rmse)

            step_metrics.append({
                "horizon_step": step + 1,
                "val_mae": float(mae),
                "val_rmse": float(rmse),
                "best_iteration": model.best_iteration_,
            })

            self.models.append(model)

            if verbose:
                print(f"  Step {step+1}/{self.forecast_horizon}: "
                      f"MAE={mae:.4f}, RMSE={rmse:.4f}, "
                      f"trees={model.best_iteration_}")

            if callback:
                callback(step + 1, total_steps)

        elapsed = time.time() - t_start
        overall_mae = float(np.mean(val_maes))
        overall_rmse = float(np.mean(val_rmses))

        # R² on full val
        y_pred_all = np.column_stack([m.predict(X_val) for m in self.models])
        ss_res = np.sum((y_val - y_pred_all) ** 2)
        ss_tot = np.sum((y_val - y_val.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))

        self.training_history = {
            "model": "LightGBM",
            "lookback": self.lookback,
            "forecast_horizon": self.forecast_horizon,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "val_mae": overall_mae,
            "val_rmse": overall_rmse,
            "val_r2": r2,
            "step_metrics": step_metrics,
            "training_time_s": elapsed,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_features": int(X.shape[1]),
        }

        self.is_trained = True

        if verbose:
            print(f"\n  ─── Summary ───")
            print(f"  Val MAE (avg):  {overall_mae:.4f}")
            print(f"  Val RMSE (avg): {overall_rmse:.4f}")
            print(f"  Val R²:         {r2:.4f}")
            print(f"  Training time:  {elapsed:.1f}s")
            print(f"{'=' * 60}")

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all horizon steps given feature matrix X."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return np.column_stack([m.predict(X) for m in self.models])

    def evaluate(self, demand_df: pd.DataFrame) -> dict:
        """
        Evaluate on a held-out demand DataFrame.

        Args:
            demand_df: Must contain ['timestamp', 'demand'] (unnormalized).

        Returns:
            dict with test_mae, test_rmse, test_mape, test_r2, predictions, targets.
        """
        demand_df = demand_df.sort_values("timestamp").reset_index(drop=True)
        timestamps = pd.to_datetime(demand_df["timestamp"])
        raw_demand = demand_df["demand"].values.astype(np.float64)
        demand_norm = self._normalize(raw_demand)

        X, y, _ = build_lag_features(
            demand_norm, timestamps, self.lookback, self.forecast_horizon
        )

        # Use last 20% as test
        split = int(len(X) * 0.80)
        X_test, y_test = X[split:], y[split:]

        preds = self.predict(X_test)

        mae = float(mean_absolute_error(y_test.flatten(), preds.flatten()))
        rmse = float(np.sqrt(mean_squared_error(y_test.flatten(), preds.flatten())))
        mape = float(np.mean(np.abs((y_test - preds) / (np.abs(y_test) + 1e-8))) * 100)
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))

        return {
            "test_mae": mae,
            "test_rmse": rmse,
            "test_mape": mape,
            "test_r2": r2,
            "predictions": preds.tolist(),
            "targets": y_test.tolist(),
        }

    def feature_importance(self) -> pd.DataFrame:
        """Aggregate feature importance across all horizon models."""
        if not self.is_trained or not self.models:
            return pd.DataFrame()

        importances = np.zeros(len(self.feature_names))
        for model in self.models:
            importances += model.feature_importances_
        importances /= len(self.models)

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    def save(self, path: str = LGBM_MODEL_PATH):
        """Save the full forecaster to disk."""
        state = {
            "lookback": self.lookback,
            "forecast_horizon": self.forecast_horizon,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "feature_names": self.feature_names,
            "models": self.models,
            "training_history": self.training_history,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

        # Also save history as JSON for UI
        with open(LGBM_HISTORY_PATH, "w") as f:
            json.dump(self.training_history, f, indent=2)

        print(f"Saved LGBM model to {path}")

    @classmethod
    def load(cls, path: str = LGBM_MODEL_PATH) -> "LGBMDemandForecaster":
        """Load a saved forecaster from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.lookback = state["lookback"]
        obj.forecast_horizon = state["forecast_horizon"]
        obj.n_estimators = state["n_estimators"]
        obj.learning_rate = state["learning_rate"]
        obj.max_depth = state["max_depth"]
        obj.num_leaves = state["num_leaves"]
        obj.scaler_mean = state["scaler_mean"]
        obj.scaler_std = state["scaler_std"]
        obj.feature_names = state["feature_names"]
        obj.models = state["models"]
        obj.training_history = state["training_history"]
        obj.is_trained = True
        return obj
