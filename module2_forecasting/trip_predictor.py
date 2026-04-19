"""
Trip Duration Predictor for NYC Taxi Competition.

Trains a Gradient Boosting regressor on spatial + temporal features
extracted from train.csv, then predicts trip_duration (seconds) for
test.csv trips in Kaggle submission format.

Features used:
- Haversine distance between pickup and dropoff
- Pickup hour, day_of_week, month
- is_rush_hour, is_weekend
- Vendor ID, passenger count
- Direction angle (bearing)
- Log-transform target (trip_duration → log1p) for stability
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle
import time
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_CONFIG, MODELS_DIR


TRIP_MODEL_PATH = os.path.join(MODELS_DIR, "trip_duration_model.pkl")
TRIP_SCALER_PATH = os.path.join(MODELS_DIR, "trip_duration_scaler.pkl")


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (km)."""
    R = 6371.0
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))


def _bearing(lat1, lon1, lat2, lon2):
    """Compute bearing angle between two coordinates (degrees)."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract feature matrix from a trip DataFrame.

    Works on both train.csv and test.csv (no trip_duration needed).

    Returns:
        X: np.ndarray of shape (N, 13)
    """
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    hour = df["pickup_datetime"].dt.hour.values
    dow = df["pickup_datetime"].dt.dayofweek.values
    month = df["pickup_datetime"].dt.month.values

    # Haversine distance (km)
    dist_km = _haversine_km(
        df["pickup_latitude"].values, df["pickup_longitude"].values,
        df["dropoff_latitude"].values, df["dropoff_longitude"].values,
    )

    # Bearing
    bear = _bearing(
        df["pickup_latitude"].values, df["pickup_longitude"].values,
        df["dropoff_latitude"].values, df["dropoff_longitude"].values,
    )

    is_rush = ((hour >= 7) & (hour <= 9) | (hour >= 17) & (hour <= 19)).astype(int)
    is_weekend = (dow >= 5).astype(int)

    passenger = df["passenger_count"].values.clip(1, 9)
    vendor = df["vendor_id"].values

    # Center coordinates (useful features)
    center_lat = (df["pickup_latitude"].values + df["dropoff_latitude"].values) / 2
    center_lon = (df["pickup_longitude"].values + df["dropoff_longitude"].values) / 2

    # Lat/lon deltas
    dlat = df["dropoff_latitude"].values - df["pickup_latitude"].values
    dlon = df["dropoff_longitude"].values - df["pickup_longitude"].values

    X = np.column_stack([
        dist_km,
        bear,
        hour,
        dow,
        month,
        is_rush,
        is_weekend,
        passenger,
        vendor,
        center_lat,
        center_lon,
        dlat,
        dlon,
    ])

    return X


class TripDurationPredictor:
    """
    Gradient Boosting model for NYC Taxi trip duration prediction.

    Target: log1p(trip_duration_seconds) — inverse: expm1
    RMSLE metric (standard for this Kaggle competition).
    """

    def __init__(self, n_estimators=200, max_depth=5, learning_rate=0.1):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            loss="squared_error",
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
            verbose=0,
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, df_train: pd.DataFrame, verbose: bool = True):
        """
        Train on a DataFrame containing trip_duration column.

        Args:
            df_train: Must include trip_duration (seconds) column.
            verbose: Print training progress.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("TRAINING TRIP DURATION PREDICTOR")
            print(f"{'=' * 60}")
            print(f"Training records: {len(df_train):,}")

        # Clean obvious outliers before training
        df_clean = df_train[
            (df_train["trip_duration"] >= 30) &
            (df_train["trip_duration"] <= 7200)  # 30s to 2h
        ].copy()

        if verbose:
            print(f"After cleaning: {len(df_clean):,} records")

        t0 = time.time()
        X = extract_features(df_clean)
        y = np.log1p(df_clean["trip_duration"].values)  # log-transform

        X_scaled = self.scaler.fit_transform(X)

        if verbose:
            print(f"Training GBM ({self.model.n_estimators} estimators)...")

        self.model.fit(X_scaled, y)
        elapsed = time.time() - t0
        self.is_trained = True

        # In-sample RMSLE
        y_pred = self.model.predict(X_scaled)
        rmsle = np.sqrt(mean_squared_error(y, y_pred))
        mae_mins = np.mean(np.abs(np.expm1(y_pred) - np.expm1(y))) / 60

        if verbose:
            print(f"Training time: {elapsed:.1f}s")
            print(f"Train RMSLE:   {rmsle:.4f}")
            print(f"Train MAE:     {mae_mins:.2f} min")
            print(f"{'=' * 60}")

        return {"rmsle": rmsle, "mae_min": mae_mins, "n_train": len(df_clean)}

    def evaluate(self, df_val: pd.DataFrame, verbose: bool = True) -> dict:
        """
        Evaluate on a labeled validation set.

        Returns:
            Dict with rmsle, mae (minutes), r2.
        """
        df_val = df_val[
            (df_val["trip_duration"] >= 30) &
            (df_val["trip_duration"] <= 7200)
        ].copy()

        X = extract_features(df_val)
        X_scaled = self.scaler.transform(X)
        y_true = np.log1p(df_val["trip_duration"].values)
        y_pred = self.model.predict(X_scaled)

        rmsle = np.sqrt(mean_squared_error(y_true, y_pred))
        mae_secs = np.mean(np.abs(np.expm1(y_pred) - np.expm1(y_true)))
        mae_mins = mae_secs / 60
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

        # Also compute actual duration error
        pred_dur = np.expm1(y_pred)
        true_dur = np.expm1(y_true)
        mape = np.mean(np.abs((true_dur - pred_dur) / (true_dur + 1))) * 100

        if verbose:
            print(f"\nValidation Results:")
            print(f"  RMSLE:  {rmsle:.4f}  (Kaggle competition metric)")
            print(f"  MAE:    {mae_mins:.2f} min")
            print(f"  MAPE:   {mape:.2f}%")
            print(f"  R²:     {r2:.4f}")

        return {
            "rmsle": rmsle,
            "mae_min": mae_mins,
            "mape": mape,
            "r2": r2,
            "predictions_sec": pred_dur,
            "targets_sec": true_dur,
        }

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Predict trip_duration (seconds) for test data.

        Args:
            df_test: test.csv DataFrame (no trip_duration needed).

        Returns:
            np.ndarray of predicted trip_duration in seconds.
        """
        X = extract_features(df_test)
        X_scaled = self.scaler.transform(X)
        log_pred = self.model.predict(X_scaled)
        return np.expm1(log_pred).clip(1, 86400)  # clip to 1s – 24h

    def predict_with_id(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predict and return a Kaggle submission DataFrame.

        Returns:
            DataFrame with columns [id, trip_duration].
        """
        preds = self.predict(df_test)
        return pd.DataFrame({"id": df_test["id"].values, "trip_duration": preds.astype(int)})

    def save(self, path: str = TRIP_MODEL_PATH, scaler_path: str = TRIP_SCALER_PATH):
        """Save model and scaler to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str = TRIP_MODEL_PATH, scaler_path: str = TRIP_SCALER_PATH) -> "TripDurationPredictor":
        """Load a saved model from disk."""
        predictor = cls.__new__(cls)
        with open(path, "rb") as f:
            predictor.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            predictor.scaler = pickle.load(f)
        predictor.is_trained = True
        return predictor

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance DataFrame."""
        feature_names = [
            "distance_km", "bearing", "pickup_hour", "pickup_dow", "pickup_month",
            "is_rush_hour", "is_weekend", "passenger_count", "vendor_id",
            "center_lat", "center_lon", "delta_lat", "delta_lon"
        ]
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
