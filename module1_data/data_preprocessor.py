"""
Data Preprocessing Pipeline for NYC Taxi Trip Duration Data.

Handles both real NYC Taxi Trip Duration competition data and synthetic data:

Real data schema:
  id, vendor_id, pickup_datetime, dropoff_datetime, passenger_count,
  pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
  store_and_fwd_flag, trip_duration (seconds)

Pipeline Steps:
1. Schema Detection & Normalization
2. Data Cleaning (outlier removal, bounds filtering)
3. Feature Engineering (temporal + spatial features)
4. Demand Aggregation (hourly city-wide)
5. Normalization (Z-score)
6. Temporal Train/Val/Test Split
"""

import numpy as np
import pandas as pd
import math
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_CONFIG, FORECAST_CONFIG, NYC_ZONES


class DataPreprocessor:
    """
    Preprocessing pipeline for NYC Taxi data.

    Handles real competition data (lat/lon) and synthetic data (zone IDs).
    """

    def __init__(self):
        """Initialize preprocessor with zone lookup structures."""
        self.scaler_params = {}
        self.feature_columns = []
        # Build zone lookup arrays for fast nearest-zone assignment
        self._zone_ids = list(NYC_ZONES.keys())
        self._zone_lats = np.array([NYC_ZONES[z][1] for z in self._zone_ids])
        self._zone_lons = np.array([NYC_ZONES[z][2] for z in self._zone_ids])

    def _detect_schema(self, df):
        """Detect whether data is real (lat/lon) or synthetic (zone IDs)."""
        if "pickup_longitude" in df.columns:
            return "real"
        elif "PULocationID" in df.columns:
            return "synthetic"
        else:
            raise ValueError(f"Unrecognized schema. Columns: {list(df.columns)}")

    def _assign_nearest_zone(self, lat, lon):
        """
        Assign the nearest NYC zone to a lat/lon coordinate.

        Uses vectorized Euclidean approximation (fast, accurate for small areas).
        """
        dlat = self._zone_lats - lat
        dlon = self._zone_lons - lon
        dist_sq = dlat ** 2 + dlon ** 2
        return self._zone_ids[np.argmin(dist_sq)]

    def _haversine_km(self, lat1, lon1, lat2, lon2):
        """Compute haversine distance in km (vectorized)."""
        R = 6371.0
        lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    def normalize_schema(self, df, verbose=True):
        """
        Normalize real data to a unified schema.

        Real → computes trip_distance, trip_duration_min, assigns zone IDs.
        Synthetic → passes through as-is.
        """
        schema = self._detect_schema(df)
        df = df.copy()

        if schema == "real":
            if verbose:
                print(f"  Schema: Real NYC Taxi Trip Duration ({len(df):,} records)")

            # Parse datetime
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
            if "dropoff_datetime" in df.columns:
                df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

            # Compute trip_duration_min from trip_duration (seconds)
            df["trip_duration_min"] = df["trip_duration"] / 60.0

            # Compute trip_distance via haversine
            df["trip_distance"] = self._haversine_km(
                df["pickup_latitude"].values, df["pickup_longitude"].values,
                df["dropoff_latitude"].values, df["dropoff_longitude"].values,
            )
            # Convert km to miles for consistency
            df["trip_distance"] = df["trip_distance"] * 0.621371

            # Assign nearest zones — fully vectorized via NumPy broadcasting
            # Shape: (N, num_zones) — compute squared distance for all points at once
            if verbose:
                print(f"  Assigning pickup & dropoff zones (vectorized, {len(df):,} rows)...")

            pu_lats = df["pickup_latitude"].values[:, np.newaxis]    # (N,1)
            pu_lons = df["pickup_longitude"].values[:, np.newaxis]   # (N,1)
            do_lats = df["dropoff_latitude"].values[:, np.newaxis]
            do_lons = df["dropoff_longitude"].values[:, np.newaxis]

            zone_lats = self._zone_lats[np.newaxis, :]  # (1,Z)
            zone_lons = self._zone_lons[np.newaxis, :]  # (1,Z)

            pu_dist_sq = (pu_lats - zone_lats) ** 2 + (pu_lons - zone_lons) ** 2
            do_dist_sq = (do_lats - zone_lats) ** 2 + (do_lons - zone_lons) ** 2

            zone_arr = np.array(self._zone_ids)
            df["PULocationID"] = zone_arr[np.argmin(pu_dist_sq, axis=1)]
            df["DOLocationID"] = zone_arr[np.argmin(do_dist_sq, axis=1)]

            if verbose:
                print(f"  Zone assignment complete: {df['PULocationID'].nunique()} unique pickup zones")

            # Estimate fare (not in real data, approximate for dashboard)
            df["fare_amount"] = 2.50 + df["trip_distance"] * 2.50 + df["trip_duration_min"] * 0.50

        else:
            if verbose:
                print(f"  Schema: Synthetic ({len(df):,} records)")
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
            if "dropoff_datetime" in df.columns:
                df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

        return df

    def clean_data(self, df, verbose=True):
        """
        Clean data by removing outliers and invalid records.

        Filters:
        - Trip duration: 1 min to 180 min (remove extreme outliers)
        - Trip distance: 0.1 to 100 miles
        - Passenger count: 1 to 9
        - NYC bounding box for coordinates
        """
        initial_count = len(df)
        df = df.copy()

        # Duration bounds (1 min to 3 hours)
        if "trip_duration_min" in df.columns:
            df = df[(df["trip_duration_min"] >= 1) & (df["trip_duration_min"] <= 180)]

        # Distance bounds
        if "trip_distance" in df.columns:
            df = df[(df["trip_distance"] >= 0.1) & (df["trip_distance"] <= 100)]

        # Passenger bounds
        if "passenger_count" in df.columns:
            df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 9)]

        # NYC coordinate bounds (if lat/lon present)
        if "pickup_latitude" in df.columns:
            df = df[
                (df["pickup_latitude"] >= 40.5) & (df["pickup_latitude"] <= 41.0) &
                (df["pickup_longitude"] >= -74.3) & (df["pickup_longitude"] <= -73.7) &
                (df["dropoff_latitude"] >= 40.5) & (df["dropoff_latitude"] <= 41.0) &
                (df["dropoff_longitude"] >= -74.3) & (df["dropoff_longitude"] <= -73.7)
            ]

        # Drop nulls
        df = df.dropna(subset=["pickup_datetime", "PULocationID"])

        removed = initial_count - len(df)
        if verbose:
            print(f"Data Cleaning: {initial_count:,} → {len(df):,} records "
                  f"({removed:,} removed, {removed / initial_count * 100:.1f}%)")

        return df.reset_index(drop=True)

    def engineer_features(self, df, verbose=True):
        """
        Add temporal and spatial features.

        Temporal: hour, day_of_week, is_weekend, month, is_rush_hour,
                  day_period, cyclical sin/cos encodings
        """
        df = df.copy()

        # Ensure datetime
        if df["pickup_datetime"].dtype == object:
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

        # --- Temporal Features ---
        df["hour_of_day"] = df["pickup_datetime"].dt.hour
        df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["month"] = df["pickup_datetime"].dt.month
        df["is_rush_hour"] = (
            df["hour_of_day"].between(7, 9) | df["hour_of_day"].between(17, 19)
        ).astype(int)
        df["day_period"] = pd.cut(
            df["hour_of_day"],
            bins=[-1, 6, 12, 18, 24],
            labels=[0, 1, 2, 3],
        ).astype(int)

        # --- Cyclical Encodings ---
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        if verbose:
            print(f"Feature Engineering: Added 12 temporal features")

        return df

    def aggregate_demand(self, df, freq="1h", verbose=True):
        """
        Aggregate trip data into hourly demand per zone, then produce
        a city-wide hourly demand time series for forecasting.
        """
        df = df.copy()
        if df["pickup_datetime"].dtype == object:
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

        df["hour_bucket"] = df["pickup_datetime"].dt.floor(freq)

        # Zone-level hourly demand
        zone_demand = (
            df.groupby(["hour_bucket", "PULocationID"])
            .size()
            .reset_index(name="demand")
        )

        if verbose:
            print(f"Demand Aggregation ({freq}):")
            print(f"  Total hourly records: {len(zone_demand):,}")
            print(f"  Time range: {zone_demand['hour_bucket'].min()} to "
                  f"{zone_demand['hour_bucket'].max()}")
            print(f"  Zones: {zone_demand['PULocationID'].nunique()}")
            print(f"  Avg hourly demand per zone: {zone_demand['demand'].mean():.2f}")
            print(f"  Max hourly demand: {zone_demand['demand'].max()}")

        # City-wide demand series (sum across zones per hour)
        city_demand = (
            zone_demand.groupby("hour_bucket")["demand"]
            .sum()
            .reset_index()
        )
        city_demand.columns = ["timestamp", "demand"]
        city_demand = city_demand.sort_values("timestamp").reset_index(drop=True)

        # Add temporal features to city-wide demand
        city_demand["hour_of_day"] = city_demand["timestamp"].dt.hour
        city_demand["day_of_week"] = city_demand["timestamp"].dt.dayofweek
        city_demand["is_weekend"] = (city_demand["day_of_week"] >= 5).astype(int)
        city_demand["month"] = city_demand["timestamp"].dt.month
        city_demand["is_rush_hour"] = (
            city_demand["hour_of_day"].between(7, 9) | city_demand["hour_of_day"].between(17, 19)
        ).astype(int)

        # Cyclical encodings
        city_demand["hour_sin"] = np.sin(2 * np.pi * city_demand["hour_of_day"] / 24)
        city_demand["hour_cos"] = np.cos(2 * np.pi * city_demand["hour_of_day"] / 24)
        city_demand["dow_sin"] = np.sin(2 * np.pi * city_demand["day_of_week"] / 7)
        city_demand["dow_cos"] = np.cos(2 * np.pi * city_demand["day_of_week"] / 7)
        city_demand["month_sin"] = np.sin(2 * np.pi * city_demand["month"] / 12)
        city_demand["month_cos"] = np.cos(2 * np.pi * city_demand["month"] / 12)

        if verbose:
            print(f"City-wide Demand Series:")
            print(f"  Length: {len(city_demand)} time steps")
            print(f"  Avg demand: {city_demand['demand'].mean():.1f} trips/hour")
            print(f"  Std demand: {city_demand['demand'].std():.1f}")

        return city_demand

    def normalize(self, df, feature_columns=None, verbose=True):
        """Apply Z-score normalization to feature columns."""
        df = df.copy()

        if feature_columns is None:
            feature_columns = [
                "demand", "hour_of_day", "day_of_week", "is_weekend",
                "is_rush_hour", "hour_sin", "hour_cos",
                "dow_sin", "dow_cos", "month_sin", "month_cos",
            ]
        # Keep only columns that exist
        feature_columns = [c for c in feature_columns if c in df.columns]
        self.feature_columns = feature_columns

        for col in feature_columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val == 0:
                std_val = 1.0
            self.scaler_params[col] = {"mean": mean_val, "std": std_val}
            df[col] = (df[col] - mean_val) / std_val

        if verbose:
            print(f"Normalization: Applied Z-score to {len(feature_columns)} features")

        return df

    def split_temporal(self, df, verbose=True):
        """Split data temporally (preserving time order)."""
        n = len(df)
        train_end = int(n * FORECAST_CONFIG["train_ratio"])
        val_end = train_end + int(n * FORECAST_CONFIG["val_ratio"])

        train = df.iloc[:train_end].reset_index(drop=True)
        val = df.iloc[train_end:val_end].reset_index(drop=True)
        test = df.iloc[val_end:].reset_index(drop=True)

        if verbose:
            print(f"Temporal Split:")
            print(f"  Train: {len(train):,} samples ({FORECAST_CONFIG['train_ratio'] * 100:.0f}%)")
            print(f"  Val:   {len(val):,} samples ({FORECAST_CONFIG['val_ratio'] * 100:.0f}%)")
            print(f"  Test:  {len(test):,} samples ({FORECAST_CONFIG['test_ratio'] * 100:.0f}%)")

        return train, val, test

    def run_pipeline(self, df, save=True, verbose=True):
        """
        Execute the full preprocessing pipeline.

        Args:
            df: Raw DataFrame (real or synthetic).
            save: Save processed artifacts to disk.
            verbose: Print progress.

        Returns:
            Dict with processed data, splits, and metadata.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("DATA PREPROCESSING PIPELINE")
            print(f"{'=' * 60}")

        # Step 1: Schema normalization
        if verbose:
            print(f"\n--- Step 1: Schema Normalization ---")
        df = self.normalize_schema(df, verbose=verbose)

        # Step 2: Cleaning
        if verbose:
            print(f"\n--- Step 2: Data Cleaning ---")
        df = self.clean_data(df, verbose=verbose)

        # Step 3: Feature Engineering
        if verbose:
            print(f"\n--- Step 3: Feature Engineering ---")
        df = self.engineer_features(df, verbose=verbose)

        # Step 4: Demand Aggregation
        if verbose:
            print(f"\n--- Step 4: Demand Aggregation ---")
        demand_df = self.aggregate_demand(df, verbose=verbose)

        # Step 5: Normalization
        if verbose:
            print(f"\n--- Step 5: Normalization ---")
        feature_cols = [
            "demand", "hour_of_day", "day_of_week", "is_weekend",
            "is_rush_hour", "hour_sin", "hour_cos",
            "dow_sin", "dow_cos", "month_sin", "month_cos",
        ]
        demand_normalized = self.normalize(demand_df.copy(), feature_cols, verbose=verbose)

        # Step 6: Split
        if verbose:
            print(f"\n--- Step 6: Train/Val/Test Split ---")
        train, val, test = self.split_temporal(demand_normalized, verbose=verbose)

        # Save artifacts
        if save:
            if verbose:
                print(f"\n--- Saving Artifacts ---")
            df.to_csv(DATA_CONFIG["processed_data_file"], index=False)
            demand_df.to_csv(DATA_CONFIG["demand_data_file"], index=False)

            scaler_data = {
                "scaler_params": self.scaler_params,
                "feature_columns": self.feature_columns,
            }
            with open(FORECAST_CONFIG["scaler_path"], "wb") as f:
                pickle.dump(scaler_data, f)
            if verbose:
                print("  Saved demand data, scaler, and processed trips")

        if verbose:
            print(f"\n{'=' * 60}")
            print("PREPROCESSING COMPLETE")
            print(f"{'=' * 60}")

        return {
            "processed_trips": df,
            "demand": demand_df,
            "train": train,
            "val": val,
            "test": test,
            "feature_columns": self.feature_columns,
            "scaler_params": self.scaler_params,
        }
