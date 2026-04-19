"""
PyTorch Dataset and DataLoader utilities for demand forecasting.

Creates sliding-window sequences from time series data for the
Transformer model training and inference.

The DemandDataset class produces (input_sequence, target) pairs where:
- input_sequence: [lookback_window, num_features] tensor of historical data
- target: [forecast_horizon] tensor of future demand values
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FORECAST_CONFIG


class DemandDataset(Dataset):
    """
    PyTorch Dataset for demand forecasting with sliding windows.

    Given a time series of demand and features, creates overlapping windows:
    - Input: past `lookback_window` time steps with all features
    - Target: next `forecast_horizon` time steps of demand only

    Example with lookback=24, horizon=6:
    Input:  [t-24, t-23, ..., t-1] × [demand, hour_sin, ...]
    Target: [t, t+1, ..., t+5] → demand values only
    """

    def __init__(self, data, feature_columns, lookback_window=None, forecast_horizon=None):
        """
        Initialize the dataset.

        Args:
            data: DataFrame or numpy array with feature columns.
            feature_columns: List of column names to use as input features.
            lookback_window: Number of past time steps to use as input.
            forecast_horizon: Number of future time steps to predict.
        """
        self.lookback = lookback_window or FORECAST_CONFIG["lookback_window"]
        self.horizon = forecast_horizon or FORECAST_CONFIG["forecast_horizon"]
        self.feature_columns = feature_columns

        if isinstance(data, pd.DataFrame):
            self.features = data[feature_columns].values.astype(np.float32)
            # Demand is always the first feature column for target extraction
            demand_idx = feature_columns.index("demand") if "demand" in feature_columns else 0
            self.targets = data[feature_columns[demand_idx]].values.astype(np.float32)
        else:
            self.features = data.astype(np.float32)
            self.targets = data[:, 0].astype(np.float32)

        self.num_features = len(feature_columns)
        self.total_len = len(self.features)

        # Valid indices: need lookback history + horizon future
        self.valid_length = self.total_len - self.lookback - self.horizon + 1

        if self.valid_length <= 0:
            raise ValueError(
                f"Data too short ({self.total_len}) for lookback={self.lookback} "
                f"+ horizon={self.horizon}. Need at least {self.lookback + self.horizon} steps."
            )

    def __len__(self):
        """Return number of valid sliding windows."""
        return self.valid_length

    def __getitem__(self, idx):
        """
        Get a single (input, target) pair.

        Args:
            idx: Window index.

        Returns:
            Tuple of (input_seq, target_seq):
            - input_seq: FloatTensor [lookback_window, num_features]
            - target_seq: FloatTensor [forecast_horizon]
        """
        start = idx
        end = start + self.lookback
        target_end = end + self.horizon

        input_seq = torch.FloatTensor(self.features[start:end])
        target_seq = torch.FloatTensor(self.targets[end:target_end])

        return input_seq, target_seq


def create_data_loaders(train_df, val_df, test_df, feature_columns,
                        lookback_window=None, forecast_horizon=None,
                        batch_size=None, verbose=True):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        feature_columns: List of feature column names.
        lookback_window: Past window size.
        forecast_horizon: Future prediction steps.
        batch_size: Training batch size.
        verbose: Print loader info.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders and metadata.
    """
    lookback = lookback_window or FORECAST_CONFIG["lookback_window"]
    horizon = forecast_horizon or FORECAST_CONFIG["forecast_horizon"]
    bs = batch_size or FORECAST_CONFIG["batch_size"]

    # Create datasets
    train_dataset = DemandDataset(train_df, feature_columns, lookback, horizon)
    val_dataset = DemandDataset(val_df, feature_columns, lookback, horizon)
    test_dataset = DemandDataset(test_df, feature_columns, lookback, horizon)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=0, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False,
        num_workers=0, drop_last=False
    )

    if verbose:
        print(f"DataLoaders Created:")
        print(f"  Lookback window: {lookback} steps")
        print(f"  Forecast horizon: {horizon} steps")
        print(f"  Batch size: {bs}")
        print(f"  Train: {len(train_dataset)} samples → {len(train_loader)} batches")
        print(f"  Val:   {len(val_dataset)} samples → {len(val_loader)} batches")
        print(f"  Test:  {len(test_dataset)} samples → {len(test_loader)} batches")
        print(f"  Input shape per sample: [{lookback}, {len(feature_columns)}]")
        print(f"  Target shape per sample: [{horizon}]")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "num_features": len(feature_columns),
        "lookback": lookback,
        "horizon": horizon,
    }
