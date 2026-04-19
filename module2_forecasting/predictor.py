"""
Demand Prediction & Inference Module.

Provides utilities for loading trained models and generating demand forecasts
with optional confidence intervals via Monte Carlo Dropout.

Usage:
    predictor = DemandPredictor()
    predictor.load_model(checkpoint_path)
    predictions = predictor.predict(input_sequence)
    predictions_with_ci = predictor.predict_with_uncertainty(input_sequence)
"""

import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FORECAST_CONFIG
from module2_forecasting.transformer_model import DemandTransformer


class DemandPredictor:
    """
    Inference engine for the trained DemandTransformer model.

    Supports:
    - Single-sequence prediction
    - Batch prediction
    - Monte Carlo Dropout for uncertainty estimation
    - Inverse scaling for interpretable results
    """

    def __init__(self, model=None, device=None):
        """
        Initialize predictor.

        Args:
            model: Pre-loaded DemandTransformer model (optional).
            device: torch.device for inference.
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()

    def load_model(self, checkpoint_path=None):
        """
        Load a trained model from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            Self for chaining.
        """
        path = checkpoint_path or FORECAST_CONFIG["model_checkpoint"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint["model_config"]

        self.model = DemandTransformer(
            num_features=config["num_features"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_encoder_layers=config["n_encoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            forecast_horizon=config["forecast_horizon"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        return self

    @torch.no_grad()
    def predict(self, input_sequence):
        """
        Generate demand predictions from input sequence.

        Args:
            input_sequence: numpy array or tensor of shape:
                - [seq_len, num_features] for single prediction
                - [batch_size, seq_len, num_features] for batch prediction

        Returns:
            numpy array of predictions:
                - [forecast_horizon] for single input
                - [batch_size, forecast_horizon] for batch input
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        self.model.eval()

        # Handle numpy arrays
        if isinstance(input_sequence, np.ndarray):
            input_sequence = torch.FloatTensor(input_sequence)

        # Add batch dimension if needed
        single_input = False
        if input_sequence.dim() == 2:
            input_sequence = input_sequence.unsqueeze(0)
            single_input = True

        input_sequence = input_sequence.to(self.device)
        predictions = self.model(input_sequence).cpu().numpy()

        if single_input:
            return predictions[0]  # Remove batch dimension
        return predictions

    def predict_with_uncertainty(self, input_sequence, n_samples=30):
        """
        Generate predictions with uncertainty estimates using MC Dropout.

        Monte Carlo Dropout keeps dropout active during inference and runs
        multiple forward passes to estimate prediction uncertainty. The
        variance across samples approximates model uncertainty.

        Args:
            input_sequence: Input tensor [seq_len, num_features] or
                           [batch_size, seq_len, num_features].
            n_samples: Number of MC forward passes (more = better estimate).

        Returns:
            Dict with:
                - mean: Mean prediction [forecast_horizon]
                - std: Standard deviation [forecast_horizon]
                - lower_ci: Lower 95% CI
                - upper_ci: Upper 95% CI
                - samples: All MC samples [n_samples, forecast_horizon]
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Enable dropout for MC sampling
        self.model.train()

        if isinstance(input_sequence, np.ndarray):
            input_sequence = torch.FloatTensor(input_sequence)

        single_input = False
        if input_sequence.dim() == 2:
            input_sequence = input_sequence.unsqueeze(0)
            single_input = True

        input_sequence = input_sequence.to(self.device)

        all_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(input_sequence).cpu().numpy()
                all_samples.append(pred)

        # Restore eval mode
        self.model.eval()

        # Stack and compute statistics
        samples = np.stack(all_samples, axis=0)  # [n_samples, batch, horizon]

        if single_input:
            samples = samples[:, 0, :]  # [n_samples, horizon]

        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)

        return {
            "mean": mean,
            "std": std,
            "lower_ci": mean - 1.96 * std,
            "upper_ci": mean + 1.96 * std,
            "samples": samples,
        }

    def inverse_transform_predictions(self, predictions, scaler, demand_col_idx=0):
        """
        Inverse-transform normalized predictions back to original scale.

        Args:
            predictions: Normalized predictions array.
            scaler: Fitted StandardScaler from preprocessing.
            demand_col_idx: Index of the demand column in the scaler.

        Returns:
            Predictions in original scale.
        """
        mean = scaler.mean_[demand_col_idx]
        scale = scaler.scale_[demand_col_idx]
        return predictions * scale + mean
