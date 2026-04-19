"""
Transformer Model Trainer.

Implements the complete training pipeline for the DemandTransformer model:
- Training loop with gradient clipping
- Validation with early stopping
- Learning rate scheduling (ReduceLROnPlateau)
- Model checkpointing (best validation loss)
- Training history tracking and persistence

The trainer supports both full training and resuming from checkpoints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FORECAST_CONFIG


class TransformerTrainer:
    """
    Training manager for the DemandTransformer model.

    Handles the full training lifecycle including optimization, validation,
    early stopping, learning rate scheduling, and checkpoint management.
    """

    def __init__(self, model, device=None):
        """
        Initialize the trainer.

        Args:
            model: DemandTransformer model instance.
            device: torch.device (auto-detects if None).
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Loss function: MSE for regression
        self.criterion = nn.MSELoss()

        # Optimizer: AdamW with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=FORECAST_CONFIG["learning_rate"],
            weight_decay=1e-4,
        )

        # Learning rate scheduler: reduce LR when validation loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=FORECAST_CONFIG["lr_scheduler_factor"],
            patience=FORECAST_CONFIG["lr_scheduler_patience"],
            min_lr=1e-6,
        )

        # Training state
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "learning_rates": [],
            "epoch_times": [],
        }
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.patience_counter = 0
        self.current_epoch = 0

    def _train_epoch(self, train_loader):
        """
        Run one training epoch.

        Args:
            train_loader: DataLoader for training data.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, val_loader):
        """
        Run validation and compute metrics.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Dict with val_loss, val_mae, val_rmse.
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Compute MAE and RMSE
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))

        return {
            "val_loss": avg_loss,
            "val_mae": mae,
            "val_rmse": rmse,
            "predictions": all_predictions,
            "targets": all_targets,
        }

    def train(self, train_loader, val_loader, max_epochs=None,
              early_stopping_patience=None, verbose=True, callback=None):
        """
        Execute the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            max_epochs: Maximum training epochs.
            early_stopping_patience: Epochs to wait before stopping.
            verbose: Print training progress.
            callback: Optional callback function(epoch, history) called each epoch.

        Returns:
            Training history dict.
        """
        max_epochs = max_epochs or FORECAST_CONFIG["max_epochs"]
        patience = early_stopping_patience or FORECAST_CONFIG["early_stopping_patience"]

        if verbose:
            params = self.model.count_parameters()
            print(f"\n{'=' * 70}")
            print(f"TRAINING DEMAND TRANSFORMER")
            print(f"{'=' * 70}")
            print(f"Device: {self.device}")
            print(f"Parameters: {params['trainable']:,}")
            print(f"Max epochs: {max_epochs} | Early stopping patience: {patience}")
            print(f"Learning rate: {FORECAST_CONFIG['learning_rate']}")
            print(f"{'─' * 70}")
            print(f"{'Epoch':>5} │ {'Train Loss':>12} │ {'Val Loss':>12} │ "
                  f"{'Val MAE':>10} │ {'Val RMSE':>10} │ {'LR':>10} │ {'Time':>6}")
            print(f"{'─' * 70}")

        for epoch in range(1, max_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss = self._train_epoch(train_loader)

            # Validate
            val_results = self._validate(val_loader)
            val_loss = val_results["val_loss"]
            val_mae = val_results["val_mae"]
            val_rmse = val_results["val_rmse"]

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_mae)
            self.history["val_rmse"].append(val_rmse)
            self.history["learning_rates"].append(current_lr)
            self.history["epoch_times"].append(epoch_time)

            # Check for improvement
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.best_model_state = deepcopy(self.model.state_dict())
                self.patience_counter = 0
                marker = " ★"
            else:
                self.patience_counter += 1
                marker = ""

            if verbose:
                print(
                    f"{epoch:>5} │ {train_loss:>12.6f} │ {val_loss:>12.6f} │ "
                    f"{val_mae:>10.4f} │ {val_rmse:>10.4f} │ {current_lr:>10.2e} │ "
                    f"{epoch_time:>5.1f}s{marker}"
                )

            # Callback for UI updates
            if callback:
                callback(epoch, self.history)

            # Early stopping check
            if self.patience_counter >= patience:
                if verbose:
                    print(f"{'─' * 70}")
                    print(f"Early stopping triggered at epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        if verbose:
            print(f"{'─' * 70}")
            print(f"Best validation loss: {self.best_val_loss:.6f}")
            total_time = sum(self.history["epoch_times"])
            print(f"Total training time: {total_time:.1f}s")
            print(f"{'=' * 70}")

        return self.history

    def save_checkpoint(self, path=None):
        """
        Save model checkpoint and training history.

        Args:
            path: Path to save the checkpoint file.
        """
        path = path or FORECAST_CONFIG["model_checkpoint"]
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "current_epoch": self.current_epoch,
            "model_config": {
                "num_features": self.model.num_features,
                "d_model": self.model.d_model,
                "n_heads": self.model.n_heads,
                "n_encoder_layers": self.model.n_layers,
                "dim_feedforward": self.model.dim_ff,
                "dropout": self.model.dropout_rate,
                "forecast_horizon": self.model.forecast_horizon,
            },
        }
        torch.save(checkpoint, path)

        # Save training history as JSON
        history_path = FORECAST_CONFIG["training_history"]
        serializable_history = {
            k: [float(v) for v in vals]
            for k, vals in self.history.items()
        }
        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=2)

    def load_checkpoint(self, path=None):
        """
        Load model checkpoint.

        Args:
            path: Path to the checkpoint file.
        """
        path = path or FORECAST_CONFIG["model_checkpoint"]
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.current_epoch = checkpoint["current_epoch"]

        # Load training history if available
        history_path = FORECAST_CONFIG["training_history"]
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                self.history = json.load(f)

    @torch.no_grad()
    def evaluate(self, test_loader, verbose=True):
        """
        Evaluate model on test set.

        Args:
            test_loader: DataLoader for test data.
            verbose: Print results.

        Returns:
            Dict with test metrics and predictions.
        """
        results = self._validate(test_loader)

        if verbose:
            print(f"\nTest Set Evaluation:")
            print(f"  MSE Loss: {results['val_loss']:.6f}")
            print(f"  MAE:      {results['val_mae']:.4f}")
            print(f"  RMSE:     {results['val_rmse']:.4f}")

        return {
            "test_loss": results["val_loss"],
            "test_mae": results["val_mae"],
            "test_rmse": results["val_rmse"],
            "predictions": results["predictions"],
            "targets": results["targets"],
        }
