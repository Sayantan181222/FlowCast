"""
Module 1: Data Collection and Preprocessing.

Handles real NYC Taxi Trip Duration data loading, cleaning,
feature engineering, and PyTorch dataset preparation for demand forecasting.
"""

from .data_preprocessor import DataPreprocessor
from .data_loader import DemandDataset, create_data_loaders

__all__ = [
    "DataPreprocessor",
    "DemandDataset",
    "create_data_loaders",
]
