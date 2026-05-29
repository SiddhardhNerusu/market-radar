"""LightGBM model + training pipeline.

Builds a binary classifier that predicts the probability of a positive 5-day
return given a signal's features. Trained on the union of backfilled SEC
filings (with resolved outcomes) and live signals once their outcomes
resolve. Versioned model files live in ``data/models/``.
"""
from .features import FEATURE_NAMES, extract_features, extract_features_df
from .predict import ModelPredictor, get_predictor
from .train import TrainResult, train_and_save

__all__ = [
    "FEATURE_NAMES",
    "extract_features",
    "extract_features_df",
    "ModelPredictor",
    "get_predictor",
    "TrainResult",
    "train_and_save",
]
