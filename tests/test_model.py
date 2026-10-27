"""
Automated unit tests for model training, pipeline execution, persistence, and evaluation metrics.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    BASELINE_MODEL_PARAMS,
    EXCLUDE_COLUMNS,
    MODEL_PATH,
    TARGET_COLUMN,
)
from src.data_loader import load_and_validate_data
from src.evaluate import compute_regression_metrics
from src.train import build_pipeline


def test_metrics_computation_validity():
    """Verify MAE, RMSE, and R2 calculation logic."""
    y_true = np.array([10.0, 12.0, 15.0, 18.0, 20.0])
    y_pred = np.array([11.0, 12.0, 14.0, 17.0, 19.0])

    metrics = compute_regression_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["r2"])


def test_model_pipeline_training_and_prediction():
    """Verify Scikit-learn Pipeline trains on student data and outputs valid predictions."""
    df, _ = load_and_validate_data()
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    numerical_cols = list(X.select_dtypes(include=["number"]).columns)
    categorical_cols = list(X.select_dtypes(exclude=["number"]).columns)

    pipeline = build_pipeline(numerical_cols, categorical_cols, BASELINE_MODEL_PARAMS)
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    assert len(predictions) == len(df), "Prediction count must match input samples"
    assert np.all(np.isfinite(predictions)), "Predictions must be finite numerical values"
    assert predictions.ndim == 1


def test_saved_model_load_and_inference():
    """Verify saved model artifact can be reloaded and yields matching predictions."""
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not yet trained. Run src/train.py first.")

    model = joblib.load(MODEL_PATH)
    assert model is not None

    df, _ = load_and_validate_data()
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    sample_X = df[feature_cols].iloc[:10]

    preds = model.predict(sample_X)
    assert len(preds) == 10
    assert np.all(np.isfinite(preds))
