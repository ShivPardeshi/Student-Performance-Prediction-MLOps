"""
Evaluation metrics and plotting utilities for model assessment.
Computes MAE, RMSE, and R2, and generates an actual vs. predicted plot.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import ACTUAL_VS_PREDICTED_PLOT_PATH, MODEL_METRICS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R-squared (R2)
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    # Compatible RMSE calculation across scikit-learn versions
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }

    return metrics


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path = ACTUAL_VS_PREDICTED_PLOT_PATH,
    model_name: str = "Random Forest Regressor",
    metrics: Dict[str, float] = None,
) -> Path:
    """
    Creates and saves a publication-quality actual vs. predicted plot
    with diagonal reference line, residuals annotations, and clean aesthetics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(
        y_true_arr,
        y_pred_arr,
        color="#2b5c8f",
        alpha=0.65,
        edgecolors="w",
        s=70,
        label="Test Students",
    )

    # Diagonal ideal line
    min_val = min(float(y_true_arr.min()), float(y_pred_arr.min())) - 1
    max_val = max(float(y_true_arr.max()), float(y_pred_arr.max())) + 1
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction (y = x)",
    )

    plt.title(
        f"Student Performance Prediction: Actual vs. Predicted (G3)\n{model_name}",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )
    plt.xlabel("Actual Mathematics Grade (G3)", fontsize=11, labelpad=10)
    plt.ylabel("Predicted Mathematics Grade (G3)", fontsize=11, labelpad=10)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle=":", alpha=0.6)

    # Text box for metrics
    if metrics:
        metric_text = (
            f"MAE:  {metrics.get('mae', 0.0):.3f}\n"
            f"RMSE: {metrics.get('rmse', 0.0):.3f}\n"
            f"R²:   {metrics.get('r2', 0.0):.3f}"
        )
        plt.gca().text(
            0.05,
            0.92,
            metric_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
        )

    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Actual vs. Predicted plot saved to: {output_path}")
    return output_path


def save_metrics_to_json(
    metrics: Dict[str, float],
    output_path: Path = MODEL_METRICS_PATH,
    extra_info: Dict = None,
) -> Path:
    """
    Saves metrics dictionary to a JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = dict(metrics)
    if extra_info:
        data.update(extra_info)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Model metrics saved to: {output_path}")
    return output_path
