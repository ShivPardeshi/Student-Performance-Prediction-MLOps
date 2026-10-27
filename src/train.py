"""
Model training and MLflow tracking module.
Builds preprocessing and regression pipelines, trains baseline and comparative models,
tracks experiments via local MLflow, and saves model artifacts.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    ACTUAL_VS_PREDICTED_PLOT_PATH,
    BASELINE_MODEL_PARAMS,
    DATA_QUALITY_SUMMARY_PATH,
    EXCLUDE_COLUMNS,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_DIR,
    MODEL_METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    RUN2_MODEL_PARAMS,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_and_validate_data
from src.evaluate import compute_regression_metrics, plot_actual_vs_predicted, save_metrics_to_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    model_params: Dict[str, Any],
) -> Pipeline:
    """
    Constructs a robust Scikit-learn Pipeline with median imputation for numeric features,
    most-frequent imputation + OneHotEncoding for categoricals, and RandomForestRegressor.
    """
    # Numerical preprocessing
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    # Categorical preprocessing (handle_unknown='ignore' prevents failure on unseen categories)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(**model_params)),
        ]
    )

    return pipeline


def train_single_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_params: Dict[str, Any],
    run_name: str = "Baseline_Run",
    log_to_mlflow: bool = True,
) -> Tuple[Pipeline, Dict[str, float], Path]:
    """
    Trains a Random Forest pipeline on training data, evaluates on test data,
    logs metadata to MLflow, and creates the actual vs predicted plot.
    """
    numerical_cols = list(X_train.select_dtypes(include=["number"]).columns)
    categorical_cols = list(X_train.select_dtypes(exclude=["number"]).columns)

    pipeline = build_pipeline(numerical_cols, categorical_cols, model_params)
    logger.info(f"Training {run_name} with parameters: {model_params}")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)

    plot_path = REPORTS_DIR / f"{run_name.lower()}_actual_vs_predicted.png"
    plot_actual_vs_predicted(
        y_true=y_test,
        y_pred=y_pred,
        output_path=plot_path,
        model_name=f"Random Forest ({run_name})",
        metrics=metrics,
    )

    if log_to_mlflow:
        try:
            import os
            import mlflow
            import mlflow.sklearn

            # Configure reproducible local file store tracking
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
            MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR.resolve()}")
            mlflow.set_experiment(EXPERIMENT_NAME)

            with mlflow.start_run(run_name=run_name):
                # Log parameters
                mlflow.log_param("model_type", "RandomForestRegressor")
                mlflow.log_param("target_variable", TARGET_COLUMN)
                mlflow.log_param("test_size", TEST_SIZE)
                mlflow.log_param("number_of_input_features", X_train.shape[1])
                for param_name, param_val in model_params.items():
                    mlflow.log_param(param_name, param_val)

                # Log metrics
                for metric_name, metric_val in metrics.items():
                    mlflow.log_metric(metric_name, metric_val)

                # Log artifacts
                mlflow.log_artifact(str(plot_path))
                if DATA_QUALITY_SUMMARY_PATH.exists():
                    mlflow.log_artifact(str(DATA_QUALITY_SUMMARY_PATH))

                # Log model pipeline using standard pickle serialization format
                try:
                    mlflow.sklearn.log_model(pipeline, name="model", serialization_format="pickle")
                except TypeError:
                    mlflow.sklearn.log_model(pipeline, artifact_path="model", serialization_format="pickle")
                logger.info(f"Successfully logged {run_name} to MLflow experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            logger.warning(f"MLflow logging encountered an issue: {e}")

    return pipeline, metrics, plot_path


def run_training_pipeline() -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Main training workflow:
    1. Loads and validates student-mat.csv.
    2. Separates background features from excluded columns (G1, G2, G3).
    3. Performs reproducible train/test split.
    4. Executes Run 1 (Baseline: n_estimators=100, max_depth=8).
    5. Executes Run 2 (Comparative: n_estimators=200, max_depth=12).
    6. Selects the best performing model based on RMSE.
    7. Saves model artifact and metrics.
    """
    df, _ = load_and_validate_data()

    # Exclude G1, G2, and G3 from input features
    feature_columns = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    logger.info(f"Dataset split: {len(feature_columns)} input features, target='{TARGET_COLUMN}'")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 1. Baseline Run
    pipe_baseline, metrics_baseline, plot_baseline = train_single_model(
        X_train, y_train, X_test, y_test,
        model_params=BASELINE_MODEL_PARAMS,
        run_name="Baseline_Model_Run1",
        log_to_mlflow=True,
    )

    # 2. Comparative Run
    pipe_run2, metrics_run2, plot_run2 = train_single_model(
        X_train, y_train, X_test, y_test,
        model_params=RUN2_MODEL_PARAMS,
        run_name="Tuned_Model_Run2",
        log_to_mlflow=True,
    )

    # Compare models (lower RMSE is preferred)
    if metrics_run2["rmse"] < metrics_baseline["rmse"]:
        best_pipeline = pipe_run2
        best_params = RUN2_MODEL_PARAMS
        best_metrics = metrics_run2
        selected_run = "Tuned_Model_Run2"
    else:
        best_pipeline = pipe_baseline
        best_params = BASELINE_MODEL_PARAMS
        best_metrics = metrics_baseline
        selected_run = "Baseline_Model_Run1"

    # Save primary actual_vs_predicted.png
    y_best_pred = best_pipeline.predict(X_test)
    plot_actual_vs_predicted(
        y_true=y_test,
        y_pred=y_best_pred,
        output_path=ACTUAL_VS_PREDICTED_PLOT_PATH,
        model_name=f"Selected Model ({selected_run})",
        metrics=best_metrics,
    )

    # Save trained model pipeline artifact
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info(f"Saved production model pipeline to: {MODEL_PATH}")

    # Save model metrics
    metrics_summary = {
        "selected_model": selected_run,
        "selected_parameters": best_params,
        "metrics": best_metrics,
        "comparison": {
            "Run1_Baseline": {
                "parameters": BASELINE_MODEL_PARAMS,
                "metrics": metrics_baseline,
            },
            "Run2_Comparative": {
                "parameters": RUN2_MODEL_PARAMS,
                "metrics": metrics_run2,
            },
        },
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "input_features_count": len(feature_columns),
        "excluded_features": EXCLUDE_COLUMNS,
    }

    save_metrics_to_json(metrics_summary, MODEL_METRICS_PATH)

    # Print summary comparison
    print("\n" + "=" * 65)
    print("MODEL TRAINING & EXPERIMENT COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Run / Model':<22} | {'MAE':<8} | {'RMSE':<8} | {'R²':<8}")
    print("-" * 65)
    print(f"{'Run 1 (Baseline)':<22} | {metrics_baseline['mae']:<8.4f} | {metrics_baseline['rmse']:<8.4f} | {metrics_baseline['r2']:<8.4f}")
    print(f"{'Run 2 (Comparative)':<22} | {metrics_run2['mae']:<8.4f} | {metrics_run2['rmse']:<8.4f} | {metrics_run2['r2']:<8.4f}")
    print("-" * 65)
    print(f"Selected Model: {selected_run}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved:  {MODEL_METRICS_PATH}")
    print(f"Plot saved:     {ACTUAL_VS_PREDICTED_PLOT_PATH}")
    print("=" * 65 + "\n")

    return best_pipeline, metrics_summary


if __name__ == "__main__":
    run_training_pipeline()
