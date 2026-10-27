"""
Summary generation module for the Student Performance MLOps Project.
Aggregates actual results from data validation, model training, MLflow tracking,
and Evidently drift detection into reports/project_summary.json and prints a clean report.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.config import (
    ACTUAL_VS_PREDICTED_PLOT_PATH,
    CURRENT_DATA_PATH,
    DATA_DRIFT_REPORT_HTML,
    DATA_DRIFT_REPORT_JSON,
    DATA_QUALITY_SUMMARY_PATH,
    EXPERIMENT_NAME,
    FEATURE_DRIFT_SUMMARY_CSV,
    MODEL_METRICS_PATH,
    MODEL_PATH,
    MONITORED_FEATURES,
    PREDICTION_COMPARISON_PLOT_PATH,
    PROJECT_SUMMARY_PATH,
    RAW_DATA_PATH,
    REFERENCE_DATA_PATH,
    REPORTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_project_summary() -> Dict[str, Any]:
    """
    Reads genuine outputs and metrics from the respective artifacts and compiles
    a structured summary dictionary.
    """
    # 1. Dataset stats
    dataset_info = {
        "name": "UCI Student Performance (Mathematics)",
        "source": "https://archive.ics.uci.edu/dataset/320/student+performance",
        "file": str(RAW_DATA_PATH.name),
        "total_records": None,
        "total_columns": None,
        "input_features_count": None,
    }
    if DATA_QUALITY_SUMMARY_PATH.exists():
        with open(DATA_QUALITY_SUMMARY_PATH, "r") as f:
            quality = json.load(f)
            dataset_info["total_records"] = quality.get("num_records")
            dataset_info["total_columns"] = quality.get("num_columns")

    # 2. Model performance stats
    model_info = {
        "experiment_name": EXPERIMENT_NAME,
        "selected_model": None,
        "metrics": {"mae": None, "rmse": None, "r2": None},
        "comparison": {},
    }
    if MODEL_METRICS_PATH.exists():
        with open(MODEL_METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
            model_info["selected_model"] = metrics_data.get("selected_model")
            model_info["metrics"] = metrics_data.get("metrics", {})
            model_info["comparison"] = metrics_data.get("comparison", {})
            dataset_info["input_features_count"] = metrics_data.get("input_features_count")

    # 3. Drift detection stats
    drift_info = {
        "reference_records": None,
        "current_records": None,
        "features_monitored": MONITORED_FEATURES,
        "drifted_features_count": None,
        "dataset_drift_detected": None,
        "feature_details": [],
    }
    if REFERENCE_DATA_PATH.exists():
        import pandas as pd
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        drift_info["reference_records"] = len(ref_df)
    if CURRENT_DATA_PATH.exists():
        import pandas as pd
        curr_df = pd.read_csv(CURRENT_DATA_PATH)
        drift_info["current_records"] = len(curr_df)

    if DATA_DRIFT_REPORT_JSON.exists():
        try:
            with open(DATA_DRIFT_REPORT_JSON, "r") as f:
                drift_data = json.load(f)
                # Check metrics structure
                drift_metrics = drift_data.get("metrics", [])
                for m in drift_metrics:
                    if "DataDrift" in m.get("metric", ""):
                        res = m.get("result", {})
                        drift_info["dataset_drift_detected"] = res.get("dataset_drift")
                        drift_by_cols = res.get("drift_by_columns", {})
                        drifted_count = sum(1 for c in drift_by_cols.values() if c.get("drift_detected", False))
                        drift_info["drifted_features_count"] = drifted_count
                        for col_k, col_v in drift_by_cols.items():
                            drift_info["feature_details"].append({
                                "feature": col_k,
                                "drift_detected": col_v.get("drift_detected"),
                                "stat_test": col_v.get("stat_test_name"),
                                "score": col_v.get("drift_score"),
                            })
                        break
                if drift_info["dataset_drift_detected"] is None:
                    drift_info["dataset_drift_detected"] = drift_data.get("dataset_drift")
                    drift_info["drifted_features_count"] = drift_data.get("drifted_features_count")
                    drift_info["feature_details"] = drift_data.get("features", [])
        except Exception as e:
            logger.warning(f"Could not parse data drift JSON: {e}")

    # 4. Artifact references
    generated_artifacts = {
        "model_pipeline": str(MODEL_PATH) if MODEL_PATH.exists() else None,
        "actual_vs_predicted_plot": str(ACTUAL_VS_PREDICTED_PLOT_PATH) if ACTUAL_VS_PREDICTED_PLOT_PATH.exists() else None,
        "evidently_html_report": str(DATA_DRIFT_REPORT_HTML) if DATA_DRIFT_REPORT_HTML.exists() else None,
        "evidently_json_report": str(DATA_DRIFT_REPORT_JSON) if DATA_DRIFT_REPORT_JSON.exists() else None,
        "feature_drift_summary_csv": str(FEATURE_DRIFT_SUMMARY_CSV) if FEATURE_DRIFT_SUMMARY_CSV.exists() else None,
        "prediction_drift_plot": str(PREDICTION_COMPARISON_PLOT_PATH) if PREDICTION_COMPARISON_PLOT_PATH.exists() else None,
    }

    summary = {
        "project_title": "Student Performance Prediction and Data Drift Detection",
        "dataset": dataset_info,
        "model_training": model_info,
        "data_drift_monitoring": drift_info,
        "artifacts": generated_artifacts,
    }

    # Save summary
    PROJECT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROJECT_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Project summary saved to {PROJECT_SUMMARY_PATH}")

    # Print summary to terminal
    print("\n" + "=" * 70)
    print("                    PROJECT EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Dataset Name:            {dataset_info['name']}")
    print(f"Dataset Size:            {dataset_info['total_records']} rows, {dataset_info['total_columns']} columns")
    print(f"Input Features Count:    {dataset_info['input_features_count']} (excluding G1, G2, G3)")
    print(f"Selected Model:          {model_info['selected_model']}")
    if model_info['metrics'].get('rmse') is not None:
        print(f"Model Metrics:           MAE={model_info['metrics']['mae']:.4f}, RMSE={model_info['metrics']['rmse']:.4f}, R²={model_info['metrics']['r2']:.4f}")
    print(f"MLflow Experiment:       {model_info['experiment_name']}")
    print("-" * 70)
    print(f"Reference Dataset Size:  {drift_info['reference_records']} records")
    print(f"Current Dataset Size:    {drift_info['current_records']} records (with simulated drift)")
    print(f"Monitored Features:      {', '.join(drift_info['features_monitored'])}")
    print(f"Drifted Features Count:  {drift_info['drifted_features_count']} / {len(drift_info['features_monitored'])}")
    print(f"Overall Drift Status:    {'DRIFT DETECTED' if drift_info['dataset_drift_detected'] else 'NO OVERALL DRIFT'}")
    print("-" * 70)
    print("Generated Artifacts:")
    for name, path in generated_artifacts.items():
        status = "EXISTS" if path else "PENDING"
        print(f"  - {name:<26}: [{status}] {path or ''}")
    print("=" * 70 + "\n")

    return summary


if __name__ == "__main__":
    generate_project_summary()
