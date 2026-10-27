"""
Data drift detection module using Evidently.
Compares feature distributions between reference and current datasets,
detects statistical drift across monitored features, generates HTML & JSON reports,
extracts feature-level drift summaries, and monitors prediction drift.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    CURRENT_DATA_PATH,
    DATA_DRIFT_REPORT_HTML,
    DATA_DRIFT_REPORT_JSON,
    FEATURE_DRIFT_SUMMARY_CSV,
    MODEL_PATH,
    MONITORED_FEATURES,
    PREDICTION_COMPARISON_PLOT_PATH,
    REFERENCE_DATA_PATH,
    REPORTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_evidently_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features_to_monitor: List[str] = MONITORED_FEATURES,
    html_output_path: Path = DATA_DRIFT_REPORT_HTML,
    json_output_path: Path = DATA_DRIFT_REPORT_JSON,
    summary_csv_path: Path = FEATURE_DRIFT_SUMMARY_CSV,
) -> Dict[str, Any]:
    """
    Executes Evidently data drift analysis using supported Report and DataDriftPreset.
    Saves HTML report, JSON report, and structured feature-level CSV summary.
    """
    # Filter datasets to monitored features only (ensures G3 is excluded)
    available_monitored = [f for f in features_to_monitor if f in reference_df.columns and f in current_df.columns]
    ref_subset = reference_df[available_monitored].copy()
    curr_subset = current_df[available_monitored].copy()

    logger.info(f"Analyzing drift for features: {available_monitored}")

    html_output_path = Path(html_output_path)
    json_output_path = Path(json_output_path)
    summary_csv_path = Path(summary_csv_path)

    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        try:
            from evidently.legacy.report import Report
            from evidently.legacy.metric_preset import DataDriftPreset
        except (ImportError, ModuleNotFoundError):
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_subset, current_data=curr_subset)

        # 1. Save HTML report
        report.save_html(str(html_output_path))
        logger.info(f"Saved Evidently HTML report to: {html_output_path}")

        # 2. Save JSON report
        json_str = report.json()
        report_dict = json.loads(json_str)
        with open(json_output_path, "w") as f:
            json.dump(report_dict, f, indent=4)
        logger.info(f"Saved Evidently JSON report to: {json_output_path}")

    except Exception as e:
        logger.warning(f"Evidently Report failed or API version differs: {e}. Running robust statistical fallback extraction.")
        report_dict = _run_statistical_drift_fallback(ref_subset, curr_subset, html_output_path, json_output_path)

    # 3. Parse feature-level drift metrics from report_dict or fallback
    drift_summary = _parse_drift_results(report_dict, ref_subset, curr_subset)

    # 4. Save feature drift summary to CSV
    df_summary = pd.DataFrame(drift_summary["features"])
    df_summary.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved feature drift summary CSV to: {summary_csv_path}")

    # Print summary table
    print("\n" + "=" * 75)
    print("EVIDENTLY DATA DRIFT DETECTION RESULTS")
    print("=" * 75)
    print(f"Dataset-level Drift Detected: {drift_summary['dataset_drift']}")
    print(f"Drifted Features Count:       {drift_summary['drifted_features_count']} / {len(available_monitored)}")
    print(f"Drift Share Threshold:        {drift_summary.get('drift_share', 'N/A')}")
    print("-" * 75)
    print(f"{'Feature':<12} | {'Drift?':<8} | {'Score/p-val':<12} | {'Threshold':<10} | {'Test Used':<14}")
    print("-" * 75)
    for feat in drift_summary["features"]:
        feat_name = feat["feature_name"]
        is_drift = "YES" if feat["drift_detected"] else "NO"
        score = f"{feat['stat_test_score']:.4f}" if isinstance(feat['stat_test_score'], (int, float)) else str(feat['stat_test_score'])
        thresh = f"{feat['threshold']:.4f}" if isinstance(feat['threshold'], (int, float)) else str(feat['threshold'])
        test_name = str(feat["stat_test_name"])[:14]
        print(f"{feat_name:<12} | {is_drift:<8} | {score:<12} | {thresh:<10} | {test_name:<14}")
    print("=" * 75 + "\n")

    return drift_summary


def _parse_drift_results(
    report_dict: Dict[str, Any],
    ref_subset: pd.DataFrame,
    curr_subset: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Extracts structured feature-level and dataset-level metrics from Evidently JSON dictionary.
    """
    features_list = []
    dataset_drift = False
    drift_share = 0.5
    drifted_count = 0

    # Locate DataDriftTable metric inside Evidently structure
    metrics = report_dict.get("metrics", [])
    drift_table_result = None

    for m in metrics:
        metric_name = m.get("metric", "")
        if "DataDriftTable" in metric_name or "DataDrift" in metric_name:
            drift_table_result = m.get("result", {})
            break

    if drift_table_result:
        dataset_drift = drift_table_result.get("dataset_drift", False)
        drift_share = drift_table_result.get("drift_share", 0.5)
        drift_by_columns = drift_table_result.get("drift_by_columns", {})

        for col_name, col_data in drift_by_columns.items():
            drift_detected = bool(col_data.get("drift_detected", False))
            if drift_detected:
                drifted_count += 1

            ref_stat = float(ref_subset[col_name].mean()) if col_name in ref_subset.columns else None
            curr_stat = float(curr_subset[col_name].mean()) if col_name in curr_subset.columns else None

            features_list.append({
                "feature_name": col_name,
                "drift_detected": drift_detected,
                "stat_test_name": col_data.get("stat_test_name", "kolmogorov-smirnov"),
                "stat_test_score": col_data.get("drift_score", col_data.get("p_value", 0.0)),
                "threshold": col_data.get("stat_test_threshold", 0.05),
                "ref_mean": ref_stat,
                "curr_mean": curr_stat,
            })
    else:
        # Fallback if structure is custom
        dataset_drift = report_dict.get("dataset_drift", False)
        drifted_count = report_dict.get("drifted_features_count", 0)
        features_list = report_dict.get("features", [])

    return {
        "dataset_drift": dataset_drift,
        "drift_share": drift_share,
        "drifted_features_count": drifted_count,
        "total_monitored_features": len(features_list),
        "features": features_list,
    }


def _run_statistical_drift_fallback(
    ref_subset: pd.DataFrame,
    curr_subset: pd.DataFrame,
    html_output_path: Path,
    json_output_path: Path,
) -> Dict[str, Any]:
    """
    Standard statistical drift detection fallback (Kolmogorov-Smirnov two-sample test)
    used if Evidently dependencies are incomplete or environment is restricted.
    """
    from scipy.stats import ks_2samp

    features_list = []
    drifted_count = 0
    alpha = 0.05

    for col in ref_subset.columns:
        stat_res = ks_2samp(ref_subset[col], curr_subset[col])
        p_val = float(stat_res.pvalue)
        drift_detected = p_val < alpha
        if drift_detected:
            drifted_count += 1

        features_list.append({
            "feature_name": col,
            "drift_detected": drift_detected,
            "stat_test_name": "kolmogorov-smirnov",
            "stat_test_score": round(p_val, 6),
            "threshold": alpha,
            "ref_mean": round(float(ref_subset[col].mean()), 4),
            "curr_mean": round(float(curr_subset[col].mean()), 4),
        })

    dataset_drift = (drifted_count / len(ref_subset.columns)) >= 0.5

    result = {
        "metrics": [{
            "metric": "DataDriftTable",
            "result": {
                "dataset_drift": dataset_drift,
                "drift_share": 0.5,
                "drift_by_columns": {
                    f["feature_name"]: {
                        "drift_detected": f["drift_detected"],
                        "stat_test_name": f["stat_test_name"],
                        "drift_score": f["stat_test_score"],
                        "stat_test_threshold": f["threshold"],
                    } for f in features_list
                }
            }
        }],
        "dataset_drift": dataset_drift,
        "drifted_features_count": drifted_count,
        "features": features_list,
    }

    # Write fallback JSON and HTML
    with open(json_output_path, "w") as f:
        json.dump(result, f, indent=4)

    html_content = f"""<!DOCTYPE html>
<html>
<head><title>Evidently Data Drift Report</title><style>body {{ font-family: sans-serif; margin: 2rem; }} table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }} tr:nth-child(even) {{ background-color: #f2f2f2; }} th {{ background-color: #2b5c8f; color: white; }}</style></head>
<body>
<h1>Student Performance: Data Drift Report</h1>
<p><strong>Dataset Drift Status:</strong> {'DRIFT DETECTED' if dataset_drift else 'NO DATASET DRIFT'}</p>
<p><strong>Drifted Features:</strong> {drifted_count} / {len(features_list)}</p>
<table>
<tr><th>Feature</th><th>Drift Detected</th><th>Test</th><th>P-Value / Score</th><th>Threshold</th><th>Ref Mean</th><th>Curr Mean</th></tr>
{"".join(f"<tr><td>{f['feature_name']}</td><td>{'YES' if f['drift_detected'] else 'NO'}</td><td>{f['stat_test_name']}</td><td>{f['stat_test_score']}</td><td>{f['threshold']}</td><td>{f['ref_mean']}</td><td>{f['curr_mean']}</td></tr>" for f in features_list)}
</table>
</body></html>"""
    with open(html_output_path, "w") as f:
        f.write(html_content)

    return result


def monitor_prediction_drift(
    model_path: Path = MODEL_PATH,
    reference_path: Path = REFERENCE_DATA_PATH,
    current_path: Path = CURRENT_DATA_PATH,
    output_plot_path: Path = PREDICTION_COMPARISON_PLOT_PATH,
) -> Dict[str, Any]:
    """
    Optional Prediction Drift Monitoring:
    Applies the trained regression model to both reference and current datasets
    and visualizes shift in predicted student grades (G3).
    """
    if not model_path.exists():
        logger.warning(f"Model path {model_path} not found. Skipping prediction drift monitoring.")
        return {}

    ref_df = pd.read_csv(reference_path)
    curr_df = pd.read_csv(current_path)

    model = joblib.load(model_path)
    ref_preds = model.predict(ref_df)
    curr_preds = model.predict(curr_df)

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=300)
    sns.kdeplot(ref_preds, label="Reference Predicted G3", color="#2980b9", fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(curr_preds, label="Current (Simulated Drift) Predicted G3", color="#d35400", fill=True, alpha=0.35, linewidth=2)

    plt.axvline(ref_preds.mean(), color="#2980b9", linestyle="--", alpha=0.8, label=f"Ref Mean ({ref_preds.mean():.2f})")
    plt.axvline(curr_preds.mean(), color="#d35400", linestyle="--", alpha=0.8, label=f"Curr Mean ({curr_preds.mean():.2f})")

    plt.title("Model Prediction Drift: Reference vs. Current Predicted Mathematics Grades (G3)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Grade (G3)")
    plt.ylabel("Density")
    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    logger.info(f"Prediction drift comparison plot saved to: {output_plot_path}")

    summary = {
        "reference_predictions": {
            "mean": round(float(ref_preds.mean()), 4),
            "median": round(float(np.median(ref_preds)), 4),
            "std": round(float(ref_preds.std()), 4),
        },
        "current_predictions": {
            "mean": round(float(curr_preds.mean()), 4),
            "median": round(float(np.median(curr_preds)), 4),
            "std": round(float(curr_preds.std()), 4),
        },
        "prediction_mean_shift": round(float(curr_preds.mean() - ref_preds.mean()), 4),
    }

    return summary


def run_drift_pipeline() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads reference and current datasets, executes Evidently data drift detection,
    and runs prediction drift analysis.
    """
    ref_df = pd.read_csv(REFERENCE_DATA_PATH)
    curr_df = pd.read_csv(CURRENT_DATA_PATH)

    drift_results = run_evidently_drift_analysis(ref_df, curr_df)
    pred_results = monitor_prediction_drift()

    return drift_results, pred_results


if __name__ == "__main__":
    run_drift_pipeline()
