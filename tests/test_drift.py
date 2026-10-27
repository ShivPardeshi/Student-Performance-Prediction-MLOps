"""
Automated unit tests for Evidently data drift detection and output artifact validation.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.config import (
    DATA_DRIFT_REPORT_HTML,
    DATA_DRIFT_REPORT_JSON,
    FEATURE_DRIFT_SUMMARY_CSV,
    MONITORED_FEATURES,
)
from src.detect_drift import run_evidently_drift_analysis


def test_drift_analysis_execution_and_reports(tmp_path):
    """Verify drift analysis generates valid HTML, JSON, and summary CSV reports."""
    # Create small dummy datasets with simulated drift
    ref_df = pd.DataFrame({
        "studytime": [1, 2, 2, 3, 1, 4] * 5,
        "absences": [2, 4, 0, 5, 1, 3] * 5,
        "failures": [0, 0, 1, 0, 0, 0] * 5,
        "age": [15, 16, 17, 15, 16, 18] * 5,
        "Medu": [4, 3, 2, 4, 1, 3] * 5,
        "Fedu": [3, 2, 2, 4, 2, 3] * 5,
    })

    curr_df = ref_df.copy()
    curr_df["studytime"] = 4  # Drastic drift
    curr_df["absences"] = curr_df["absences"] + 15  # Drastic drift

    test_html = tmp_path / "test_report.html"
    test_json = tmp_path / "test_report.json"
    test_csv = tmp_path / "test_summary.csv"

    drift_summary = run_evidently_drift_analysis(
        reference_df=ref_df,
        current_df=curr_df,
        features_to_monitor=MONITORED_FEATURES,
        html_output_path=test_html,
        json_output_path=test_json,
        summary_csv_path=test_csv,
    )

    # 1. HTML verification
    assert test_html.exists()
    assert test_html.stat().st_size > 100

    # 2. JSON verification
    assert test_json.exists()
    with open(test_json, "r") as f:
        data = json.load(f)
        assert isinstance(data, dict)

    # 3. CSV summary verification
    assert test_csv.exists()
    summary_df = pd.read_csv(test_csv)
    assert "feature_name" in summary_df.columns
    assert "drift_detected" in summary_df.columns
    assert len(summary_df) == len(MONITORED_FEATURES)

    # 4. Check that drift detection flagged changes without fabricating
    assert isinstance(drift_summary["dataset_drift"], bool)
    assert drift_summary["drifted_features_count"] >= 1
