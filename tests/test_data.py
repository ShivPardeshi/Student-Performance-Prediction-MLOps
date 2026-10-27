"""
Automated unit tests for dataset loading, schema validation, and drift data generation.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.config import (
    CURRENT_DATA_PATH,
    EXCLUDE_COLUMNS,
    REFERENCE_DATA_PATH,
    TARGET_COLUMN,
)
from src.data_loader import ensure_dataset_in_data_dir, load_and_validate_data
from src.prepare_data import generate_current_dataset_with_drift, generate_reference_dataset


def test_dataset_loading_and_target_exists():
    """Verify that student-mat.csv is located, loaded, and contains target G3."""
    df, quality = load_and_validate_data()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 300, f"Expected at least 300 records, got {len(df)}"
    assert TARGET_COLUMN in df.columns, f"Target column '{TARGET_COLUMN}' must exist"
    assert quality["num_records"] == len(df)
    assert quality["num_columns"] == len(df.columns)


def test_feature_exclusion_from_inputs():
    """Verify G1, G2, and G3 are strictly excluded from model input features."""
    df, _ = load_and_validate_data()
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]

    assert "G1" not in feature_cols, "G1 must be excluded from input features"
    assert "G2" not in feature_cols, "G2 must be excluded from input features"
    assert "G3" not in feature_cols, "G3 (target) must be excluded from input features"
    assert len(feature_cols) == len(df.columns) - 3


def test_reference_and_current_datasets_schema_match():
    """Verify reference and current datasets share identical schemas, column orders, and row counts."""
    df, _ = load_and_validate_data()
    ref_df = generate_reference_dataset(df, sample_size=150, random_state=42)
    curr_df, summary = generate_current_dataset_with_drift(ref_df)

    assert len(ref_df) == 150
    assert len(curr_df) == 150
    assert list(ref_df.columns) == list(curr_df.columns), "Schemas and column ordering must match"
    assert TARGET_COLUMN not in ref_df.columns, "Target G3 should not be in reference features"
    assert TARGET_COLUMN not in curr_df.columns, "Target G3 should not be in current features"


def test_simulated_drift_integrity():
    """Verify simulated transformations applied to studytime, absences, and failures without NaNs."""
    df, _ = load_and_validate_data()
    ref_df = generate_reference_dataset(df, sample_size=150, random_state=42)
    curr_df, _ = generate_current_dataset_with_drift(ref_df)

    # Check that no NaNs were introduced
    assert curr_df.isnull().sum().sum() == 0, "Transformations introduced unexpected missing values"

    # Check studytime values are within valid bounds (1 to 4)
    assert curr_df["studytime"].isin([1, 2, 3, 4]).all(), "studytime contains values out of range [1-4]"

    # Check absences increased by 5
    assert (curr_df["absences"] == ref_df["absences"] + 5).all(), "absences should be shifted uniformly by +5"

    # Check failures are capped at 4
    assert (curr_df["failures"] <= 4).all(), "failures must be capped at 4"
