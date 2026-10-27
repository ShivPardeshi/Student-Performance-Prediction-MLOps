"""
Data preparation module for drift monitoring.
Generates reference and current datasets with controlled, reproducible,
simulated data drift for educational MLOps demonstration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    CURRENT_DATA_PATH,
    DRIFT_SAMPLE_RANDOM_STATE,
    EXCLUDE_COLUMNS,
    MONITORED_FEATURES,
    REFERENCE_DATA_PATH,
    REFERENCE_SAMPLE_SIZE,
    REPORTS_DIR,
)
from src.data_loader import ensure_dataset_in_data_dir, load_and_validate_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_reference_dataset(
    df: pd.DataFrame,
    sample_size: int = REFERENCE_SAMPLE_SIZE,
    random_state: int = DRIFT_SAMPLE_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Samples a fixed subset of records and excludes target & intermediate grade columns
    (G1, G2, G3) to build a clean baseline reference dataset.
    """
    # Exclude G1, G2, G3 from drift monitoring feature set
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    df_features = df[feature_cols].copy()

    # Reproducible sampling
    if len(df_features) > sample_size:
        reference_df = df_features.sample(n=sample_size, random_state=random_state).copy()
    else:
        reference_df = df_features.copy()

    reference_df = reference_df.reset_index(drop=True)
    logger.info(f"Generated reference dataset with shape: {reference_df.shape}")
    return reference_df


def generate_current_dataset_with_drift(reference_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Creates current_data.csv by copying the reference dataset and applying
    three deterministic, controlled transformations to simulate data drift:
    1. studytime: Lower study categories (1 and 2) shifted to category 4.
    2. absences: Added 5 absences to every student.
    3. failures: Increased failures by 1, capped at 4.
    """
    current_df = reference_df.copy()

    # 1. Transform studytime: 1 and 2 become 4; 3 and 4 remain unchanged
    original_studytime = current_df["studytime"].copy()
    current_df["studytime"] = current_df["studytime"].apply(lambda x: 4 if x in [1, 2] else x)

    # 2. Transform absences: Increase by 5
    original_absences = current_df["absences"].copy()
    current_df["absences"] = current_df["absences"] + 5

    # 3. Transform failures: Increase by 1, capped at max 4
    original_failures = current_df["failures"].copy()
    current_df["failures"] = (current_df["failures"] + 1).clip(upper=4)

    # Verification: schemas, columns, and lengths match
    assert list(reference_df.columns) == list(current_df.columns), "Columns do not match!"
    assert len(reference_df) == len(current_df), "Row counts do not match!"

    summary_stats = {
        "num_records": len(current_df),
        "num_features": len(current_df.columns),
        "transformations": {
            "studytime": {
                "description": "Categories 1 and 2 shifted to 4; categories 3 and 4 kept unchanged.",
                "ref_mean": float(reference_df["studytime"].mean()),
                "curr_mean": float(current_df["studytime"].mean()),
                "ref_distribution": reference_df["studytime"].value_counts().to_dict(),
                "curr_distribution": current_df["studytime"].value_counts().to_dict(),
            },
            "absences": {
                "description": "Absences uniformly increased by +5 for each record.",
                "ref_mean": float(reference_df["absences"].mean()),
                "curr_mean": float(current_df["absences"].mean()),
                "ref_median": float(reference_df["absences"].median()),
                "curr_median": float(current_df["absences"].median()),
            },
            "failures": {
                "description": "Past class failures incremented by +1, capped at 4.",
                "ref_mean": float(reference_df["failures"].mean()),
                "curr_mean": float(current_df["failures"].mean()),
                "ref_distribution": reference_df["failures"].value_counts().to_dict(),
                "curr_distribution": current_df["failures"].value_counts().to_dict(),
            },
        },
    }

    return current_df, summary_stats


def plot_drift_comparisons(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Path = REPORTS_DIR / "simulated_drift_distributions.png",
) -> Path:
    """
    Visualizes distribution changes across the modified features (studytime, absences, failures)
    and unmodified reference features (age, Medu, Fedu).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_to_plot = ["studytime", "absences", "failures", "age"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()

    for idx, col in enumerate(features_to_plot):
        ax = axes[idx]
        ref_vals = reference_df[col]
        curr_vals = current_df[col]

        sns.kdeplot(ref_vals, ax=ax, label="Reference (Baseline)", color="#2980b9", fill=True, alpha=0.3)
        sns.kdeplot(curr_vals, ax=ax, label="Current (Simulated)", color="#e67e22", fill=True, alpha=0.3)

        drift_flag = " (Drift Injected)" if col in ["studytime", "absences", "failures"] else " (Unchanged Baseline)"
        ax.set_title(f"Distribution: {col}{drift_flag}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Simulated Data Drift: Reference vs. Current Distributions", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Drift comparison plot saved to: {output_path}")
    return output_path


def prepare_and_save_drift_datasets(
    reference_path: Path = REFERENCE_DATA_PATH,
    current_path: Path = CURRENT_DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Orchestrates reference dataset creation and current dataset generation with simulated drift.
    Saves CSV files to data/ and comparison metadata to reports/.
    """
    df, _ = load_and_validate_data()

    reference_df = generate_reference_dataset(df)
    current_df, drift_summary = generate_current_dataset_with_drift(reference_df)

    # Save datasets
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_df.to_csv(reference_path, index=False)
    current_df.to_csv(current_path, index=False)
    logger.info(f"Saved reference dataset to: {reference_path}")
    logger.info(f"Saved current dataset to:   {current_path}")

    # Plot distributions
    plot_drift_comparisons(reference_df, current_df)

    # Save summary json
    summary_path = REPORTS_DIR / "simulated_drift_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(drift_summary, f, indent=4)
    logger.info(f"Saved drift comparison summary to: {summary_path}")

    print("=" * 60)
    print("REFERENCE AND CURRENT DATASETS PREPARED")
    print("=" * 60)
    print(f"Reference dataset: {reference_path} ({len(reference_df)} rows, {len(reference_df.columns)} cols)")
    print(f"Current dataset:   {current_path} ({len(current_df)} rows, {len(current_df.columns)} cols)")
    print("Features modified: studytime, absences, failures")
    print("Features excluded: G1, G2, G3 (target and prior exam grades)")
    print("=" * 60)

    return reference_df, current_df, drift_summary


if __name__ == "__main__":
    prepare_and_save_drift_datasets()
