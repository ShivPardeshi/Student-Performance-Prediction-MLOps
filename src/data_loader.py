"""
Data loader and validation module for the Student Performance dataset.
Locates the raw dataset automatically, validates its schema and integrity,
and generates a data quality summary report.
"""

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.config import (
    DATA_DIR,
    DATA_QUALITY_SUMMARY_PATH,
    DATASET_CANDIDATE_PATHS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_MIN_COLUMNS = 30
EXPECTED_COLUMNS_COUNT = 33
EXPECTED_APPROX_ROWS = 395


def find_raw_dataset() -> Optional[Path]:
    """
    Search for student-mat.csv across known locations, root, subdirectories,
    and inspect any zip archives if present.
    """
    # 1. Check direct candidate paths
    for candidate in DATASET_CANDIDATE_PATHS:
        if candidate.is_file() and candidate.stat().st_size > 0:
            logger.info(f"Located dataset at: {candidate}")
            return candidate

    # 2. Check for zip files in data/ or project root
    search_dirs = [DATA_DIR, DATA_DIR.parent]
    for directory in search_dirs:
        for zip_file in directory.glob("*.zip"):
            try:
                with zipfile.ZipFile(zip_file, "r") as z:
                    for name in z.namelist():
                        if "student-mat.csv" in name:
                            logger.info(f"Extracting student-mat.csv from zip archive: {zip_file}")
                            DATA_DIR.mkdir(parents=True, exist_ok=True)
                            z.extract(name, path=DATA_DIR)
                            extracted = DATA_DIR / name
                            if extracted.name != "student-mat.csv":
                                target = DATA_DIR / "student-mat.csv"
                                shutil.move(str(extracted), str(target))
                                return target
                            return extracted
            except Exception as e:
                logger.warning(f"Failed to inspect zip file {zip_file}: {e}")

    # 3. Recursive search in workspace (excluding virtual environments and caches)
    for p in DATA_DIR.parent.glob("**/student-mat.csv"):
        if ".venv" not in p.parts and "venv" not in p.parts:
            logger.info(f"Located dataset via recursive search at: {p}")
            return p

    return None


def ensure_dataset_in_data_dir() -> Path:
    """
    Ensures that student-mat.csv is present in the data/ directory.
    If found elsewhere, copies it without modifying the original.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.is_file() and RAW_DATA_PATH.stat().st_size > 0:
        return RAW_DATA_PATH

    found_path = find_raw_dataset()
    if found_path is None:
        raise FileNotFoundError(
            f"Dataset 'student-mat.csv' could not be found.\n"
            f"Please place 'student-mat.csv' in either:\n"
            f"  1. {RAW_DATA_PATH}\n"
            f"  2. {DATA_DIR.parent / 'student-mat.csv'}\n"
            f"Download from: https://archive.ics.uci.edu/dataset/320/student+performance"
        )

    if found_path.resolve() != RAW_DATA_PATH.resolve():
        logger.info(f"Copying located dataset from {found_path} to {RAW_DATA_PATH}")
        shutil.copy2(str(found_path), str(RAW_DATA_PATH))

    return RAW_DATA_PATH


def load_and_validate_data(file_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Loads student-mat.csv using semicolon delimiter and validates its integrity.
    Returns the DataFrame and a data quality summary dictionary.
    """
    target_path = file_path if file_path else ensure_dataset_in_data_dir()

    if not target_path.exists():
        raise FileNotFoundError(f"File not found: {target_path}")

    # Try reading with semicolon delimiter (standard for UCI Student dataset)
    try:
        df = pd.read_csv(target_path, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed, retrying with latin1 encoding.")
        df = pd.read_csv(target_path, sep=";", encoding="latin1")

    # If semicolon failed (e.g. single column read), fallback to comma check
    if len(df.columns) == 1 and "," in str(df.columns[0]):
        logger.warning("Dataset appears comma-separated rather than semicolon-separated. Reloading with comma.")
        df = pd.read_csv(target_path, sep=",")

    # Basic validations
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' not found in dataset columns: {list(df.columns)}"
        )

    num_rows, num_cols = df.shape
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    # Check expected characteristics
    if num_rows < 350 or num_rows > 450:
        logger.warning(
            f"Row count ({num_rows}) differs from typical UCI student-mat count (~{EXPECTED_APPROX_ROWS})."
        )
    if num_cols < EXPECTED_MIN_COLUMNS:
        logger.warning(
            f"Column count ({num_cols}) is fewer than expected ({EXPECTED_COLUMNS_COUNT})."
        )

    # Print summary
    print("=" * 60)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 60)
    print(f"File path:         {target_path}")
    print(f"Dataset shape:     {df.shape} ({num_rows} records, {num_cols} columns)")
    print(f"Target column:     {TARGET_COLUMN} (min: {df[TARGET_COLUMN].min()}, max: {df[TARGET_COLUMN].max()}, mean: {df[TARGET_COLUMN].mean():.2f})")
    print(f"Missing values:    {missing_count}")
    print(f"Duplicate rows:    {duplicate_count}")
    print(f"Numerical columns: {len(df.select_dtypes(include=['number']).columns)}")
    print(f"Categorical cols:  {len(df.select_dtypes(exclude=['number']).columns)}")
    print("=" * 60)

    quality_summary = {
        "dataset_path": str(target_path.resolve()),
        "num_records": num_rows,
        "num_columns": num_cols,
        "columns": list(df.columns),
        "target_column": TARGET_COLUMN,
        "target_stats": {
            "min": float(df[TARGET_COLUMN].min()),
            "max": float(df[TARGET_COLUMN].max()),
            "mean": float(df[TARGET_COLUMN].mean()),
            "median": float(df[TARGET_COLUMN].median()),
            "std": float(df[TARGET_COLUMN].std()),
        },
        "missing_values_total": missing_count,
        "missing_values_by_column": {k: int(v) for k, v in df.isnull().sum().items() if v > 0},
        "duplicate_rows": duplicate_count,
        "numerical_columns": list(df.select_dtypes(include=["number"]).columns),
        "categorical_columns": list(df.select_dtypes(exclude=["number"]).columns),
    }

    # Save data quality summary
    DATA_QUALITY_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_QUALITY_SUMMARY_PATH, "w") as f:
        json.dump(quality_summary, f, indent=4)
    logger.info(f"Saved data quality summary to {DATA_QUALITY_SUMMARY_PATH}")

    return df, quality_summary


if __name__ == "__main__":
    df, summary = load_and_validate_data()
    print("Columns loaded:")
    print(df.columns.tolist())
    print("\nFirst 3 records:")
    print(df.head(3))
