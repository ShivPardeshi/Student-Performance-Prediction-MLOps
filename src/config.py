"""
Configuration module for the Student Performance Prediction and Data Drift project.
Provides centralized path management, model hyperparameters, and drift monitoring parameters.
"""

from pathlib import Path

# Base project directory (resolved dynamically to ensure portability)
BASE_DIR = Path(__file__).resolve().parent.parent

# Key directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

# Dataset file locations
DATASET_CANDIDATE_PATHS = [
    DATA_DIR / "student-mat.csv",
    BASE_DIR / "student-mat.csv",
    BASE_DIR / "student" / "student-mat.csv",
    DATA_DIR / "student-mat.txt",
]

RAW_DATA_PATH = DATA_DIR / "student-mat.csv"
REFERENCE_DATA_PATH = DATA_DIR / "reference_data.csv"
CURRENT_DATA_PATH = DATA_DIR / "current_data.csv"

# Model artifacts and reports
MODEL_PATH = MODELS_DIR / "student_performance_model.joblib"
MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.json"
ACTUAL_VS_PREDICTED_PLOT_PATH = REPORTS_DIR / "actual_vs_predicted.png"
DATA_DRIFT_REPORT_HTML = REPORTS_DIR / "data_drift_report.html"
DATA_DRIFT_REPORT_JSON = REPORTS_DIR / "data_drift_report.json"
FEATURE_DRIFT_SUMMARY_CSV = REPORTS_DIR / "feature_drift_summary.csv"
PREDICTION_COMPARISON_PLOT_PATH = REPORTS_DIR / "prediction_drift_comparison.png"
PROJECT_SUMMARY_PATH = REPORTS_DIR / "project_summary.json"
DATA_QUALITY_SUMMARY_PATH = REPORTS_DIR / "data_quality_summary.json"

# Machine learning configuration
TARGET_COLUMN = "G3"
EXCLUDE_COLUMNS = ["G1", "G2", "G3"]
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Baseline RandomForest hyperparameters
BASELINE_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "random_state": RANDOM_STATE,
}

# Alternative RandomForest hyperparameters for experiment comparison
RUN2_MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "random_state": RANDOM_STATE,
}

# MLflow tracking
EXPERIMENT_NAME = "Student_Performance_Prediction"
MLFLOW_TRACKING_DIR = BASE_DIR / "mlruns"
MLFLOW_DB_PATH = MLFLOW_TRACKING_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = MLFLOW_TRACKING_DIR / "artifacts"

# Drift simulation and monitoring configuration
REFERENCE_SAMPLE_SIZE = 150
DRIFT_SAMPLE_RANDOM_STATE = 42

# 6 features to explicitly monitor in Evidently drift analysis
MONITORED_FEATURES = [
    "studytime",
    "absences",
    "failures",
    "age",
    "Medu",
    "Fedu",
]
