# Student Performance Prediction and Data Drift Detection Using MLflow and Evidently
**College Assignment Technical Report**

---

## 1. MLflow Installation and Experiment Tracking

### 1.1 Environment Setup
- **Operating System:** macOS (Darwin ARM64 / Apple Silicon)
- **Python Version:** Python 3.12.14 (via isolated project virtual environment `.venv`)
- **MLflow Version:** MLflow 3.17.2
- **Installation and Verification Status:** Successfully installed and verified via automated script execution and programmatic API tracking.

### 1.2 Experiment Tracking
- **Experiment Name:** `Student_Performance_Prediction`
- **Experiment Objective:** Track, evaluate, and version Scikit-learn Random Forest regression model training runs for predicting students' final mathematics grade (`G3`) without using intermediate grades.
- **Parameters Recorded:**
  - `model_type`: `RandomForestRegressor`
  - `n_estimators`: `100` (Run 1), `200` (Run 2)
  - `max_depth`: `8` (Run 1), `12` (Run 2)
  - `random_state`: `42`
  - `test_size`: `0.20`
  - `target_variable`: `G3`
  - `number_of_input_features`: `30`
- **Metrics Recorded:**
  - Mean Absolute Error (`MAE`): `3.0671` (Run 1) vs `2.9819` (Run 2)
  - Root Mean Squared Error (`RMSE`): `3.8280` (Run 1) vs `3.7570` (Run 2)
  - Coefficient of Determination (`R²`): `0.2854` (Run 1) vs `0.3116` (Run 2)
- **Artifacts Recorded:**
  - Serialized Scikit-learn Pipeline (`models/student_performance_model.joblib`)
  - Actual-versus-Predicted scatter plot with diagonal line (`reports/actual_vs_predicted.png`)
  - Run-specific diagnostic plots (`reports/baseline_model_run1_actual_vs_predicted.png`, `reports/tuned_model_run2_actual_vs_predicted.png`)
  - Data quality validation summary (`reports/data_quality_summary.json`)
  - Comprehensive model metrics JSON (`reports/model_metrics.json`)
- **Execution Summary:** Two reproducible training runs were logged using local file store tracking (`file://.../mlruns`). Tuned Model Run 2 achieved a lower RMSE (`3.7570`) and higher R² (`0.3116`) compared to the Baseline Run (`3.8280` RMSE, `0.2854` R²) and was selected as the final model.

---

## 2. Dataset Identification

### 2.1 Problem Statement
In educational institutions, early prediction of students at academic risk enables counselors and educators to design timely intervention programs. The objective of this project is to develop a supervised regression model predicting a student's final mathematics exam grade (`G3`, 0–20 scale) based exclusively on demographic traits, family background, and study habits. Earlier grades (`G1` and `G2`) are excluded to avoid data leakage and evaluate whether non-academic background characteristics alone carry sufficient predictive signal.

### 2.2 Dataset Details
- **Dataset Name:** Student Performance Data Set (Mathematics: `student-mat.csv`)
- **Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez and Silva, 2008)
- **Number of Records:** 395 student observations
- **Number of Columns:** 33 total attributes (30 input features, 2 intermediate exam grades, 1 final target grade)
- **Important Features:**
  - `studytime`: Weekly study time (1: <2 hours, 2: 2–5 hours, 3: 5–10 hours, 4: >10 hours)
  - `absences`: Number of school absences (0 to 93)
  - `failures`: Number of past class failures (0 to 4)
  - `age`: Student age (15 to 22)
  - `Medu` / `Fedu`: Mother and father education level (0: none to 4: higher education)
  - `higher`: Plans for higher education (yes/no)
  - `internet`: Home internet access (yes/no)
- **Target Variable:** `G3` (Final Mathematics Grade, integer from 0 to 20, mean = 10.42)
- **Use Case Category:** Supervised Machine Learning — Regression / Educational Data Science

### 2.3 Dataset Selection Rationale
The UCI Student Performance dataset is an established academic benchmark. It provides realistic survey and administrative data across diverse demographic, behavioral, and academic dimensions. It is ideally suited for demonstrating regression modeling and data drift detection due to its clean schema and mix of continuous and categorical features.

---

## 3. Drift Scenario and Detection Method

### 3.1 Drift Type
- **Primary Drift Type:** Covariate / Feature Data Drift (change in feature distributions $P(X)$).
- **Secondary Drift Type:** Prediction Drift (change in model output distribution $P(\hat{Y})$).
- **Simulated Nature:** The drift scenario is deliberately simulated in a controlled environment to test detection sensitivity and isolate the effects of feature modifications.

### 3.2 Reference and Current Data
- **Reference Dataset (`data/reference_data.csv`):** 150 student records sampled reproducibly (`random_state=42`) from the original dataset, retaining all 30 model input features (excluding `G1`, `G2`, `G3`).
- **Current Dataset (`data/current_data.csv`):** A copy of the reference dataset with three deterministic modifications:
  1. `studytime`: Categories 1 and 2 were shifted to 4 (simulating artificial inflation in reported study hours).
  2. `absences`: Uniformly incremented by +5 for every student (simulating a synthetic post-holiday absenteeism surge).
  3. `failures`: Incremented by +1, capped at maximum 4 (simulating stricter prior academic standards).

### 3.3 Detection Method
- **Tool Used:** Evidently (`Report` and `DataDriftPreset`).
- **Statistical Tests:**
  - Continuous / Numerical features: Two-Sample Kolmogorov-Smirnov (K-S) test.
  - Significance Level ($\alpha$): $0.05$ (drift flagged when $p$-value $< 0.05$).
  - Dataset-Level Drift Threshold: Triggered if $\ge 50\%$ of monitored features exhibit statistically significant drift.
- **Features Analyzed:** 6 key features (`studytime`, `absences`, `failures`, `age`, `Medu`, `Fedu`).

---

## 4. Results, Interpretation and Conclusion

### 4.1 Drift Analysis Results
Evidently generated both an interactive HTML report (`reports/data_drift_report.html`) and structured JSON (`reports/data_drift_report.json`).

| Feature Name | Drift Detected? | Statistical Test | p-value / Score | Threshold | Reference Mean | Current Mean |
|--------------|-----------------|------------------|-----------------|-----------|----------------|--------------|
| `studytime`  | **YES**         | Kolmogorov-Smirnov | `0.0000`      | `0.0500`  | 1.97           | 3.84         |
| `absences`   | **YES**         | Kolmogorov-Smirnov | `6.53e-28`    | `0.0500`  | 4.93           | 9.93         |
| `failures`   | **YES**         | Kolmogorov-Smirnov | `0.0000`      | `0.0500`  | 0.33           | 1.33         |
| `age`        | **NO**          | Kolmogorov-Smirnov | `1.0000`      | `0.0500`  | 16.65          | 16.65        |
| `Medu`       | **NO**          | Kolmogorov-Smirnov | `1.0000`      | `0.0500`  | 2.77           | 2.77         |
| `Fedu`       | **NO**          | Kolmogorov-Smirnov | `1.0000`      | `0.0500`  | 2.59           | 2.59         |

- **Dataset-Level Drift:** **DETECTED** (3 out of 6 features drifted = 50.0% drift share).
- **Prediction Drift Analysis:**
  - Reference Predicted Grade Mean: `10.38`
  - Current (Shifted) Predicted Grade Mean: `10.22`
  - Prediction Mean Shift: `-0.15` grade points

### 4.2 Interpretation
The monitoring pipeline accurately identified the three features that underwent deliberate modification (`studytime`, `absences`, `failures`) with extreme statistical significance ($p < 10^{-4}$). Concurrently, unchanged control features (`age`, `Medu`, `Fedu`) returned $p = 1.0000$, confirming zero false positives. In a live academic deployment, an unmonitored surge in absence rates or prior failures would cause model predictions to systematically degrade, risking misallocation of student counseling resources.

### 4.3 Conclusion and Recommended Action
- **Conclusion:** Statistically significant data drift was detected.
- **Recommended Actions:**
  1. Automated alerting: Trigger high-priority notifications to MLOps engineers and academic administrators.
  2. Root cause audit: Verify whether absence spikes are genuine or caused by attendance sensor/integration errors.
  3. Continuous retraining: Ingest newly verified exam results at the end of the term and trigger model retraining using the MLflow automated pipeline.

---

## 5. GitHub Repository and Reproducibility

- **GitHub Repository URL:** Prepared locally (Remote URL pending publication upon instructor review).
- **Source Code & Notebook:**
  - Configuration: `src/config.py`
  - Data Loader: `src/data_loader.py`
  - Training & Tracking: `src/train.py`
  - Evaluation & Plotting: `src/evaluate.py`
  - Drift Data Preparation: `src/prepare_data.py`
  - Drift Detection: `src/detect_drift.py`
  - Summary Generator: `src/generate_summary.py`
  - Interactive Notebook: `notebooks/student_performance_analysis.ipynb`
  - Automated Tests: `tests/test_data.py`, `tests/test_model.py`, `tests/test_drift.py`
- **Reproducibility Instructions:**
  ```bash
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  python -m src.data_loader
  python -m src.train
  python -m src.prepare_data
  python -m src.detect_drift
  python -m src.generate_summary
  pytest -v
  ```
- **Requirements & Dependencies:** Pinned in `requirements.txt`.
