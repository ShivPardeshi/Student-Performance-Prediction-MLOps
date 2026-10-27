# Student Performance Prediction and Data Drift Detection Using MLflow and Evidently

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLflow](https://img.shields.io/badge/MLflow-3.17.2-0194E2.svg)](https://mlflow.org/)
[![Evidently](https://img.shields.io/badge/Evidently-0.7.23-red.svg)](https://www.evidentlyai.com/)
[![Tests](https://img.shields.io/badge/pytest-8%20passed-brightgreen.svg)](https://docs.pytest.org/)

An end-to-end, reproducible, beginner-friendly MLOps project demonstrating supervised machine learning regression, local experiment tracking with **MLflow**, and continuous distribution monitoring with **Evidently**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement & Objectives](#problem-statement--objectives)
3. [Dataset Details & Placement](#dataset-details--placement)
4. [Technology Stack](#technology-stack)
5. [Project Architecture & Directory Structure](#project-architecture--directory-structure)
6. [Installation & Environment Setup](#installation--environment-setup)
7. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
8. [MLflow Experiment Tracking & UI](#mlflow-experiment-tracking--ui)
9. [Simulated Data Drift & Evidently Detection](#simulated-data-drift--evidently-detection)
10. [Automated Testing](#automated-testing)
11. [Actual Experimental Results](#actual-experimental-results)
12. [Generated Artifacts & Reports](#generated-artifacts--reports)
13. [Demonstration Evidence & Screenshots](#demonstration-evidence--screenshots)
14. [Limitations & Future Improvements](#limitations--future-improvements)
15. [License & Attribution](#license--attribution)

---

## Project Overview

In real-world machine learning deployments, model performance degrades over time not because the code changes, but because **data changes**. This phenomenon is known as **data drift**.

This educational project establishes a complete machine learning and MLOps pipeline that:
- Trains a Scikit-learn Random Forest regression model to predict student mathematics exam grades (`G3`).
- Intentionally isolates demographic, family background, and study habits by **excluding previous period grades (`G1`, `G2`)**.
- Uses **MLflow** for local experiment tracking, comparing multiple model architectures, logging parameters, metrics, plots, and models.
- Prepares a baseline **Reference Dataset** and introduces deterministic, controlled modifications to generate a **Current Dataset** with simulated drift.
- Employs **Evidently** to evaluate statistical drift across key features, generating interactive HTML reports and structured JSON summaries.
- Provides comprehensive unit test suites using **pytest**.

---

## Problem Statement & Objectives

### Problem Statement
Educational institutions often struggle to identify students at risk of academic failure early in the semester. Standard evaluation approaches rely heavily on midterm exam scores (`G1` and `G2`). However, true early-warning systems must detect risk factors *before* any test is administered, using student background, family context, and study habits.

### Objectives
1. **Develop a Machine Learning Pipeline:** Train a supervised regression pipeline with dynamic imputation and one-hot encoding.
2. **Track Experiments with MLflow:** Record at least two model configurations (Baseline vs. Comparative Run), log parameters, metrics, and serialized artifacts.
3. **Simulate Controlled Drift:** Apply deterministic shifts to `studytime`, `absences`, and `failures`.
4. **Detect Drift with Evidently:** Conduct Two-Sample Kolmogorov-Smirnov hypothesis tests, compute p-values, and generate visual HTML dashboards.
5. **Demonstrate Prediction Drift:** Analyze how shifts in input distributions affect model output predictions.
6. **Ensure Strict Reproducibility:** Ensure the entire pipeline runs locally on any laptop with fixed random seeds.

---

## Dataset Details & Placement

- **Name:** Student Performance Data Set (Mathematics)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Paper Citation:** Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance*.
- **Records:** 395 students
- **Total Columns:** 33 columns (30 input features, 2 intermediate exam grades `G1` and `G2`, 1 final grade `G3`)
- **Target Variable:** `G3` (Final Mathematics Grade, integer scale 0–20, mean = 10.42)

### Automatic Dataset Discovery
The project's dataset loader (`src/data_loader.py`) locates `student-mat.csv` automatically in any of the following locations:
1. `data/student-mat.csv` (Default recommended location)
2. `student-mat.csv` (Project root directory)
3. `student/student-mat.csv` (Extracted subfolder)
4. Any `.zip` archive containing `student-mat.csv` placed in the project folder.

If found elsewhere, the loader automatically copies it into `data/student-mat.csv` without modifying the original source file.

---

## Technology Stack

- **Language:** Python 3.12
- **Data Manipulation:** Pandas 2.2.3, NumPy 2.5.4
- **Machine Learning:** Scikit-learn 1.9.1, Joblib 1.6.0
- **Experiment Tracking:** MLflow 3.17.2
- **Drift Detection:** Evidently 0.7.23, SciPy 1.18.1
- **Visualization:** Matplotlib 3.11.2, Seaborn 0.13.2
- **Testing:** Pytest 9.1.1
- **Version Control:** Git

---

## Project Architecture & Directory Structure

```
MLOps Student Project/
│
├── data/
│   ├── .gitkeep                      # Git folder placeholder
│   ├── student-mat.csv               # Raw dataset (semicolon-separated)
│   ├── reference_data.csv            # 150 sampled baseline records (features only)
│   └── current_data.csv              # 150 records with simulated drift
│
├── models/
│   └── student_performance_model.joblib  # Serialized Scikit-learn pipeline
│
├── notebooks/
│   └── student_performance_analysis.ipynb # 15-section walkthrough notebook
│
├── reports/
│   ├── actual_vs_predicted.png       # Publication-quality regression plot
│   ├── baseline_model_run1_actual_vs_predicted.png
│   ├── tuned_model_run2_actual_vs_predicted.png
│   ├── model_metrics.json            # Run comparison & evaluation metrics
│   ├── data_quality_summary.json     # Initial dataset validation report
│   ├── simulated_drift_summary.json  # Drift parameter shifts summary
│   ├── simulated_drift_distributions.png # Distribution comparison KDE plot
│   ├── data_drift_report.html        # Interactive Evidently HTML report
│   ├── data_drift_report.json        # Evidently statistical test outputs
│   ├── feature_drift_summary.csv     # Tabular p-values & drift flags
│   ├── prediction_drift_comparison.png # Reference vs. Current predicted G3
│   ├── project_summary.json          # Unified end-to-end project summary
│   └── assignment_report.md          # College assignment documentation
│
├── screenshots/
│   └── README.md                     # Evidence capture checklist
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Centralized paths, seeds, and constants
│   ├── data_loader.py                # Dataset locator & schema validation
│   ├── train.py                      # Preprocessing pipeline, model training, MLflow
│   ├── evaluate.py                   # Metrics (MAE, RMSE, R²) and visualization
│   ├── prepare_data.py               # Reference & drift dataset synthesis
│   ├── detect_drift.py               # Evidently report generation & prediction drift
│   └── generate_summary.py           # Project summary compiler
│
├── tests/
│   ├── test_data.py                  # Schema & data integrity tests
│   ├── test_model.py                 # Training & inference tests
│   └── test_drift.py                 # Report generation & drift tests
│
├── .gitignore                        # Standard exclusions (venv, mlruns, caches)
├── LICENSE                           # MIT License
├── pytest.ini                        # Pytest configuration
├── requirements.txt                  # Pinned dependencies
└── README.md                         # Comprehensive documentation
```

---

## Installation & Environment Setup

### 1. Clone or Open the Repository
```bash
git clone <repository_url>
cd "MLOps Student Project"
```

### 2. Set Up a Virtual Environment

**On macOS / Linux:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Step-by-Step Execution Guide

Run the pipeline sequentially using the standard Python module syntax:

### Step 1: Validate Dataset
```bash
python -m src.data_loader
```
- Locates `student-mat.csv`.
- Validates 395 rows, 33 columns, 0 missing values, and confirmed target `G3`.
- Saves `reports/data_quality_summary.json`.

### Step 2: Train Models & Track Experiments with MLflow
```bash
python -m src.train
```
- Trains **Run 1 (Baseline)**: `n_estimators=100`, `max_depth=8`.
- Trains **Run 2 (Comparative)**: `n_estimators=200`, `max_depth=12`.
- Logs parameters, metrics, plots, and models to MLflow (`mlruns/`).
- Evaluates test set performance and selects the best model.
- Saves `models/student_performance_model.joblib` and `reports/actual_vs_predicted.png`.

### Step 3: Prepare Reference & Current Datasets
```bash
python -m src.prepare_data
```
- Extracts 150 sampled records for `data/reference_data.csv` (excluding `G1`, `G2`, `G3`).
- Applies 3 deterministic transformations to create `data/current_data.csv`.
- Saves distribution comparison plots to `reports/simulated_drift_distributions.png`.

### Step 4: Detect Data Drift Using Evidently
```bash
python -m src.detect_drift
```
- Executes Evidently `DataDriftPreset` across 6 key features.
- Generates interactive HTML dashboard `reports/data_drift_report.html`.
- Exports machine-readable `reports/data_drift_report.json` and `reports/feature_drift_summary.csv`.
- Analyzes prediction drift on model outputs (`reports/prediction_drift_comparison.png`).

### Step 5: Compile Project Summary
```bash
python -m src.generate_summary
```
- Reads actual execution results and outputs `reports/project_summary.json`.
- Displays a formatted console summary of all metrics and artifacts.

### Step 6: Run Automated Tests
```bash
pytest -v
```
- Executes all 8 unit tests in `tests/`.

---

## MLflow Experiment Tracking & UI

MLflow tracks all hyperparameters, metrics, and models locally without requiring a remote server or cloud account.

### Launch the MLflow UI
Run the following command from the project root:
```bash
mlflow ui
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Navigating the MLflow UI
1. Click on the **`Student_Performance_Prediction`** experiment in the left sidebar.
2. Select both runs: `Baseline_Model_Run1` and `Tuned_Model_Run2`.
3. Click the **Compare** button to view side-by-side metric tables (MAE, RMSE, R²).
4. Click on any run to inspect logged artifacts:
   - `model/` (Serialized MLmodel & pickle pipeline)
   - `actual_vs_predicted.png`
   - `data_quality_summary.json`

---

## Simulated Data Drift & Evidently Detection

### The Controlled Simulation
To objectively test the monitoring pipeline without confounding variables, three deterministic modifications were applied to a reproducible sample of 150 records:
1. **`studytime` (Weekly Study Hours):** Lower categories (1 and 2) were remapped to 4, simulating an artificial inflation in self-reported study habits.
2. **`absences` (School Absences):** Uniformly increased by +5 for all students, simulating an attendance shock.
3. **`failures` (Past Class Failures):** Incremented by +1 (capped at 4), simulating a stricter academic policy.
4. **Control Features (`age`, `Medu`, `Fedu`):** Left completely unmodified to verify that the monitoring tool does not produce false positives.

### Evidently Monitoring Configuration
- **Monitored Features:** `studytime`, `absences`, `failures`, `age`, `Medu`, `Fedu`.
- **Target Exclusion:** `G3` is strictly excluded from feature drift monitoring.
- **Hypothesis Test:** Two-Sample Kolmogorov-Smirnov (K-S) test.
- **Threshold ($\alpha$):** 0.05.
- **Dataset-Level Drift Condition:** $\ge 50\%$ drifted features.

---

## Automated Testing

The project includes an automated test suite implemented with `pytest` in `tests/`:

```bash
pytest -v
```

### Test Coverage
- **`tests/test_data.py`:**
  - `test_dataset_loading_and_target_exists`: Confirms `student-mat.csv` loads with 395 rows and target `G3`.
  - `test_feature_exclusion_from_inputs`: Confirms `G1`, `G2`, and `G3` are excluded from model training inputs.
  - `test_reference_and_current_datasets_schema_match`: Asserts schemas, columns, and 150 sample rows match.
  - `test_simulated_drift_integrity`: Confirms transformations introduce zero missing values and respect bounds.
- **`tests/test_model.py`:**
  - `test_metrics_computation_validity`: Validates mathematical boundaries of MAE, RMSE, and R².
  - `test_model_pipeline_training_and_prediction`: Asserts the pipeline trains and outputs finite 1D arrays.
  - `test_saved_model_load_and_inference`: Confirms the saved `.joblib` model can be reloaded for inference.
- **`tests/test_drift.py`:**
  - `test_drift_analysis_execution_and_reports`: Verifies generation and schema of HTML, JSON, and CSV reports.

---

## Actual Experimental Results

*(All metrics reflect real execution on the dataset)*

### 1. Model Training & Comparison

| Metric / Parameter | Run 1 (Baseline) | Run 2 (Comparative) | Winner / Selected |
|--------------------|------------------|---------------------|-------------------|
| `n_estimators`     | 100              | 200                 | -                 |
| `max_depth`        | 8                | 12                  | -                 |
| **MAE**            | 3.0671           | **2.9819**          | Run 2 (-0.0852)   |
| **RMSE**           | 3.8280           | **3.7570**          | **Run 2 (-0.0710)** |
| **R²**             | 0.2854           | **0.3116**          | **Run 2 (+0.0262)** |

**Decision:** `Tuned_Model_Run2` achieved superior accuracy across all regression metrics and was saved to `models/student_performance_model.joblib`.

### 2. Evidently Data Drift Analysis

| Feature Name | Injected Modification | Statistical Test | p-value | Drift Detected? |
|--------------|-----------------------|------------------|---------|-----------------|
| `studytime`  | Categories 1 & 2 -> 4 | Kolmogorov-Smirnov | `0.0000` | **YES** |
| `absences`   | +5 to all records     | Kolmogorov-Smirnov | `6.53e-28` | **YES** |
| `failures`   | +1 (capped at 4)      | Kolmogorov-Smirnov | `0.0000` | **YES** |
| `age`        | *Unchanged (Control)* | Kolmogorov-Smirnov | `1.0000` | **NO** |
| `Medu`       | *Unchanged (Control)* | Kolmogorov-Smirnov | `1.0000` | **NO** |
| `Fedu`       | *Unchanged (Control)* | Kolmogorov-Smirnov | `1.0000` | **NO** |

- **Dataset-Level Drift Status:** **DRIFT DETECTED** (3 out of 6 features drifted = 50.0% drift share).
- **False Positive Rate:** 0.0% (all 3 control features correctly retained null hypothesis).

### 3. Model Prediction Drift Analysis
- **Reference Predictions Mean (G3):** `10.38`
- **Current (Drifted) Predictions Mean (G3):** `10.22`
- **Prediction Shift:** `-0.15` grade points.

---

## Generated Artifacts & Reports

| Artifact Path | Format | Description |
|---------------|--------|-------------|
| `reports/actual_vs_predicted.png` | PNG Image | Publication-quality scatter plot with ideal $y=x$ reference line |
| `reports/simulated_drift_distributions.png` | PNG Image | KDE density plots comparing Reference vs. Current features |
| `reports/prediction_drift_comparison.png` | PNG Image | KDE plot comparing model output grade distributions |
| `reports/data_drift_report.html` | Interactive HTML | Evidently full visual report with interactive charts and summaries |
| `reports/data_drift_report.json` | JSON | Evidently raw statistical test scores and thresholds |
| `reports/feature_drift_summary.csv` | CSV | Tabular summary of drift status, p-values, and means |
| `reports/model_metrics.json` | JSON | Machine-readable hyperparameters and comparative metrics |
| `reports/project_summary.json` | JSON | Unified dictionary of end-to-end execution outcomes |
| `reports/assignment_report.md` | Markdown | Academic report adhering to college assignment specifications |

---

## Demonstration Evidence & Screenshots

Refer to [`screenshots/README.md`](screenshots/README.md) for the complete list of 11 required screenshots for assignment submission and instructions on how to capture them from your local browser and terminal.

---

## Limitations & Future Improvements

### Limitations
1. **Dataset Size:** The UCI Student Performance dataset contains 395 records. While ideal for education and rapid iteration, deep learning or high-capacity models require larger student cohorts.
2. **Exclusion of Intermediate Grades:** Removing `G1` and `G2` limits the maximum achievable $R^2$ to ~0.31, as final grades correlate heavily with earlier term tests. However, this satisfies our objective of evaluating purely demographic and behavioral predictors.
3. **Simulated Nature of Drift:** The drift was artificially injected to provide a controlled environment. Real-world drift may involve subtler, multi-feature covariance shifts.

### Future Improvements
1. **CI/CD Integration:** Automate drift checks via GitHub Actions to trigger automated retraining workflows when drift is detected.
2. **Model Registry:** Integrate MLflow Model Registry with staging and production model tags.
3. **Web Dashboard:** Build a lightweight FastAPI or Streamlit front-end for interactive single-student grade predictions.

---

## License & Attribution

- **Dataset Attribution:** Courtesy of P. Cortez and A. Silva, *Student Performance Data Set*, UCI Machine Learning Repository, 2008.
- **License:** Distributed under the [MIT License](LICENSE).
