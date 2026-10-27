# Project Demonstration & Screenshot Evidence Checklist

This directory is prepared for storing submission evidence and screenshots required for the college assignment and GitHub repository demonstration.

---

## Required Screenshot Checklist

| # | Screenshot Filename | Description / Step | How to Capture |
|---|---------------------|---------------------|----------------|
| 1 | `01_env_verification.png` | Python environment & installed packages | Run `python --version` and `pip list` in the terminal |
| 2 | `02_dataset_validation.png` | Dataset loading, shape, and quality report | Run `python -m src.data_loader` |
| 3 | `03_model_training.png` | Baseline & comparative model training | Run `python -m src.train` |
| 4 | `04_mlflow_dashboard.png` | MLflow experiments dashboard showing runs | Launch `mlflow ui` and open `http://127.0.0.1:5000` |
| 5 | `05_mlflow_run_comparison.png` | MLflow comparison table (Run 1 vs Run 2) | Select both runs in MLflow UI and click **Compare** |
| 6 | `06_actual_vs_predicted.png` | Actual vs. Predicted G3 regression plot | Open `reports/actual_vs_predicted.png` |
| 7 | `07_drift_distribution.png` | Distribution shift before and after drift | Open `reports/simulated_drift_distributions.png` |
| 8 | `08_evidently_html_report.png` | Evidently Interactive HTML Data Drift Report | Open `reports/data_drift_report.html` in a web browser |
| 9 | `09_feature_drift_summary.png` | Feature-level statistical test scores | View `reports/feature_drift_summary.csv` or terminal output |
| 10 | `10_pytest_execution.png` | Automated test suite execution passing | Run `pytest -v` in the terminal |
| 11 | `11_github_repository.png` | Published GitHub repository home page | View repository page on GitHub once published |

---

## Instructions for College Report Submission

1. Ensure all scripts have been executed in sequence:
   ```bash
   python -m src.data_loader
   python -m src.train
   python -m src.prepare_data
   python -m src.detect_drift
   python -m src.generate_summary
   pytest -v
   ```
2. Start the MLflow UI:
   ```bash
   mlflow ui --backend-store-uri mlruns
   ```
3. Capture full-window or focused screenshots of each step corresponding to the checklist above and save them directly into this `screenshots/` directory.
