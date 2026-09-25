# Materials Data Lab

[![CI](https://github.com/0nkagh/materials-data-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/0nkagh/materials-data-lab/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Materials Data Lab is an independent learning and portfolio project investigating the empirical and metallurgical relationship between steel alloy composition, tempering conditions (temperature and soaking time), and the resulting post-tempering hardness (HRC) in carbon and low-alloy steels.

## Project Status
- **Phase 1-5 & Phase 6a: Completed.**
- **V2-A..V2-F: Completed.** Repository polishing, tests, local demo, and final evaluation. See [docs/methodology.md](docs/methodology.md) for the complete scientific narrative.
- **Roadmap (Candidate Future Work)**:
  - Investigating the remaining UNKNOWN variations.
  - Adding SHAP explainability analysis.
  - Building a local Gradio demo for inference.
  - Developing an `initial_hrc` subset model for the partial data available.

## Key Findings
1. **The power of physical baselines**: The 1945 Hollomon-Jaffe parameter, as a single variable, achieves a remarkably high R²=0.81 in predicting final hardness on unseen steels.
2. **Machine Learning capacity**: The Random Forest MVP (GroupSplit) captures non-linear metallurgical complexities, reaching R²=0.93 and an MAE of 2.7 HRC on unseen steel types. XGBoost (tuned) became the champion under grouped CV (V2-C2): MAE 2.35 ± 0.39 vs RF 2.64 ± 0.24 on unseen steel grades.
3. **Identified weak spots**: The model's weakest regions (<25 HRC and ≥650°C) precisely overlap with the low-hardness extrapolations flagged earlier by the quality auditing pipeline, validating the necessity of a data-quality-first approach.

## Repository Structure
```text
.
├── .github/workflows/   # Continuous Integration (CI) configuration
├── data/
│   ├── inventory/       # Generated JSON reports (metrics, flags, insights)
│   └── raw/             # local-only, git-ignored (immutable Kaggle CSV)
├── docs/                # Project documentation and decision logs
├── reports/             # Generated Markdown analytical reports
├── src/                 # Main Python package (materials_data_lab)
└── tests/               # Pytest suite
```

## Results
Below are the results of the MVP modeling (Phase 4) and the subset study (Phase 6a). 

| Model | S1 (RandomSplit) | S2 (GroupSplit: unseen steel) |
| :--- | :--- | :--- |
| **B1_Naive** | MAE: 11.81 / RMSE: 14.84 / R²: -0.05 | MAE: 11.51 / RMSE: 14.25 / R²: -0.07 |
| **B2_Physics** | MAE: 4.56 / RMSE: 5.89 / R²: 0.83 | MAE: 4.67 / RMSE: 5.76 / R²: 0.81 |
| **B3_Composite_C** | MAE: 3.06 / RMSE: 4.25 / R²: 0.91 | MAE: 3.34 / RMSE: 4.34 / R²: 0.90 |
| **M1_RF** | MAE: 1.52 / RMSE: 2.35 / R²: 0.97 | MAE: 2.69 / RMSE: 3.57 / R²: 0.93 |
| **M1_RF_tuned** | MAE: 1.53 / RMSE: 2.48 / R²: 0.97 | MAE: 2.64 / RMSE: 3.41 / R²: 0.94 |
| **M2_XGB_default** | MAE: 1.20 / RMSE: 2.10 / R²: 0.98 | MAE: 2.50 / RMSE: 3.41 / R²: 0.94 |
| **M2_XGB_tuned** (Champion) | MAE: 1.05 / RMSE: 1.89 / R²: 0.98 | MAE: 2.35 / RMSE: 3.21 / R²: 0.94 |
| **Phase 6a Set A** | (subset) | MAE: 4.02 / RMSE: 4.91 / R²: 0.83 [observational] |
| **Phase 6a Set B** | (subset, with initial_hrc) | MAE: 3.53 / RMSE: 4.45 / R²: 0.86 [observational] |

## Uncertainty (Split-Conformal)
Point predictions are supplemented with a split-conformal prediction interval (Phase V3-A). Based on out-of-fold residuals (GroupKFold), a **±5.08 HRC** interval provides a rigorously calibrated empirical coverage of **~90%** across all hardness ranges and unseen steel alloys.

## Local Demo
The repository includes a local Gradio web interface for interacting with the model. As of V2-F, the demo serves the XGBoost (tuned) champion model. The model artifact is automatically trained on your local machine the first time you run the app (as it's excluded from git). 

As of V3-B, the demo also provides:
1. **Conformal Uncertainty**: A rigorous 90% confidence interval based on `q90` out-of-fold residuals.
2. **SHAP Explanations**: A waterfall plot and top 3 feature effects to explain the model's reasoning for each specific prediction.
```bash
pip install -r requirements.txt
python app.py
```
This will open the predictor interface in your default web browser.

## Recommender System (V5-A)
The project includes a local "Recipe Recommender" via a REST API endpoint (`POST /recommend`). 
Given a target hardness (HRC) and steel composition, the system performs a grid scan (0.5°C resolution) across tempering temperatures and times to find the optimal recipe.
- Returns the shortest tempering time ($t$) and lowest temperature ($T$) within a ±0.5 HRC tolerance.
- Automatically rejects targets that are `NOT_ACHIEVABLE` due to physical constraints or being out of model support range.
- Incorporates conformal prediction intervals (`q90`) for the suggested recipe.

## Usage
To run the automated test suite:
```bash
pytest -q
```

To run the data pipelines and generate reports:
```bash
python -m materials_data_lab.inventory --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
python -m materials_data_lab.breakdown --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
python -m materials_data_lab.eda --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
python -m materials_data_lab.modeling --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
python -m materials_data_lab.evaluation --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
python -m materials_data_lab.phase6a --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
```

## Data Attribution

This project uses the dataset **"Tempering data for carbon and low alloy steels"**
by **Raiipa Technologies** (Kaggle: rgerschtzsauer), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Source: https://www.kaggle.com/datasets/rgerschtzsauer/tempering-data-for-carbon-and-low-alloy-steels
- Data version: Kaggle dataset version 3 (file dated 2024-05-20); accessed 2026-09-22.
- Changes: **No modifications** were made to the data file. It is analyzed read-only
  and excluded from version control via `.gitignore`.
- The literature sources cited by the dataset card (Hollomon & Jaffe 1945;
  Grange & Baughman 1956; Penha 2010) remain the primary experimental sources;
  see docs/data_dictionary.md for notes.
