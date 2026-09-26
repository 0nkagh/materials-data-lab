# Materials Data Lab

[![CI](https://github.com/0nkagh/materials-data-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/0nkagh/materials-data-lab/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Materials Data Lab is an independent learning and portfolio project investigating the empirical and metallurgical relationship between steel alloy composition, tempering conditions (temperature and soaking time), and the resulting post-tempering hardness (HRC) in carbon and low-alloy steels.

## Project Status
- **Phase 1-5 & Phase 6a: Completed.**
- **V2-A..V2-F: Completed.** Repository polishing, tests, local demo, and final evaluation. See [docs/methodology.md](docs/methodology.md) for the complete scientific narrative.

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
Below are the results of the MVP modeling (Phase 4) and the subset study (Phase 6a). *(Note: Aggregations labeled as MAE for XGB_tuned are 2.35 HRC fold-mean and 2.48 HRC pooled OOF).* See [docs/EVALUATION.md](docs/EVALUATION.md) for more details. 

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

## Problem
Steel parts get their final hardness during tempering, and picking the right temperature and soak time still comes down to handbooks and trial runs. I wanted to see how far a careful model could go with a public dataset — and where it stops being trustworthy.

## Why
Heat treatment work doesn't need another black box. It needs a tool that says "I don't know" outside its comfort zone. So this project ships predictions with error bands, recipe suggestions marked as lab candidates, and failure cases written down instead of hidden.

## Data
Please see the Data Attribution section and [docs/DATA_SHEET.md](docs/DATA_SHEET.md) for full details on dataset provenance, licenses, and structural limits.

## Methodology
Full scientific narrative lives in `docs/methodology.md`; every decision is logged in `docs/cleaning_decisions.md` (D-01..D-24).

## Model
Please see [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for full details on the XGBoost champion model architecture and limitations.

## Limitations
See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) and [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) for full context, limitations, and the pre-registered commercial pilot protocol ([docs/COMMERCIAL_PILOT.md](docs/COMMERCIAL_PILOT.md)).

## How to run
See the Local Demo and Usage sections above for instructions on running the app and test suites.

## Example output
Run `python app.py` for the demo or `python api.py` for the REST service. Example `/recommend` response (4140 steel, target 45 HRC) is in `reports/phase_v5a_recommender.md`.

## Roadmap
- steel_type UNKNOWN study (V2-D)
- initial_hrc subset model (future work — D-15 imputation rejected; Phase 6a observational only)
- Shadow pilot with real lab samples (protocol ready, docs/COMMERCIAL_PILOT.md)

Done in earlier releases: SHAP explainability, local Gradio demo, REST API, recipe recommender, per-family calibration study.
