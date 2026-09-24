# Materials Data Lab

[![CI](https://github.com/0nkagh/materials-data-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/0nkagh/materials-data-lab/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Materials Data Lab is an independent learning and portfolio project investigating the empirical and metallurgical relationship between steel alloy composition, tempering conditions (temperature and soaking time), and the resulting post-tempering hardness (HRC) in carbon and low-alloy steels.

## Project Status
- **Phase 1-5: Completed.** The project has successfully established a read-only data inventory, broken down data anomalies, generated numerical EDA, established a modeling MVP (RandomForest vs Physics Baseline), and evaluated the predictions in-depth.
- **Phase 6a (completed 2026-09-25)**: permutation importance cross-check of MDI rankings + observational initial_hrc subset study (517 rows, MAE 4.02 → 3.53).
- **V2-A**: Repository polishing (CI setup, CITATION.cff, README finalization).
- **Roadmap (Candidate Future Work)**:
  - Investigating the remaining UNKNOWN variations.
  - Adding SHAP explainability analysis.
  - Building a local Gradio demo for inference.
  - Developing an `initial_hrc` subset model for the partial data available.

## Key Findings
1. **The power of physical baselines**: The 1945 Hollomon-Jaffe parameter, as a single variable, achieves a remarkably high R²=0.81 in predicting final hardness on unseen steels.
2. **Machine Learning capacity**: The Random Forest MVP (GroupSplit) captures non-linear metallurgical complexities, reaching R²=0.93 and an MAE of 2.7 HRC on unseen steel types.
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
| **M1_RF** | MAE: 1.52 / RMSE: 2.35 / R²: 0.97 | MAE: 2.69 / RMSE: 3.57 / R²: 0.93 |
| **Phase 6a Set A** | (subset) | MAE: 4.02 / RMSE: 4.91 / R²: 0.83 [observational] |
| **Phase 6a Set B** | (subset, with initial_hrc) | MAE: 3.53 / RMSE: 4.45 / R²: 0.86 [observational] |

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
