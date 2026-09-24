# Materials Data Lab

Materials Data Lab is an independent learning and portfolio project investigating the empirical and metallurgical relationship between steel alloy composition, tempering conditions (temperature and soaking time), and the resulting post-tempering hardness (HRC) in carbon and low-alloy steels.

## Project Status
- **Phase 1-5: Completed.** The project has successfully established a read-only data inventory, broken down data anomalies, generated numerical EDA, established a modeling MVP (RandomForest vs Physics Baseline), and evaluated the predictions in-depth.
- **Phase 6a (completed 2026-09-25)**: permutation importance cross-check of MDI rankings + observational initial_hrc subset study (517 rows, MAE 4.02 → 3.53).
- **Roadmap (Candidate Future Work)**:
  - Investigating the remaining UNKNOWN variations.
  - Adding SHAP explainability analysis.
  - Building a local Gradio demo for inference.
  - Developing an `initial_hrc` subset model for the partial data available.

## Results
Below are the results of the MVP modeling (Phase 4). The physics baseline (B2) relies solely on the Hollomon-Jaffe parameter, while M1 uses the full composition and process variables.

| Model | S1 (RandomSplit) - MAE / RMSE / R² | S2 (GroupSplit) - MAE / RMSE / R² |
| :--- | :--- | :--- |
| **B1_Naive** | 11.81 / 14.84 / -0.05 | 11.51 / 14.25 / -0.07 |
| **B2_Physics** | 4.56 / 5.89 / 0.83 | 4.67 / 5.76 / 0.81 |
| **M1_RF** | 1.52 / 2.35 / 0.97 | 2.69 / 3.57 / 0.93 |

*Gözlem*: B2 (Fizik Baseline), tek bir birleşik özellikle R²=0.81 başarısına ulaşarak metalurjik denklemin veri üzerindeki sağlamlığını kanıtlamıştır.

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
