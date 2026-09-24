# Materials Data Lab

Materials Data Lab is an independent learning and portfolio project investigating the empirical and metallurgical relationship between steel alloy composition, tempering conditions (temperature and soaking time), and the resulting post-tempering hardness (HRC) in carbon and low-alloy steels.

## Current Project Status
- **Phase 1: Project Setup and Read-Only Data Inventory.**
- Scope is strictly limited to repository skeleton setup, non-destructive data auditing, physical threshold checking, and provenance tagging.
- No data cleaning, feature engineering, predictive modeling, or dashboard development is included in this phase.

## Running Tests and Inventory
To run the automated test suite:
```bash
pytest -q
```

To run the data inventory CLI without installation:
```bash
python -m materials_data_lab.inventory --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
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
