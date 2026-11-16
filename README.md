# Materials Data Lab

[![CI](https://github.com/0nkagh/materials-data-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/0nkagh/materials-data-lab/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Materials Data Lab is an independent learning and research prototype investigating the empirical relationship between steel alloy composition, tempering conditions (temperature and soaking time), and post-tempering hardness (HRC) in carbon and low-alloy steels. It is **not** validated for production heat-treatment control.

## Project Status
- **Phase 1–5 & Phase 6a / V2–V5:** Prototype completed ([docs/methodology.md](docs/methodology.md)).
- **Safety & Scope Hardening (2026-11-17):** Runtime auto-training disabled in `app.py` / `api.py`, observed dataset boundaries (`OBSERVED_UNIVARIATE_BOUNDS`) reconciled against the raw CSV, application-supported input scope (`SUPPORTED_INPUT_BOUNDS`, distinct from observed range and metallurgical suitability) enforced before inference, out-of-scope chemistry blocked in `/recommend`, and empirical claims narrowed to match code behavior.
- **P1 Unbiased Grouped Evaluation (`docs/p1-results`, 2026-11-17):** Added `5`-fold outer `GroupKFold(steel_type)` evaluation with target-blind `proper_train`/`calibration` splitting and inner `3`-fold hyperparameter search ([reports/p1_grouped_evaluation/README.md](reports/p1_grouped_evaluation/README.md)). The served demo/API still loads legacy `models/` artifacts (`artifact_meta.json`, `q90 = 4.9673 HRC`), not P1 `proper_train` models.

## Key Findings (Historical Internal CV Study vs. New P1 Benchmark)
1. **Physical baselines (`B2`):** Single-variable linear regression on Hollomon-Jaffe (`C = 19.5` assumed) yields `4.67 HRC` (`R² = 0.81`, historical `S2`) and **`4.69 HRC`** pooled (`4.69 ± 1.25 HRC` fold-mean, `R² = 0.82`) under P1 `proper_train` evaluation.
2. **XGBoost — Historical vs. Unbiased P1 Benchmark:** Historical shared-CV study reported `2.35 ± 0.39 HRC` fold-mean MAE (`2.48 HRC` pooled OOF). Under strict P1 nested evaluation with held-out calibration (`proper_train` = `20–21` groups per fold; `inner 3-fold × 20` candidates; `34` grades across `1466` rows), XGBoost achieves **`2.98 HRC` pooled MAE (`2.9838 ± 0.6322 HRC` fold-mean, `3.0450 ± 1.6607 HRC` group-macro, `R² = 0.92`)**, **`87.24%` empirical coverage** (`1279/1466`; group-macro `87.12%`), and **`11.44 HRC` mean full band width** (`2 × q90_k`). Because P1 shrinks training groups (`-25%`) to hold out calibration and uses a different inner search budget, this gap cannot be explained solely by historical selection optimism.
3. **Identified weak spots & guarantees:** Error and under-coverage remain severe on `AISI-SAE 9264` (`10.0%`), `0.31%C plain carbon` (`37.8%`), `AISI-SAE 4640` (`50.0%`), `Nitriding Steel` (`60.0%`), and `AISI-SAE 4037` (`62.5%`), plus `<25 HRC` / `>=650 °C` tails. There is **no independent external dataset and no theoretical 90% coverage guarantee** for unseen steel grades.

## Repository Structure
```text
.
├── .github/workflows/   # Continuous Integration (CI) configuration
├── data/
│   ├── inventory/       # Generated JSON reports (metrics, flags, insights)
│   └── raw/             # local-only, git-ignored (immutable Kaggle CSV)
├── docs/                # Project documentation and decision logs
├── models/              # local-only, git-ignored pre-trained artifacts
├── reports/             # Generated Markdown analytical reports
├── src/                 # Main Python package (materials_data_lab)
└── tests/               # Hermetic pytest suite
```

## Results

### P1 Unbiased Grouped Evaluation (Held-Out Calibration, New Protocol)
Executed via `materials_data_lab.p1_evaluation` (`outer GroupKFold(5)` → target-blind `proper_train`/`calibration` split `seed=42` → inner `GroupKFold(3)` `n_iter=20` on `20–21` `proper_train` groups only → finite-sample `ceil((n+1)*(1-alpha))` `q90_k` on `7` held-out calibration groups → outer-test evaluation across all `1466` rows / `34` steel grades; full report in [reports/p1_grouped_evaluation/README.md](reports/p1_grouped_evaluation/README.md)):

| Model (`proper_train` `20–21` groups) | Fold-Mean MAE ± Std | Pooled MAE (`n=1466`) | Pooled RMSE | Pooled R² | Group-Macro MAE (`G=34`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B1_Naive** (median) | `11.5008 ± 0.6353` | `11.4993` | `14.3426` | `-0.0385` | `11.6793 ± 2.5711` |
| **B2_Physics** (`p_hj`, `C=19.5`) | `4.6867 ± 1.2491` | `4.6943` | `5.8927` | `0.8247` | `4.7513 ± 2.5639` |
| **XGBoost (P1)** | **`2.9838 ± 0.6322`** | **`2.9842`** | **`3.8800`** | **`0.9240`** | **`3.0450 ± 1.6607`** |

- **P1 Empirical Band & Coverage (`outer_test`):** Global row-weighted empirical coverage = **`87.24%`** (`1279/1466`; fold-mean `87.28% ± 5.01%`), group-macro coverage = **`87.12% ± 20.35%`**, fold-mean `q90_k = 5.7217 HRC`, and mean full band width (`2 × q90_k`) = **`11.4433 HRC`** (`11.4343 HRC` row-weighted). There is **no independent external dataset and no theoretical 90% coverage guarantee** for unseen steel groups.
- **Why P1 Differs From Historical `2.35 / 2.48 HRC`:** The shift cannot be attributed solely to historical selection optimism—holding out `7` calibration groups shrinks `proper_train` from `27–28` groups (`~80%`) to `20–21` groups (`~60%`), and P1 runs `3`-fold × `20`-iter inner search rather than `5`-fold × `30`-iter on full data.

### Historical Internal Study Tables (Shared-CV Legacy Benchmark)
Reported metrics below come from historical Phase 4 / V2-C2 (`GroupKFold` over 34 `Steel type` strings within the same 3 literature sources; hyperparameter search and final scoring shared that CV partition without nested CV) and the observational Phase 6a subset. For baseline discrepancies (`reports/phase7a_baselines.md` vs earlier summary figures), see [docs/EVALUATION.md](docs/EVALUATION.md).

| Model | S1 (RandomSplit — leaky reference) | S2 (`GroupKFold(5, steel_type)`, fold-mean) |
| :--- | :--- | :--- |
| **B1_Naive** | MAE: 11.81 / RMSE: 14.84 / R²: -0.05 | MAE: 11.51 / RMSE: 14.25 / R²: -0.07 |
| **B2_Physics** | MAE: 4.56 / RMSE: 5.89 / R²: 0.83 | MAE: 4.67 / RMSE: 5.76 / R²: 0.81 |
| **B3_Composite_C** | MAE: 3.06 / RMSE: 4.25 / R²: 0.91 | MAE: 3.34 / RMSE: 4.34 / R²: 0.90 |
| **M1_RF** | MAE: 1.52 / RMSE: 2.35 / R²: 0.97 | MAE: 2.69 / RMSE: 3.57 / R²: 0.93 |
| **M1_RF_tuned** | MAE: 1.53 / RMSE: 2.48 / R²: 0.97 | MAE: 2.64 / RMSE: 3.41 / R²: 0.94 |
| **M2_XGB_default** | MAE: 1.20 / RMSE: 2.10 / R²: 0.98 | MAE: 2.50 / RMSE: 3.41 / R²: 0.94 |
| **M2_XGB_tuned** | MAE: 1.05 / RMSE: 1.89 / R²: 0.98 | MAE: 2.35 ± 0.39 / RMSE: 3.21 / R²: 0.94 *(pooled OOF MAE: 2.48)* |
| **Phase 6a Set A** | (517-row subset) | MAE: 4.02 / RMSE: 4.91 / R²: 0.83 `[observational]` |
| **Phase 6a Set B** | (517-row subset, +`initial_hrc`) | MAE: 3.53 / RMSE: 4.45 / R²: 0.86 `[observational]` |

## Uncertainty (Empirical OOF Prediction Band)
Point predictions are accompanied by a fixed-width empirical prediction band derived by pooling all `1466` `GroupKFold(5)` out-of-fold (OOF) absolute residuals and taking the unweighted `0.90` quantile (`np.quantile(residuals, 0.90)` without finite-sample `(n+1)(1-alpha)/n` or group-level correction):
- **Not independent split-conformal:** Calibration and the reported `~89.97%` empirical coverage are evaluated on the **same** 1466 OOF residual pool; there is **no independent external coverage guarantee**.
- **Recorded `q90` provenance (discrepancy unverified without re-run):** `v3_physics.py` (`phase_v3a_physics.json`, `n_jobs=-1`) recorded `5.080 HRC`; `model_artifact.py` (`models/artifact_meta.json`, `n_jobs=1`) recorded `4.9673 HRC`; `reports/phase7a_baselines.md` recorded `|resid| p90 = 4.98 HRC`.
- **Severe per-family under-coverage:** Empirical OOF coverage drops below target in five groups: `AISI-SAE 9264` (`12.5%`, `n=40`), `0.31%C plain carbon` (`37.8%`, `n=45`), `AISI-SAE 5160` (`47.1%`, `n=51`; Cr-steel, not Si-spring), `Nitriding Steel` (`67.5%`, `n=40`), and `AISI-SAE E52100` (`74.5%`, `n=51`). See [reports/phase7d_calibration.md](reports/phase7d_calibration.md).

## Local Demo & Explicit Training
`app.py` and `api.py` load **legacy pre-built artifacts from `models/`** (`artifact_meta.json`, `XGB_tuned` from historical shared-CV study, `conformal_q90 = 4.9673 HRC`) using `load_artifact`—they do **not** run or reflect P1 `proper_train` models—and **never** start automatic retraining or silently overwrite existing `.joblib` files (they return an explicit error / HTTP `503 model_not_ready` if artifacts are missing or fail to load). Because `.joblib` files use Python `pickle` serialization, only load artifacts from trusted local `models/` directories.

To install dependencies in an isolated virtual environment, explicitly build artifacts (when permitted), and launch the demo or API:
```bash
pip install -r requirements.txt
# Explicit artifact creation (only when you intend to train & write models/):
python -m materials_data_lab.model_artifact --csv "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv"
# Runtime demo / REST service (load-only; does not require raw CSV):
python app.py
python api.py
```
SHAP outputs in the demo represent **model output attribution across correlated alloying inputs**, not causal metallurgical effects.

## Candidate Processing Condition Recommender (`POST /recommend`)
Given a target hardness (HRC) and steel composition, `/recommend` scans a grid (`T` from `100.0` to `704.0 °C` at a `0.5 °C` numerical scan step across tree thresholds; discrete soaking times `1800, 3600, 7200, 14400 s`) to propose **candidate tempering conditions requiring laboratory verification** (not production-ready "optimal" recipes):
- Chemistry outside `OBSERVED_UNIVARIATE_BOUNDS` returns `recommendable: false` (`OUT_OF_OBSERVED_SCOPE`) with **no candidate conditions generated**.
- **Three separate concepts:** Passing this application's supported input limits (`SUPPORTED_INPUT_BOUNDS`, including the software cap `0 < T <= 800 °C`, which is not a universal metallurgical boundary or composition-independent `Ac1` limit) or `OBSERVED_UNIVARIATE_BOUNDS` does **not** establish **joint multivariate support or true metallurgical suitability** (quenching medium, section thickness, initial hardness, composition-dependent `Ac1`, temper embrittlement).
- The `±0.5 HRC` matching tolerance applies **only to the model's point prediction** on the numerical scan grid (`0.5 °C` step does not imply sub-degree physical accuracy); the empirical OOF prediction band remains `±q90` (~`±5 HRC`).
- Selecting shortest `t` then lowest `T` over the 4 discrete durations is a simple heuristic rule, **not** furnace energy/cost or embrittlement optimization.
- `NO_CANDIDATE_FOUND` (`NOT_ACHIEVABLE` in legacy notes) indicates no discrete grid point met the `±0.5 HRC` point-prediction tolerance—not physical impossibility.

## Usage
To run lint and the hermetic test suite:
```bash
ruff check .
pytest -q --basetemp=.pytest_tmp
```

To run individual read-only / pipeline scripts explicitly:
```bash
python -m materials_data_lab.inventory --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
python -m materials_data_lab.breakdown --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
python -m materials_data_lab.eda --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
python -m materials_data_lab.modeling --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
python -m materials_data_lab.evaluation --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
python -m materials_data_lab.phase6a --input "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir reports
# Workflow A — Full P1 reproduction from a fresh clone (note: GitHub clone does NOT include data/raw/ or local P1 outputs):
#   1. Download the Raiipa CSV to data/raw/ and verify its SHA-256 matches 3f88fdccf4bb2f3c03be6eeca03001b4bd8a2b0de5cd4818195486182a4762b0:
python -c "import hashlib; print(hashlib.sha256(open('data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv', 'rb').read()).hexdigest())"
#   2. Run P1 grouped evaluation (write to a local directory outside models/):
python -m materials_data_lab.p1_evaluation --csv "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv" --outdir <local_p1_run_dir> --mode full --max-seconds 900
#   3. Build sanitized summary.json, README.md, and PNG figures from that local run:
python -m materials_data_lab.p1_reporting --input-dir <local_p1_run_dir> --outdir reports/p1_grouped_evaluation

# Workflow B — Regenerate report & charts only from an already-saved <local_p1_run_dir> (no training or raw CSV needed):
python -m materials_data_lab.p1_reporting --input-dir <local_p1_run_dir> --outdir reports/p1_grouped_evaluation
```

## Data Attribution
This project uses **"Tempering data for carbon and low alloy steels"** by **Raiipa Technologies** (Kaggle: `rgerschtzsauer`), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Source: https://www.kaggle.com/datasets/rgerschtzsauer/tempering-data-for-carbon-and-low-alloy-steels
- Data version: Kaggle dataset version 3 (file dated `2024-05-20`; accessed `2026-09-22`; SHA-256 `3f88fdccf4bb2f3c03be6eeca03001b4bd8a2b0de5cd4818195486182a4762b0`).
- Stored locally and excluded from Git via `.gitignore` as a workflow practice (not a legal determination; reproducing local runs from a fresh clone requires downloading the verified CSV first; general CC BY 4.0 attribution sharing vs. uploader rights over compiled original literature sources remains distinct—see [reports/p1_grouped_evaluation/README.md](reports/p1_grouped_evaluation/README.md)).
- Primary literature references (`Hollomon & Jaffe 1945`, `Grange & Baughman 1956`, ambiguous `Penha 2010`) are documented in [docs/data_dictionary.md](docs/data_dictionary.md) and [docs/references.md](docs/references.md).

## Documentation Map
- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) · [docs/DATA_SHEET.md](docs/DATA_SHEET.md) · [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- [docs/EVALUATION.md](docs/EVALUATION.md) · [docs/LIMITATIONS.md](docs/LIMITATIONS.md) · [docs/COMMERCIAL_PILOT.md](docs/COMMERCIAL_PILOT.md)
- [docs/methodology.md](docs/methodology.md) · [docs/cleaning_decisions.md](docs/cleaning_decisions.md)
