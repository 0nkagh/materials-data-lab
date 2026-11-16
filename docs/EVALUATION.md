# Evaluation & Metrology

This document summarizes the validation methodology, recorded study results, and known methodological gaps.

## P1 Unbiased Grouped Evaluation (Held-Out Calibration Benchmark)
Executed via `materials_data_lab.p1_evaluation` (`outer GroupKFold(5)` → target-blind `proper_train`/`calibration` split `seed=42` → inner `GroupKFold(3)` `n_iter=20` on `proper_train` only → finite-sample `ceil((n+1)*(1-alpha))` order-statistic `q90_k` on `calibration` only → outer-test evaluation; reproducible summary and figures in [reports/p1_grouped_evaluation/README.md](../reports/p1_grouped_evaluation/README.md)):

| Model (`proper_train` `20–21` groups) | Fold-Mean MAE ± Std | Pooled Row-Weighted MAE (`n=1466`) | Pooled RMSE | Pooled R² | Group-Macro MAE (`G=34`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`B1_Naive`** (median) | `11.5008 ± 0.6353` | `11.4993` | `14.3426` | `-0.0385` | `11.6793 ± 2.5711` |
| **`B2_Physics`** (`p_hj`, `C=19.5`) | `4.6867 ± 1.2491` | `4.6943` | `5.8927` | `0.8247` | `4.7513 ± 2.5639` |
| **`XGBoost` (P1)** | **`2.9838 ± 0.6322`** | **`2.9842`** | **`3.8800`** | **`0.9240`** | **`3.0450 ± 1.6607`** |

- **P1 Empirical Uncertainty & Coverage (`outer_test`):** Global row-weighted empirical coverage = **`87.24%`** (`1279/1466`), fold-mean coverage = `87.28% ± 5.01%`, group-macro coverage = **`87.12% ± 20.35%`**, fold-mean `q90_k = 5.7217 HRC`, and mean full band width (`2 × q90_k`) = **`11.4433 HRC`** (`11.4343 HRC` row-weighted).
- **Comparing P1 (`2.98`) vs. Historical `V2-C2` (`2.35` / `2.48`):** Both structural training reductions (`20–21` `proper_train` groups vs `27–28` groups after holding out `7` calibration groups), inner `3`-fold/`20`-iter tuning differences, and removal of selection/calibration overlap contribute; the score shift must **not** be attributed solely to historical selection optimism.
- **Boundary & Serving Separation:** There is **no independent external dataset and no theoretical 90% coverage guarantee** for unseen steel groups. The runtime demo/API (`app.py`, `api.py`) continues to serve **legacy pre-built `models/` artifacts** (`conformal_q90 = 4.9673 HRC`), not P1 `proper_train` models.

## Historical Baselines & Model Comparison (Legacy Shared-CV Study)
> **Baseline Provenance Notice (2026-11-17):** Earlier revisions of this file cited `Dummy Regressor: 8.94 HRC` and `Ridge Regression: 4.70 HRC` while referencing `reports/phase7a_baselines.md`. Those two figures do not match `reports/phase7a_baselines.md` and have no reproducible generator script in `src/`, so they are **not** presented as verified current results. Similarly, `phase7a` and `phase7b` lack complete standalone CLI generators (`ablation_study.py` is a non-writing stub).

Recorded `S2` (`GroupKFold(n_splits=5, groups=steel_type)`) values from existing report artifacts are:
- **`reports/phase4_modeling.md` & `data/inventory/phase4_modeling.json` (fold-mean):**
  - `B1_Naive` (median `DummyRegressor`): `MAE = 11.508 ± 0.688 HRC`
  - `B2_Physics` (`LinearRegression` on `p_hj`, assumed `C = 19.5`): `MAE = 4.673 ± 1.093 HRC`
  - `M1_RF` (default): `MAE = 2.687 ± 0.389 HRC`
- **`data/inventory/phase_v2e_hj_evolution.json` (fold-mean):**
  - `Composite_C`: `MAE = 3.336 ± 0.397 HRC`
- **`data/inventory/phase_v2c2_compare.json` (fold-mean):**
  - `RF_tuned`: `MAE = 2.639 ± 0.245 HRC`
  - `XGB_tuned`: `MAE = 2.350 ± 0.388 HRC`, `R² = 0.944 ± 0.020`
- **`reports/phase7a_baselines.md` (pooled OOF across 1466 rows):**
  - `DummyRegressor (mean / median)`: `MAE = 11.59 / 11.51 HRC`
  - `Ridge (std-scale ile)`: `MAE = 3.86 HRC` *(generator script absent from `src/`)*
  - `RF_tuned`: `MAE = 2.71 HRC`
  - `XGB_tuned`: `MAE = 2.48 HRC`, `R² = 0.944`, `|resid| p90 = 4.98 HRC`

*Note:* Hyperparameter searches (`RandomizedSearchCV` for RF and XGB), model selection (`D-12`), and feature/constraint decisions (`D-14`, `D-17`, `D-18`) were evaluated on the **same 34-group `GroupKFold(5)` split** without nested outer CV or an untouched external holdout set, so mild selection optimism cannot be excluded.

## Ablation Study (Recorded in `reports/phase7b_ablation.md`)
`reports/phase7b_ablation.md` records identical 4-decimal pooled OOF MAE (`2.4762 HRC`) with or without `log_time` alongside `temper_time_s` for tree-based XGBoost *(generator script absent from `src/`)*.

## Uncertainty Calibration (OOF Residual Pooling — Not Independent Split-Conformal)
- **Method:** `q90` is computed via `np.quantile(abs_oof_residuals, 0.90)` over all `1466` `GroupKFold(5)` OOF residuals without finite-sample `(n+1)(1-alpha)/n` or group-level correction. Calibration and global coverage (`~89.97%`) use the **same** 1466 residual pool; there is **no independent calibration holdout** and **no per-family or external guarantee**.
- **Recorded `q90` Sources (discrepancy unverified without re-run):**
  - `5.0804 HRC` from `v3_physics.py` (`n_jobs=-1`, `data/inventory/phase_v3a_physics.json`)
  - `4.9673 HRC` from `model_artifact.py` (`n_jobs=1`, `models/artifact_meta.json`)
  - `4.98 HRC` (`|resid| p90`) in `reports/phase7a_baselines.md`
- **Group Failures (`reports/phase7d_calibration.md`, `q90 = 5.08` run):** Five steel groups show severe empirical under-coverage below `90%`: `AISI-SAE 9264` (`12.5%`, `n=40`), `0.31%C plain carbon` (`37.8%`, `n=45`), `AISI-SAE 5160` (`47.1%`, `n=51`; Cr-steel, not Si-spring), `Nitriding Steel` (`67.5%`, `n=40`), and `AISI-SAE E52100` (`74.5%`, `n=51`).

## Grouping & Leakage Semantics
- **`S1` vs `S2`:** `S1` (`RandomSplit`) leaks identical steel series across train/test (`~1.05 MAE`). `S2` (`GroupKFold(groups=steel_type)`) prevents identical `Steel type` strings from appearing in both train and test folds.
- **Scope of `"unseen steel"`:** Because `GroupKFold` groups only by the 34 `Steel type` strings within the same 3 literature compilations, chemically neighboring grades (`1045` vs `1049`, `1038` vs `1045`) and rows from the same literature sources remain across folds, and heat/quench/sample IDs are missing. `S2` is an internal grade-string holdout, **not** independent external validation.

## Automated Testing Scope
The test suite relies on hermetic synthetic and mocked unit tests via `pytest` and `fastapi.testclient.TestClient`, without touching real `models/`, running live network servers, or verifying external lab data.
