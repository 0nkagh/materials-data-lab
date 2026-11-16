# Model Card: Materials Data Lab (v1.5.0)

## Model Details
- **Architecture:** XGBoost Regressor (hyperparameters selected in internal V2-C2 CV study; `n_estimators=800`, `max_depth=7`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=1.0`, `reg_lambda=1`).
- **Package Version:** `1.5.0`
- **Input Features (13):** 11 compositional variables (`wt%`) + `temper_temp_c` (`°C`) + `log_time` (`log10(temper_time_s)`).
- **Output:** Point prediction for post-tempering hardness (`HRC`) + fixed-width empirical OOF prediction band (`±q90`, **not** externally calibrated or guaranteed).

## Intended Use
- **Primary Use Case:** Research prototype for studying historical tempering data limits, baseline comparisons, and candidate tempering windows for laboratory screening.
- **Laboratory Verification:** Any candidate prediction or tempering condition requires domain-expert review and physical testing.

## Out-of-Scope Use
This prototype is **NOT** intended for:
- Production heat-treatment recipes or furnace control in manufacturing facilities.
- Automated decision making without metallurgist oversight and physical validation.
- Steel chemistries outside observed 1D dataset bounds or unverified/under-covered steel families.

## Training Data
- **Shape:** `1466` rows, `17` raw columns (`13` active model features; `initial_hrc` excluded due to `64.7%` missingness in Grange subset).
- **Source:** "Tempering data for carbon and low alloy steels" (Raiipa Technologies, Kaggle: `rgerschtzsauer`), `CC BY 4.0`.
- **File Integrity:** Verified via SHA-256 (`3f88fdccf4bb2f3c03be6eeca03001b4bd8a2b0de5cd4818195486182a4762b0`). See [DATA_SHEET.md](DATA_SHEET.md).

## Evaluation Data & Methodology
- **Strategy:** Internal `5`-fold `GroupKFold` (`S2`), grouped by the `34` `Steel type` strings within the same 3 literature sources.
- **Methodological Caveats:** Hyperparameter search, model selection, and reported `S2` metrics share the same 5-fold `GroupKFold` split (no nested CV or external test set). Chemically similar grades and same-literature experimental protocols can appear in both train and validation folds. See [EVALUATION.md](EVALUATION.md).

## Recorded Internal CV Metrics (Historical Study)
- **Mean Absolute Error (MAE):** `2.35 ± 0.39 HRC` (fold-mean) / `2.48 HRC` (pooled OOF).
- **R² Score:** `0.94` (fold-mean / pooled OOF).
- **Empirical Uncertainty Band (`q90`):** Unweighted `0.90` quantile of pooled `1466` OOF residuals (evaluated on the same 1466 OOF pool, without finite-sample or group correction — no external coverage guarantee).
  - Recorded `q90` values differ across artifacts (`5.080 HRC` in V3-A `phase_v3a_physics.json`, `4.9673 HRC` in `models/artifact_meta.json`, `4.98 HRC` in `phase7a_baselines.md`; exact discrepancy cause unverified without re-running).
  - Global same-pool OOF empirical hit rate: `89.97%` (`phase_v3a_physics.json`).

## Failure Modes & Weaknesses
- **OOD Flatlining:** Tree-based XGBoost outputs constant leaf values beyond observed splits.
- **Fixed Band Width, Application Input Scope vs. 1D Observed Bounds:** `±q90` does not adapt to input uncertainty or distance from training support. Furthermore, application-supported input bounds (`SUPPORTED_INPUT_BOUNDS`, including `0 < T <= 800 °C` software input cap), observed 1D dataset bounds (`100.0–704.4 °C`), and true metallurgical suitability (`Ac1` and transformation depend on steel composition and conditions) are distinct; passing 1D min–max or application input bounds does **not** verify joint multivariate support or metallurgical validity.
- **Severe Family Under-Coverage (`reports/phase7d_calibration.md`, `q90=5.08` run):** Five specific steel groups showed severe empirical under-coverage: `AISI-SAE 9264` (`12.5%`, high-Si `1.62%`), `0.31%C plain carbon` (`37.8%`), `AISI-SAE 5160` (`47.1%`, `Cr_steel` with `Si=0.22%, Cr=0.79%` — previously mislabeled `Si-spring` in text), `Nitriding Steel` (`67.5%`), and `AISI-SAE E52100` (`74.5%`).
- **Missing Pre-Temper & Quench Variables:** `initial_hrc` (`64.7%` missing), cooling rate/medium, austenitizing schedule, and section geometry are not modeled.

## Limitations & Oversight
See [LIMITATIONS.md](LIMITATIONS.md). Physical laboratory coupon verification and qualified metallurgical sign-off are mandatory prior to any experimental or pilot use.
