# Evaluation & Metrology

This document summarizes the validation strategy and results. For full output tables, refer to the respective `reports/phase*.md` auto-generated log files.

## Baselines & Machine Learning Ladder
Based on the full baseline suite detailed in `reports/phase7a_baselines.md`, the model performances (measured on the S2 unseen-grade GroupKFold holdout) are as follows:
- **Dummy Regressor:** 8.94 HRC
- **Ridge Regression (Linear):** 4.70 HRC
- **Random Forest:** 2.64 HRC
- **XGBoost (Tuned Champion):** 2.35 HRC

*(Note: Hyperparameter tuning for XGBoost yielded marginal material gains over default settings, but is maintained for stability).*

## Ablation Study & Feature Impacts
The detailed feature removal study (Ablation Ladder) is available in `reports/phase7b_ablation.md`. 
**Key Finding on `log_time`:** The inclusion of `log_time` alongside `temper_time_s` is completely mathematically redundant for tree-based architectures. The exact 4-decimal Out-Of-Fold MAE is perfectly identical (2.4762 = 2.4762 HRC) whether the feature is present or removed in the 13-feature baseline.

## Uncertainty Calibration (Split-Conformal)
The XGBoost champion model uses a static conformal band `q90 = 5.08 HRC`. 
- **Global Coverage:** 89.9% (on S2 OOF)
- **Family Breakdown:** See `reports/phase7d_calibration.md` for the Wilson 95% Confidence Interval annotation of family-specific coverage.
- **Failures:** 5 chemical groups present severe, statistically verifiable under-coverage (far below 90% target) that exceeds small-sample noise: 9264, 5160, E52100, Nitriding Steel, and 0.31%C plain carbon steel.

## Data Leakage & Validation Semantics
- **S1 vs S2 Difference:** The performance difference between RandomSplit (S1, ~1.05 MAE) and GroupKFold (S2, ~2.35 MAE) highlights the danger of chemical interpolation leakage. S2 is the strictly enforced truth metric.
- **Grade-Disjoint Unit Test:** A synthetic hermetic test confirms `GroupKFold(groups=steel_type)` strictly guarantees no overlap of steel families between training and validation folds.
- **Determinism:** The `make_prediction` and `/recommend` functions are purely deterministic. The underlying XGBoost training is stochastic but locked via random seeds; however, cross-platform bit-level reproducible training is not guaranteed (see Model Card).

## Testing Approach
The CI pipeline relies entirely on hermetic, mocked, and synthetic local unit tests via `pytest` and `fastapi.testclient.TestClient`. 
**Honest Note:** There is consciously no End-to-End (E2E) "live server" network test in the automated pipeline; the API validation simulates the ASGI layer locally.
