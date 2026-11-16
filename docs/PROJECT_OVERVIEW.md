# Materials Data Lab: Project Overview

## 1. Problem Statement
Heat treatment of carbon and low-alloy steels requires selecting tempering temperatures (`T`) and soaking times (`t`) toward a target hardness (`HRC`). This research prototype explores how well tabular models trained on an aggregated literature dataset predict post-tempering hardness and flag boundary/uncertainty risks before physical laboratory trials.

## 2. Approach
The pipeline combines a rule-based physical baseline (`Hollomon-Jaffe 1945`, assumed `C = 19.5`) with tree-based models trained on `1466` cleaned literature rows (`17` raw columns, `initial_hrc` excluded due to `64.7%` missingness). Internal evaluation uses `GroupKFold(5)` grouped by the `34` `Steel type` strings within those 3 literature sources (`S2`).

## 3. Champion Model Summary (Historical Internal CV vs. New P1 Benchmark)
- **Served Legacy Artifact (Historical Internal Shared `S2` CV):** On the internal 5-fold `GroupKFold(steel_type)` split (shared with hyperparameter search and model selection, `27–28` groups per train fold), the legacy tuned XGBoost configuration served in `models/` recorded `2.35 ± 0.39 HRC` fold-mean MAE (`2.48 HRC` pooled OOF, `R² = 0.94`).
- **New P1 Unbiased Grouped Benchmark (`docs/p1-results`, [reports/p1_grouped_evaluation/README.md](../reports/p1_grouped_evaluation/README.md)):** Using outer `5`-fold `GroupKFold`, target-blind `proper_train` (`20–21` groups) vs. `calibration` (`7` groups) separation, and inner `3`-fold × `20`-candidate tuning, P1 achieved **`2.98 HRC` pooled MAE (`2.9838 ± 0.6322 HRC` fold-mean, `R² = 0.92`)**, `B2_Physics = 4.69 HRC` pooled MAE, **`87.24%` empirical coverage**, and **`11.44 HRC` mean full band width**. Because P1 trims training size (`-25%` groups) and changes the tuning/calibration design, this shift reflects both smaller `proper_train` pools and removal of optimism—not historical optimism alone.

## 4. Uncertainty Band Limits & Artifact Separation
The runtime demo/API (`app.py`, `api.py`) continues to load the **legacy `models/` artifacts** (`conformal_q90 = 4.9673 HRC` from unweighted `1466` OOF residual pooling, vs `5.08 HRC` in V3-A and `4.98 HRC` in `phase7a`), which must **not** be confused with P1 `proper_train` models. Across both historical OOF and new P1 held-out calibration (`87.24%`), there is **no independent external validation and no theoretical 90% coverage guarantee** for unseen steel groups (severe under-coverage occurs in families such as `AISI-SAE 9264`, `0.31%C`, `AISI-SAE 4640`/`5160`, `Nitriding Steel`, and `E52100`).

## 5. Candidate Tempering Condition Recommender (`/recommend`)
`/recommend` grids `T` (`100–704 °C`, `0.5 °C` numerical scan step across tree splits) and 4 discrete durations (`1800, 3600, 7200, 14400 s`), matching the model's **point prediction** within `±0.5 HRC` (separate from the `~±5 HRC` empirical band) and prioritizing shortest `t` then lowest `T` heuristically (not energy/cost or embrittlement optimization). Out-of-bounds chemistry is rejected with `recommendable = False` (`OUT_OF_OBSERVED_SCOPE`); `NO_CANDIDATE_FOUND` denotes no grid match within `±0.5 HRC` rather than physical impossibility. In-box results still lack verified multivariate support.

## 6. Documentation Map
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) · [MODEL_CARD.md](MODEL_CARD.md) · [DATA_SHEET.md](DATA_SHEET.md)
- [EVALUATION.md](EVALUATION.md) · [COMMERCIAL_PILOT.md](COMMERCIAL_PILOT.md) · [LIMITATIONS.md](LIMITATIONS.md)
- [cleaning_decisions.md](cleaning_decisions.md) · [data_dictionary.md](data_dictionary.md) · [methodology.md](methodology.md)
- `reports/phase*.md`: Historical phase outputs (with post-audit correction notes where needed).
