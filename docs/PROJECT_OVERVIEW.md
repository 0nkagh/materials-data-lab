# Materials Data Lab: Project Overview

## 1. Problem Statement
Heat treatment of carbon and low-alloy steels requires selecting tempering temperatures (`T`) and soaking times (`t`) toward a target hardness (`HRC`). This research prototype explores how well tabular models trained on an aggregated literature dataset predict post-tempering hardness and flag boundary/uncertainty risks before physical laboratory trials.

## 2. Approach
The pipeline combines a rule-based physical baseline (`Hollomon-Jaffe 1945`, assumed `C = 19.5`) with tree-based models trained on `1466` cleaned literature rows (`17` raw columns, `initial_hrc` excluded due to `64.7%` missingness). Internal evaluation uses `GroupKFold(5)` grouped by the `34` `Steel type` strings within those 3 literature sources (`S2`).

## 3. Champion Model Summary (Historical Internal CV)
On the internal 5-fold `GroupKFold(steel_type)` split (shared with hyperparameter search and model selection, with no nested outer CV or external test set), the tuned XGBoost configuration achieved:
- **MAE:** `2.35 ± 0.39 HRC` (fold-mean) / `2.48 HRC` (pooled OOF)
- **R²:** `0.94`

## 4. Uncertainty Band (OOF Residual Pooling)
Predictions include a fixed-width empirical prediction band from the unweighted `0.90` quantile of pooled `1466` OOF absolute residuals (`5.08 HRC` in V3-A `phase_v3a_physics.json`, `4.9673 HRC` in `models/artifact_meta.json`, `4.98 HRC` in `phase7a_baselines.md`; exact discrepancy cause unverified without re-running). Calibration and global same-pool hit rate (`89.97%`) share the same 1466 residuals, providing **no external or per-family coverage guarantee**. Five steel groups (`9264`, `0.31%C plain carbon`, `5160` [Cr-steel, not Si-spring], `Nitriding Steel`, `E52100`) show severe empirical under-coverage and are excluded from pilot trials.

## 5. Candidate Tempering Condition Recommender (`/recommend`)
`/recommend` grids `T` (`100–704 °C`, `0.5 °C` numerical scan step across tree splits) and 4 discrete durations (`1800, 3600, 7200, 14400 s`), matching the model's **point prediction** within `±0.5 HRC` (separate from the `~±5 HRC` empirical band) and prioritizing shortest `t` then lowest `T` heuristically (not energy/cost or embrittlement optimization). Out-of-bounds chemistry is rejected with `recommendable = False` (`OUT_OF_OBSERVED_SCOPE`); `NO_CANDIDATE_FOUND` denotes no grid match within `±0.5 HRC` rather than physical impossibility. In-box results still lack verified multivariate support.

## 6. Documentation Map
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) · [MODEL_CARD.md](MODEL_CARD.md) · [DATA_SHEET.md](DATA_SHEET.md)
- [EVALUATION.md](EVALUATION.md) · [COMMERCIAL_PILOT.md](COMMERCIAL_PILOT.md) · [LIMITATIONS.md](LIMITATIONS.md)
- [cleaning_decisions.md](cleaning_decisions.md) · [data_dictionary.md](data_dictionary.md) · [methodology.md](methodology.md)
- `reports/phase*.md`: Historical phase outputs (with post-audit correction notes where needed).
