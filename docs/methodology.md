# Materials Data Lab: Analytical Methodology & Pipeline Architecture

This document summarizes the empirical workflow, data-quality rules, modeling evolution, and prototype boundaries of Materials Data Lab.

---

## 1. Data Provenance & Integrity
- **Dataset:** "Tempering data for carbon and low alloy steels" (Raiipa Technologies, Kaggle: `rgerschtzsauer`).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Core Literature Sources:** Hollomon & Jaffe (1945), Grange & Baughman (1956), Penha (2010, ambiguous citation in dataset card).
- **Immutability:** The raw CSV (`1466×17`, `SHA-256 = 3f88fd...62b0`) is stored locally, excluded from Git, and accessed read-only. Row-level cleaning and feature transforms (`log_time`, `p_hj`) use no cross-row fitting statistics.

---

## 2. Quality-First Pipeline & Decision Log
Anomalies are preserved and flagged rather than silently deleted (`D-01..D-24`, `docs/cleaning_decisions.md`):
- Missing `initial_hrc` (`?`, `949/1466` rows) is parsed as `NaN` (`D-01`), dropped from main MVP features (`D-08`), and not synthetically imputed (`D-15`).
- `<20 HRC` (`123` rows, `E140_EXTRAPOLATION_SUSPECT`), `Al > 0.15 %wt` (`40` rows), and `P > 0.05 %wt` (`17` rows) remain flagged (`D-02..D-04`).
- `GroupKFold(5)` uses `.str.strip()` cleaned `Steel type` strings (`34` grades; `0` exact composition overlap between different names, though neighboring chemistries and same-literature series span folds; `heat ID`, quench medium, and geometry are absent).

---

## 3. Modeling Protocol & Selection Scope
- **`S1 (RandomSplit)`:** Standard `80/20` split (`random_state=42`) — intentionally documented as leaky/optimistic across rows of the same steel grade.
- **`S2 (GroupKFold(5, groups=steel_type))`:** Grade-string holdout within the 3 literature compilations. Note: Hyperparameter tuning (`RandomizedSearchCV`), champion comparison (`D-12`), and feature/constraint ablation (`D-14/17/18`) share this same 5-fold partition (no nested outer CV or external test set). Because `GroupKFold` fold sizes are unequal, both `fold-mean` and `pooled OOF` metrics are reported (`D-24`).

---

## 4. Physics Baseline Study (`V2-E`)
Regressing a composition-linear interaction design (`Composite_C`) improved historical `S2` fold-mean MAE from `4.67 HRC` (`B2`, assumed `C=19.5`) to `3.34 HRC` (`Composite_C`), though individual element weights are unidentifiable due to collinearity (`[REPORT_ONLY]`).

---

## 5. Champion Configuration (`V2-C2`) & Attribution Caveat
- **Internal `S2` Benchmark:** Tuned `XGBRegressor` achieved `2.35 ± 0.39 HRC` fold-mean MAE (`2.48 HRC` pooled OOF, `R² = 0.94`), outperforming tuned RF (`2.64 ± 0.24 HRC`) under `D-12` (`>0.15 HRC` rule). Feeding `p_hj`, monotonic constraints, or engineered metallurgical proxies failed to clear `>0.15 HRC` (`D-14/17/18`).
- **SHAP Interpretability Caveat:** Tree SHAP decomposes **model predictions** across correlated alloying features (`temper_temp_c`, `log_time`, `c_wt`, `cr_wt`, `si_wt`, `mo_wt`); SHAP bars must **not** be read as isolated physical/causal metallurgical effects.

---

## 6. Empirical OOF Uncertainty Band (`D-19`) & Candidate Condition Search (`V5-A`)
- **Uncertainty Band (`OOF residual pooling`):** `q90` is taken as `np.quantile(|y - oof_preds|, 0.90)` across all `1466` OOF residuals and evaluated on the same pool (`~89.97%` hit rate; no independent holdout or finite-sample/group correction). Recorded `q90` differs across V3-A (`5.080 HRC`), `models/artifact_meta.json` (`4.9673 HRC`), and `phase7a` (`4.98 HRC`) — discrepancy source unverified without re-running. Five steel groups exhibit severe under-coverage (`reports/phase7d_calibration.md`).
- **Recommender (`/recommend`):** Performs a brute-force grid scan (`T ∈ [100, 704] °C` at `0.5 °C` numerical step across piecewise-constant tree splits; `t ∈ {1800, 3600, 7200, 14400} s`) to return **candidate tempering conditions requiring lab validation** (not optimal production recipes). Chemistry outside `OBSERVED_UNIVARIATE_BOUNDS` returns `OUT_OF_OBSERVED_SCOPE` with zero candidates; `±0.5 HRC` applies only to the point prediction (not the `~±5 HRC` uncertainty band); `(t_s, T_c)` sorting is heuristic and ignores furnace energy cost and temper embrittlement (`250–575 °C`); `NO_CANDIDATE_FOUND` reflects grid/model limits rather than physical impossibility.

---

## 7. Reproducibility, Artifact Safety & Testing
- **Load-Only Runtime (`app.py`, `api.py`):** Runtime entrypoints call `load_artifact` and **never** trigger automatic retraining or overwrite `models/` on missing/failing pickles. Explicit training requires `python -m materials_data_lab.model_artifact --csv ... [--overwrite]`.
- **Tests & CI:** `14` hermetic test files (`56` test cases; historical earlier note mentioned `39`) verify mocked/synthetic logic (`pytest`, `ruff check .`). Runtime boundary definitions live solely in `src/materials_data_lab/data_bounds.py`.
