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
- **Tests & CI:** Hermetic tests (`pytest`, `ruff check .`) verify synthetic/mocked behavior without hitting raw CSV or real `.joblib` pickles. Runtime boundary definitions live in `src/materials_data_lab/data_bounds.py`.

---

## 8. P1 Unbiased Grouped Evaluation & Reproducible Reporting
- **Protocol (`p1_evaluation.py`):** Uses outer `GroupKFold(5)`, target-blind `proper_train` (`20–21` groups) vs. `calibration` (`7` groups) separation (`seed=42`), inner `GroupKFold(3)` search (`20` deterministic candidates), single refit on `proper_train`, and finite-sample `ceil((n + 1) * (1 - alpha))` calibration (`overflow_policy="inf"`).
- **Results (`reports/p1_grouped_evaluation/`):** `XGBoost` pooled MAE `2.9842 HRC` (`2.9838 ± 0.6322` fold-mean, `3.0450` group-macro), `B2_Physics` pooled MAE `4.6943 HRC`, global coverage `87.24%` (`87.12%` group-macro), mean full band width `11.4433 HRC`. This gap relative to legacy `2.35 / 2.48` stems jointly from `-25%` smaller training groups (`proper_train`), differing inner search budgets, and held-out calibration—not historical optimism alone.
- **Artifact Separation:** Demo/API (`app.py`, `api.py`) continues to serve legacy pre-built `models/` artifacts (`conformal_q90 = 4.9673 HRC`); P1 metrics do not replace those weights and still carry **no 90% group-level guarantee or external dataset validation**.
- **Reporting Script (`p1_reporting.py`):** Regenerates `summary.json`, `README.md`, and PNG charts from saved local P1 outputs. Per-row `predictions.csv`, reproducible `split_manifest.json`, and local OS `run_manifest.json` are omitted from Git because publishing them is optional here, source-rights provenance over underlying literature compilations is not separately cleared (distinct from general CC BY 4.0 attribution), and they can be regenerated locally from the verified source CSV.
