# Consolidated Limitations

This document lists established boundaries, methodological limits, and blind spots of the Materials Data Lab research prototype.

1. **Missing Pre-Temper State (`initial_hrc`)**
   - **Evidence:** [DATA_SHEET.md](DATA_SHEET.md) and `reports/phase6a_initial_hrc.md`.
   - **Details:** `949/1466` rows (`64.7%`, entire Grange 1956 subset) lack `initial_hrc`. Omitting `initial_hrc` prevents modeling variations in as-quenched hardness; the Phase 6a subset study (`n=517`, `11` groups) is purely observational with selection bias.

2. **Missing Metallurgical & Experimental Identity Metadata**
   - **Evidence:** [DATA_SHEET.md](DATA_SHEET.md) and `data/raw` CSV audit.
   - **Details:** Quenching medium (water/oil/air), austenitizing conditions, sample geometry/section thickness, and heat/series identifiers are absent. `GroupKFold(groups=steel_type)` therefore measures grade-string holdout within the 3 literature compilations, not independent cross-laboratory generalization.

3. **Non-Nested CV Model Selection & Missing Baseline Generators**
   - **Evidence:** `tuning.py`, `compare_models.py`, `v3_physics.py`, `ablation_study.py`, [EVALUATION.md](EVALUATION.md).
   - **Details:** Hyperparameter search, champion selection, and `S2` metrics share the same 34-group `GroupKFold(5)` partition (no nested outer CV). Baseline scripts for `phase7a` / `phase7b` are incomplete or absent from `src/`, leaving historical `8.94` / `4.70` numbers in prior docs unverified.

4. **Out-of-Distribution (OOD) Flatlining and 1D-Only Support Checks**
   - **Evidence:** `demo.py`, `data_bounds.py`, `recommender.py`, `api.py`.
   - **Details:** Tree models emit flat constant predictions outside observed splits. Runtime distinguishes three concepts: (a) **observed dataset range** (`OBSERVED_UNIVARIATE_BOUNDS`, 1D marginal min–max only), (b) **application supported input scope** (`SUPPORTED_INPUT_BOUNDS`, including the software cap `0 < T <= 800 °C` aligned with legacy thresholds—not a universal metallurgical impossibility limit or composition-independent austenitizing/`Ac1` threshold), and (c) **true metallurgical suitability** (composition-dependent `Ac1`/phase transformations, quenching conditions, geometry, initial state). Passing 1D or application bounds cannot detect unusual multivariate combinations or interpolation gaps; `/recommend` blocks 1D out-of-bounds chemistry (`OUT_OF_OBSERVED_SCOPE`), but in-box multivariate validity and metallurgical suitability remain unverified.

5. **OOF Empirical Prediction Band Limits & Severe Under-Coverage**
   - **Evidence:** `v3_physics.py`, `model_artifact.py`, `reports/phase7d_calibration.md`.
   - **Details:** `q90` is an unweighted `0.90` quantile of pooled OOF residuals evaluated on the **same** 1466 OOF pool (no independent calibration split or finite-sample/group correction, and an unresolved discrepancy between `5.08`, `4.9673`, and `4.98 HRC`). Width is fixed and fails badly on 5 groups: `AISI-SAE 9264` (`12.5%`), `0.31%C plain carbon` (`37.8%`), `AISI-SAE 5160` (`47.1%`, `Cr_steel` — previously mislabeled `Si-spring` in text), `Nitriding Steel` (`67.5%`), and `AISI-SAE E52100` (`74.5%`).

6. **Inverse Recommender & SHAP Interpretability Boundaries**
   - **Evidence:** `recommender.py`, `demo.py`, `docs/data_dictionary.md`.
   - **Details:** `/recommend` matches the tree model's **point prediction** within `±0.5 HRC` on a `0.5 °C` numerical scan step across `4` discrete times (`1800–14400 s`)—this reflects piecewise-constant tree splits rather than sub-degree physical accuracy, ignores the `~±5 HRC` uncertainty band, ignores temper embrittlement zones (`250–575 °C`), and uses a heuristic sort rather than cost optimization. SHAP attributions decompose model output across correlated alloy features and do **not** show causal metallurgical effects.

7. **Hermetic Test Scope & Artifact Deserialization Risk**
   - **Evidence:** [EVALUATION.md](EVALUATION.md), `model_artifact.py`.
   - **Details:** Automated tests use synthetic/mocked data without validating real physical accuracy or unpickling untrusted `.joblib` files. Runtime loading does not auto-retrain; `.joblib` artifacts use Python `pickle` and must come only from trusted local directories.
