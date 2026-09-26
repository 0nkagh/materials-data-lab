# Consolidated Limitations

This document lists all established constraints, blind spots, and known weaknesses of the Materials Data Lab prototype. Every limitation is grounded in the analytical reports and validation tests.

1. **Missing Pre-Temper State (As-Quenched Hardness)**
   - **Evidence:** `docs/DATA_SHEET.md` and Phase 6a (`reports/phase6a_subset.md`).
   - **Details:** 65% of the dataset lacks `initial_hrc` values. A model trained on the subset with `initial_hrc` (Phase 6a) showed observational improvements, proving that the omission of this feature in the global model introduces a ceiling to the predictive accuracy.

2. **Missing Metallurgical Context (Quench Medium & Part Geometry)**
   - **Evidence:** `docs/DATA_SHEET.md`
   - **Details:** The training data does not specify whether the steel was water-quenched, oil-quenched, or air-cooled, nor does it define the physical dimensions (mass effect/hardenability depth). The model essentially predicts an "ideal" laboratory response.

3. **Out-of-Distribution (OOD) Extrapolation Flatlining**
   - **Evidence:** `reports/phase7d_calibration.md` and API logic (`src/materials_data_lab/api.py`).
   - **Details:** Tree-based models (like the champion XGBoost) cannot mathematically extrapolate beyond the minimum/maximum bounds seen in training. They output a flat, constant leaf value when queried with temperatures, times, or chemistries outside their learned scope, leading to physically impossible bounds if unchecked.

4. **Static Conformal Band Over-Confidence**
   - **Evidence:** `reports/phase7d_calibration.md`
   - **Details:** The ±5.08 HRC conformal interval is static. It does not dynamically widen when predicting inputs that are far from the training distribution center (OOD regions), leading to potentially over-confident bounds.

5. **Severe Family-Specific Under-Coverage**
   - **Evidence:** Wilson 95% CI Annotation in `reports/phase7d_calibration.md`.
   - **Details:** While global coverage is ~89.9%, five specific chemical groups exhibit mathematically proven under-coverage: AISI-SAE 9264 (Si-spring), 5160 (Si-spring), E52100 (bearing), Nitriding Steel, and 0.31%C plain carbon steel.

6. **End-to-End Environment Validation**
   - **Evidence:** `docs/EVALUATION.md`
   - **Details:** Validation relies entirely on hermetic, in-memory `pytest` and `TestClient` scripts. A fully deployed E2E live server integration test is outside the scope of this prototype's validation boundaries.
