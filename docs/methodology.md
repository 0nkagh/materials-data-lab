# Materials Data Lab: Analytical Methodology & Pipeline Architecture

This document synthesizes the empirical approach, data quality pipeline, and modeling evolution of the Materials Data Lab project. It serves as the definitive guide to how the project addresses the metallurgical relationship between steel composition, tempering conditions, and resulting hardness (HRC).

---

## 1. Data Provenance & Integrity
- **Dataset:** "Tempering data for carbon and low alloy steels" (Raiipa Technologies, Kaggle: `rgerschtzsauer`).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Core Literature Sources:**
  - Hollomon & Jaffe (1945): Foundational time-temperature equivalence in tempering.
  - Grange & Baughman (1956): Hardness measurements of tempered martensite.
  - Penha (2010) / Kang & Lee (2014): Kinetic formulations of tempering hardness.
- **Immutability:** The raw CSV file is stored locally, excluded from version control, and accessed strictly in read-only mode via a hash-pinned loader to prevent data fabrication or accidental corruption.

---

## 2. Quality-First Pipeline & Decision Log
The project enforces a strict "data quality first" pipeline. Anomalies are not silently deleted; they are flagged using a custom taxonomy and handled via explicit, recorded decisions.

| Decision ID | Summary of Action / Rule | Status |
| :--- | :--- | :--- |
| **D-01** | Missing `initial_hrc` values (literal `?`) are parsed as `NaN`. | `APPROVED` |
| **D-02** | Values < 20 HRC (ASTM E140 extrapolation suspect) are kept but flagged. | `APPROVED` |
| **D-03** | High Al (>0.15% wt) is flagged as potential Nitralloy (intentional) and kept. | `APPROVED` |
| **D-04** | High P (>0.05% wt) is flagged as impurity suspect but kept. | `APPROVED` |
| **D-05** | Missing `steel_type` is filled with `UNKNOWN_GRADE`. | `APPROVED` |
| **D-06** | Logical duplicates (same composition & conditions) are kept to preserve variance. | `APPROVED` |
| **D-07** | Column names are mapped to safe slugs via a hardcoded dictionary. | `APPROVED` |
| **NK-1/2** | String whitespace stripped; Fahrenheit conversion traces logged `[REPORT_ONLY]`. | `APPROVED` |
| **D-08** | `initial_hrc` is omitted from the main MVP to avoid dropping 64% of rows. | `APPROVED` |
| **D-09/11** | Hyperparameter tuning must be leakage-free using `GroupKFold(5, steel_type)`. | `APPROVED` |
| **D-10** | Champion models are selected based strictly on GroupKFold (S2) performance. | `APPROVED` |
| **D-12/14** | A new model/feature becomes champion only if S2 MAE improves by >0.15 HRC. | `APPROVED` |

*(Detailed records can be found in `docs/cleaning_decisions.md` and Phase 2/3 reports).*

---

## 3. Modeling Protocol
To prevent optimistic evaluation bias caused by identical steel samples appearing in both training and testing sets, models are evaluated under two splits:
- **S1 (RandomSplit):** A standard 80/20 split. Prone to data leakage across rows of the same steel type.
- **S2 (GroupSplit / GroupKFold):** Cross-validation grouped by `steel_type`. Simulates performance on entirely *unseen* steel grades. **S2 is the sole metric for champion selection (D-10).**

---

## 4. Physics Evolution Study (V2-E)
The classical Hollomon-Jaffe (HJ) parameter models tempering hardness using a constant $C$ (often assumed as 19.5). In Phase V2-E, inspired by Kang & Lee (2014), we tested a composition-dependent `Composite_C` formulation.

- **Performance Gain:** The `Composite_C` formulation improved the physical baseline's S2 MAE from **4.67 HRC** to **3.34 HRC**.
- **Metallurgical Consistency [REPORT_ONLY]:** The learned coefficients for strong carbide formers and hardenability elements (V, C, Mo, Si, Cr) were all strongly positive, correctly reflecting their physical role in delaying softening (increasing tempering resistance).
- **Collinearity Caveat [REPORT_ONLY]:** Because the model acts as a closed-form linear interaction design regressed directly on HRC, recovering a single absolute 'effective C' is unidentifiable due to collinearity between the time term and intercept. However, the predictive CV results and directional trends remain entirely valid. *(See `reports/phase_v2e_hj_evolution.md`)*.

---

## 5. The Champion: Tuned XGBoost
Following the V2-C2 Challenger Study, a tuned **XGBoost Regressor** defeated the MVP Random Forest and was crowned the champion model.
- **Performance:** S2 MAE of **2.35 ± 0.39 HRC** (R² = 0.94).
- **Explainability:** SHAP (Shapley Additive exPlanations) analysis confirmed that tempering temperature and time dominate the model's decisions, while alloy compositions (C, Mo, Cr, Si) act as secondary adjusters. This completely aligns with both MDI and permutation importance rankings. *(See `reports/phase_v2c2_compare.md`)*.
- **Feature Addition:** Attempting to feed the classical $p_{hj}$ feature into the XGBoost model yielded a marginal gain of ~0.046 HRC, failing the D-14 threshold (>0.15 HRC). The champion remained unmodified.

---

## 6. Recommender System & Inverse Optimization (V5-A)
The project includes a local "Recipe Recommender" to perform inverse optimization: finding the optimal tempering temperature and time for a given target HRC and composition.
- **Methodology (Grid Scan):** Because tree-based models (Random Forest, XGBoost) create step-wise predictions rather than continuous gradients, gradient-based optimization tools (like `scipy.optimize` or `optuna`) fail. We use a brute-force grid scan over realistic domains ($T \in [100, 705]^\circ\text{C}$ at $0.5^\circ\text{C}$ resolution, $t \in [1800, 3600, 7200, 14400]$ seconds).
- **Physical Constraints:** The system returns the shortest possible time and lowest temperature within a $\pm 0.5$ HRC tolerance. It enforces data support boundaries (extrapolation blocks) and rejects unachievable targets with a `NOT_ACHIEVABLE` flag.
- **Validation:** Tests on 5 real steel grades (e.g., 4140, 1045, 4340) confirmed the system successfully identifies achievable recipes well within the $\pm 0.5$ HRC threshold across targets from 30 to 50 HRC.

---

## 7. Limitations & Vulnerabilities
- **Single Dataset Bias:** The model is trained on a single aggregated literature dataset. Real-world industrial validation is required before any physical use.
- **Weak Spot (Si-Alloys):** Both RF and XGBoost models show significant predictive weakness on Silicon-alloy steels (e.g., 9260 type), exhibiting high MAE (up to ~9.2 HRC). Silicon's complex role in delaying cementite precipitation is mathematically under-captured by the current feature space.
- **Extrapolation Risk:** Predictions below 20 HRC or above 650°C enter the dataset's extrapolation zones and should be treated as highly suspect.

---

## 8. Reproducibility & Deployment
- **Determinism:** All data splits and model trainings enforce `random_state=42`.
- **Artifact Schema (v2):** The trained model is serialized alongside a `meta.json` file (Schema v2) that hashes the raw CSV and validates feature ordering. If the schema is outdated or the CSV hash mismatches, the system gracefully forces a retraining event.
- **Continuous Integration (CI):** 39 strict `pytest` unit tests run on every push via GitHub Actions.
- **Local Demo:** Users can launch a local Gradio interface running the Champion XGBoost model by simply executing `python app.py` (which seamlessly handles missing artifacts through automatic background training).
