# Materials Data Lab: Project Overview

## 1. Problem Statement
The heat treatment of carbon and low-alloy steels requires determining precise tempering temperatures ($T$) and times ($t$) to achieve a specific target hardness (HRC). Traditional methods rely heavily on trial-and-error laboratory testing or static historical charts. This project addresses the need for a data-driven prototype capable of mapping steel alloy composition to post-tempering hardness and providing automated, uncertainty-aware recipe recommendations. 

## 2. Approach
The project employs a dual baseline approach: a purely physical baseline using the Hollomon-Jaffe (1945) parameter, and a set of machine learning models trained on empirical composition and treatment data. A data-quality-first approach was utilized to clean, audit, and structure the data (1466 rows, 17 columns) before training. Evaluation is built on a rigorous `GroupKFold` strategy (grouped by steel grade) to test the model's ability to generalize to unseen steel alloys.

## 3. Champion Model Summary
The final champion model is a tuned XGBoost regressor (V1.1 model architecture). 
When tested on entirely unseen steel families (Grade-Disjoint Cross-Validation), the model achieves:
- **MAE:** 2.35 ± 0.39 HRC (fold-mean aggregation) / 2.48 HRC (pooled OOF aggregation)
- **R²:** 0.94 (fold-mean aggregation)

## 4. Uncertainty & Calibration
To prevent over-confident predictions, the system uses Split-Conformal Prediction. 
Based on out-of-fold (OOF) residuals, a constant width interval of **q90 = 5.08 HRC** was derived. This guarantees a global empirical coverage of ~89.9%. However, per-family calibration analysis revealed severe under-coverage in specific chemical groups (e.g., Si-spring steels, E52100 bearing steels), which are excluded from commercial pilot testing.

## 5. Recipe Recommender
An API endpoint (`/recommend`) provides reverse-engineering capabilities. Given a composition and a target HRC, it scans a dense grid of $T$ and $t$, groups valid solutions into contiguous temperature windows (0.5°C step), and returns the top 3 recipe candidates (prioritizing shortest time, then lowest temperature). It includes strict safety boundaries, flagging Out-Of-Distribution (OOD) chemistry and "High Uncertainty" scenarios.

## 6. Documentation Map (Single Source of Truth)
To prevent duplication, specific details are isolated in the following documents:
- **`PROJECT_OVERVIEW.md`** (This file): High-level summary of the prototype.
- **`MODEL_CARD.md`**: Intended use, out-of-scope use, quantitative evaluation, and failure modes.
- **`DATA_SHEET.md`**: Dataset origin, URL, SHA hashes, shape, caveats, and licensing.
- **`EVALUATION.md`**: Complete evaluation history (baselines, ablation, leakage, calibration metrics).
- **`COMMERCIAL_PILOT.md`**: Pre-registered protocol for pilot testing in a real laboratory environment.
- **`LIMITATIONS.md`**: Consolidated list of all known boundaries and constraints.
- **`cleaning_decisions.md`**: Log of all rules applied during data cleaning and preprocessing.
- **`data_dictionary.md`**: Metadata and column descriptions for the raw dataset.
- **`methodology.md`**: The scientific narrative and chronological phase decisions (V1-V6).
- **`reports/phase*.md`**: Auto-generated evaluation tables directly sourced from the codebase.
