# Commercial Pilot Protocol

## A. Commercial Summary
- **The Problem:** Heat treatment foundries spend significant time and energy on laboratory trial-and-error to find the correct tempering recipe to reach a target hardness.
- **The Beneficiary:** Metallurgical engineers, quality control labs, and heat treatment process designers.
- **The Acceleration:** Reduces the initial search space for tempering parameters, providing a strong mathematical starting point and candidate recipe windows.
- **Data Footprint:** Trained on 1466 historical data points representing carbon and low alloy steels.
- **Reliability:** Statistically evaluated to a ~2.35 HRC error margin on unseen steel grades, backed by a 90% conformal uncertainty interval.
- **When NOT to use:** Cannot be used for high-alloy tool steels, unverified families (e.g., Si-spring steels), multi-stage tempering, or without mandatory human-in-the-loop laboratory verification.
- **The Pilot:** A strictly scoped, physical laboratory validation sequence designed to measure real-world prototype viability against the digital baseline.

## B. Pilot Protocol
The pilot phase moves the prototype from digital cross-validation to physical laboratory evaluation.
- **Allowed Family:** Cr-Mo steels exclusively. (Evidence from `reports/phase7d_calibration.md`: AISI-SAE 41xx / 43xx / 46xx exhibit stable and verified coverage of 94–98%).
- **Excluded Families:** Si-spring (9264, 5160), Bearing (E52100), Nitralloy, and ultra-low carbon plain steels.
- **Process Scope:** Single-stage continuous tempering only.
- **Sample Size:** 5 to 20 physical lab coupons.
- **Execution:** 
  1. Forward-Predict: Generate predictions for known lab conditions, comparing predicted vs. actual hardness. Calculate interval coverage.
  2. Reverse-Recommend: Use the `/recommend` endpoint to request a recipe for a target HRC. Run the laboratory furnace at the recommended candidate window (T, t). Compare the final physical HRC to the target.

## C. Pre-Registered Acceptance Criteria
To proceed beyond the pilot phase, the following physical thresholds must be met:
1. **In-Family MAE:** ≤ 3.0 HRC error average on the physical coupons.
2. **Interval Coverage:** ≥ 85% of physical results must fall within the model's ±5.08 HRC conformal band.
3. **Recipe Round-Trip Accuracy:** The physical hardness produced by the recommended recipe candidate must be within ≤ 0.5 HRC of the predicted candidate hardness.

## D. Pilot Results
`DATA_NOT_AVAILABLE`
*(This section remains strictly empty until physical laboratory measurements are recorded and integrated).*

## E. Success Metrics & Commercial Limitations
While the acceptance criteria govern the pilot's success, scaling up commercially is strictly bound by these fundamental limitations:
- **Out-Of-Scope Families:** Applying the model outside the validated Cr-Mo family requires a completely new validation protocol.
- **Missing `initial_hrc`:** The lack of pre-temper hardness (as-quenched HRC) in the training data means variations in the initial quench quality cannot be modeled.
- **Missing Quench/Geometry Context:** The model assumes a standardized metallurgical quench state. Differences in cooling mediums (oil vs. water) and cross-sectional part thicknesses (mass effect) are blind spots.
- **Domain Shift:** Furnace calibrations, thermocouple lag, and lab-specific practices can cause systematic shifts between the Kaggle dataset and a specific facility.
- **Human Oversight:** The model provides a mathematical hypothesis. A qualified metallurgical engineer must authorize every production batch.
