# Commercial Pilot Protocol (Pre-Registered Research Screening)

## A. Commercial Summary
- **The Problem:** Heat-treatment trial planning seeks starting tempering parameters (`T`, `t`) before laboratory coupons are tested.
- **The Beneficiary:** Metallurgical engineers and heat-treatment QC labs (with mandatory domain-expert sign-off).
- **Intended Role:** Reduces initial experimental search space by proposing **candidate processing windows** as a starting hypothesis—never an autonomous or production-ready recipe.
- **Data Footprint:** `1466` literature rows across 3 sources (`Grange 1956`, `Hollomon 1945`, `Penha 2010`).
- **Observed Internal Performance & Uncertainty Caveats:** Historical internal `GroupKFold(5)` study yielded `~2.35 HRC` fold-mean MAE (`2.48 HRC` pooled OOF) and an empirical OOF residual quantile (`±5.08 HRC` in V3-A study / `±4.9673 HRC` in `models/artifact_meta.json`). Because OOF pooling has no independent external coverage guarantee, physical acceptance must be established empirically in the lab.
- **When NOT to use:** High-alloy tool steels, multi-stage tempering, unverified or under-covered steel families (high-Si spring `9264`, Cr-steel `5160`, bearing `E52100`, `Nitriding Steel`, `0.31%C plain carbon`), out-of-scope chemistry, or any setting without human-in-the-loop laboratory testing.

## B. Pilot Protocol
The pilot moves from internal digital cross-validation to physical laboratory coupon testing.
- **Allowed Candidate Grades (based on historical OOF empirical coverage, NOT strictly single SAE prefix):**
  - Documented as `41xx` (`Cr-Mo`: `4140`, `97.8%`), `43xx` (`Ni-Cr-Mo`: `4340`, `94.1%`), and `46xx` (`Ni-Mo`: `4640`, `Cr=0.06%`, `98.0%`). *(Correction note: earlier prose grouped all three under "Cr-Mo"; `4340` is `Ni-Cr-Mo` and `4640` is `Ni-Mo` classified as `UNKNOWN_CLASS` in `steel_classes.py`.)*
- **Excluded Families:** High-Si spring (`9264`), Cr-steel (`5160` — previously mislabeled `Si-spring` in text), bearing (`E52100`), `Nitriding Steel`, and `0.31%C plain carbon` steel (`reports/phase7d_calibration.md`).
- **Process Scope:** Single-stage continuous tempering only.
- **Sample Size:** `5` to `20` physical lab coupons.
- **Execution:**
  1. Forward-Predict: Compare point predictions and empirical band hits against measured lab coupon hardness.
  2. Candidate Conditions (`/recommend`): Evaluate whether furnace runs at proposed candidate windows `(T, t)` fall within acceptable physical tolerances (accounting for both the `±0.5 HRC` grid-match tolerance and the `~±5 HRC` empirical band).

## C. Pre-Registered Acceptance Criteria
1. **In-Family Physical MAE:** `≤ 3.0 HRC` error average on physical lab coupons.
2. **Physical Band Hit Rate:** `≥ 85%` of physical coupon measurements fall within the model's empirical `±q90` band (`±5.08 HRC` study benchmark / `±4.9673 HRC` artifact value; state clearly which artifact is evaluated).
3. **Candidate Round-Trip Physical Consistency:** Physical hardness measured at recommended candidate `(T, t)` is within `≤ 0.5 HRC` of the predicted candidate hardness (or document empirical deviation if `0.5 HRC` reflects grid-step precision rather than furnace repeatability).

## D. Pilot Results
`DATA_NOT_AVAILABLE`
*(This section remains strictly empty until physical laboratory measurements are recorded and integrated).*

## E. Limitations Binding Any Commercial Scale-Up
- **Out-of-Scope & Unverified Multivariate Chemistry:** 1D min–max bounds do not verify multivariate support; unvalidated grades require separate calibration.
- **Missing `initial_hrc`, Quench Medium, Geometry & Cost/Embrittlement Constraints:** Variations in as-quenched state, oil vs. water quenching, section mass effects, and temper embrittlement (`250–575 °C`) are unmodeled.
- **Domain Shift & Human Oversight:** Furnace calibration and thermocouple lag can shift lab hardness systematically; a qualified metallurgist must authorize every physical trial and batch.
