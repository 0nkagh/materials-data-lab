"""Single source of truth for observed dataset bounds vs. application supported input scope.

Separates three distinct concepts:
1. OBSERVED_UNIVARIATE_BOUNDS: Exact marginal min/max from the 1466 clean rows of the
   immutable Raiipa dataset (SHA-256 pinned below). Univariate ranges do NOT verify
   joint multivariate support or true metallurgical suitability across combinations.
2. SUPPORTED_INPUT_BOUNDS: Software input-acceptance limits defining this application's
   supported input scope before inference (finite real numbers, 0 <= wt% <= 100 with
   alloy sum <= 100, 0 < T <= 800 C, t > 0 s). Enforcing 0 < T <= 800 C reflects the
   repository's configured input scope rather than universal metallurgical validity,
   physical impossibility, or an automatic austenitizing threshold for all steels
   (Ac1 and phase transformations depend on composition and thermal conditions).
3. True metallurgical suitability: Composition-dependent phase stability (Ac1),
   quenching conditions, initial microstructure/hardness, section size, and
   embrittlement risks, none of which are verified by passing 1D or app bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

COMP_COLS = [
    "c_wt", "mn_wt", "p_wt", "s_wt", "si_wt",
    "ni_wt", "cr_wt", "mo_wt", "v_wt", "al_wt", "cu_wt",
]
TREATMENT_COLS = ["temper_temp_c", "temper_time_s"]
ALL_FEATURE_COLS = COMP_COLS + TREATMENT_COLS

BOUNDS_PROVENANCE: dict[str, Any] = {
    "source_file": "data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv",
    "csv_sha256": "3f88fdccf4bb2f3c03be6eeca03001b4bd8a2b0de5cd4818195486182a4762b0",
    "row_scope": "All 1466 clean rows (D-01..D-07, non-NaN observed values)",
    "missing_policy": "Literal '?' parsed as NaN (na_values=['?']); univariate min/max computed over non-null values.",
    "units": {
        "composition": "%wt (mass percentage)",
        "temper_temp_c": "degC",
        "temper_time_s": "s",
        "hardness": "HRC",
    },
    "multivariate_note": (
        "1D min/max intervals check each variable marginally only and are distinct from both "
        "application supported input bounds and true metallurgical suitability; neither verifies "
        "joint multivariate support."
    ),
    "supported_scope_note": (
        "0 < T <= 800 °C and composition limits in SUPPORTED_INPUT_BOUNDS define this application's "
        "supported input scope (aligned with historical thresholds.py rules), not universal "
        "metallurgical validity, physical impossibility, or an automatic austenitizing boundary "
        "(Ac1 and phase transformations depend on composition and conditions)."
    ),
}

# Exact observed univariate bounds from Raiipa CSV (1466 rows; SHA-256 3f88fd...62b0)
OBSERVED_UNIVARIATE_BOUNDS: dict[str, tuple[float, float]] = {
    "c_wt": (0.25, 1.15),
    "mn_wt": (0.30, 1.85),
    "p_wt": (0.007, 0.054),
    "s_wt": (0.005, 0.055),
    "si_wt": (0.06, 1.62),
    "ni_wt": (0.0, 3.41),
    "cr_wt": (0.0, 1.57),
    "mo_wt": (0.0, 0.36),
    "v_wt": (0.0, 0.16),
    "al_wt": (0.0, 1.26),
    "cu_wt": (0.0, 0.08),
    "temper_temp_c": (100.0, 704.4),
    "temper_time_s": (10.0, 115200.0),
}

# Target and excluded initial_hrc reference bounds (non-NaN counts: 1466 and 517)
TARGET_AND_INITIAL_BOUNDS: dict[str, tuple[float, float]] = {
    "final_hrc": (0.9, 68.5),
    "initial_hrc": (46.5, 67.0),
}

# Application supported input scope (distinct from observed dataset range and true metallurgical suitability)
SUPPORTED_INPUT_BOUNDS: dict[str, dict[str, Any]] = {
    **{col: {"min": 0.0, "max": 100.0, "min_exclusive": False, "max_exclusive": False, "unit": "%wt"} for col in COMP_COLS},
    "temper_temp_c": {"min": 0.0, "max": 800.0, "min_exclusive": True, "max_exclusive": False, "unit": "°C"},
    "temper_time_s": {"min": 0.0, "max": None, "min_exclusive": True, "max_exclusive": False, "unit": "s"},
}

FEATURE_UNITS: dict[str, str] = {
    **{col: "%wt" for col in COMP_COLS},
    "temper_temp_c": "°C",
    "temper_time_s": "s",
}

CSV_TO_SLUG = {
    "C (%wt)": "c_wt",
    "Mn (%wt)": "mn_wt",
    "P (%wt)": "p_wt",
    "S (%wt)": "s_wt",
    "Si (%wt)": "si_wt",
    "Ni (%wt)": "ni_wt",
    "Cr (%wt)": "cr_wt",
    "Mo (%wt)": "mo_wt",
    "V (%wt)": "v_wt",
    "Al (%wt)": "al_wt",
    "Cu (%wt)": "cu_wt",
    "Tempering temperature (ºC)": "temper_temp_c",
    "Tempering time (s)": "temper_time_s",
    "Final hardness (HRC) - post tempering": "final_hrc",
    "Initial hardness (HRC) - post quenching": "initial_hrc",
}


def validate_finite_number(name: str, value: Any) -> float:
    """Validate that value is a real, non-boolean, finite float."""
    if isinstance(value, bool):
        raise ValueError(f"Invalid input type for {name}: boolean is not a valid numeric measurement.")
    if not isinstance(value, (int, float)):
        try:
            if isinstance(value, str):
                raise ValueError(f"Invalid input type for {name}: string {value!r} is not accepted; provide a finite number.")
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for {name}: {value!r}") from exc
    else:
        num = float(value)

    if not math.isfinite(num):
        raise ValueError(f"Invalid non-finite value for {name}: {value!r} (NaN/Infinity disallowed).")
    return num


def validate_supported_inputs(
    inputs: dict[str, Any],
    *,
    require_treatment: bool = True,
) -> dict[str, float]:
    """Validate that inputs lie within this application's supported input scope and finite numeric types.

    Enforcing SUPPORTED_INPUT_BOUNDS (including 0 < T <= 800 °C) defines software input acceptance;
    it is distinct from observed dataset bounds and true metallurgical suitability (Ac1 and phase
    transformations depend on steel composition and conditions).
    """
    required = ALL_FEATURE_COLS if require_treatment else COMP_COLS
    validated: dict[str, float] = {}

    for key in required:
        if key not in inputs:
            raise ValueError(f"Missing required input feature: {key}")
        num = validate_finite_number(key, inputs[key])
        spec = SUPPORTED_INPUT_BOUNDS[key]
        lo = spec["min"]
        hi = spec["max"]
        scope_suffix = (
            "(outside application supported input scope; distinct from observed dataset range "
            "and composition-dependent metallurgical suitability)."
        )
        if lo is not None:
            if spec["min_exclusive"] and num <= lo:
                raise ValueError(f"Input outside application supported scope for {key}: {num} {spec['unit']} must be > {lo} {scope_suffix}")
            if not spec["min_exclusive"] and num < lo:
                raise ValueError(f"Input outside application supported scope for {key}: {num} {spec['unit']} must be >= {lo} {scope_suffix}")
        if hi is not None:
            if spec["max_exclusive"] and num >= hi:
                raise ValueError(f"Input outside application supported scope for {key}: {num} {spec['unit']} must be < {hi} {scope_suffix}")
            if not spec["max_exclusive"] and num > hi:
                raise ValueError(f"Input outside application supported scope for {key}: {num} {spec['unit']} must be <= {hi} {scope_suffix}")
        validated[key] = num

    alloy_sum = sum(validated[c] for c in COMP_COLS)
    if alloy_sum > 100.0 + 1e-9:
        raise ValueError(
            f"Input outside application supported scope: sum of alloying elements ({alloy_sum:.4f} %wt) exceeds 100 %wt."
        )

    return validated


def validate_target_hrc(value: Any) -> float:
    """Validate target hardness input (finite real number in [0.0, 100.0] HRC)."""
    num = validate_finite_number("target_hrc", value)
    if num < 0.0 or num > 100.0:
        raise ValueError(f"Invalid target_hrc: {num} HRC must be within [0.0, 100.0] HRC.")
    return num


def check_observed_bounds(
    inputs: dict[str, float],
    *,
    keys: list[str] | tuple[str, ...] | None = None,
) -> tuple[bool, list[str]]:
    """Check inputs against observed 1D min/max dataset bounds.

    Note: Observed dataset range, application supported input scope, and true
    metallurgical suitability are distinct concepts; 1D bounds do not verify
    joint multivariate support or metallurgical suitability.
    """
    target_keys = list(keys) if keys is not None else [k for k in ALL_FEATURE_COLS if k in inputs]
    oob_details: list[str] = []

    for key in target_keys:
        if key not in OBSERVED_UNIVARIATE_BOUNDS or key not in inputs:
            continue
        val = float(inputs[key])
        lo, hi = OBSERVED_UNIVARIATE_BOUNDS[key]
        unit = FEATURE_UNITS.get(key, "")
        if val < lo or val > hi:
            oob_details.append(f"{key}={val:g} {unit} outside observed [{lo:g}, {hi:g}]".strip())

    return len(oob_details) == 0, oob_details


def derive_bounds_from_csv(csv_path: str | Path) -> dict[str, Any]:
    """Compute exact univariate observed bounds directly from a CSV file (lazy pandas import)."""
    import pandas as pd

    path = Path(csv_path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)

    df = pd.read_csv(path, keep_default_na=True, na_values=["?"], encoding="utf-8")
    bounds: dict[str, dict[str, Any]] = {}
    for raw_col, slug in CSV_TO_SLUG.items():
        if raw_col not in df.columns:
            raise ValueError(f"Expected column missing in CSV: {raw_col}")
        series = pd.to_numeric(df[raw_col], errors="coerce").dropna()
        bounds[slug] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "non_null_count": int(len(series)),
            "missing_count": int(len(df) - len(series)),
        }

    return {
        "csv_path": str(path),
        "csv_sha256": h.hexdigest(),
        "n_rows": int(len(df)),
        "bounds": bounds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and compare observed CSV bounds against data_bounds.py")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv",
        help="Path to raw CSV dataset",
    )
    args = parser.parse_args(argv)
    report = derive_bounds_from_csv(args.csv)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
