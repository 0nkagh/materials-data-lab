"""Tests for single-source data bounds, application supported input scope, and recommender scope policy."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from materials_data_lab.data_bounds import (
    BOUNDS_PROVENANCE,
    OBSERVED_UNIVARIATE_BOUNDS,
    SUPPORTED_INPUT_BOUNDS,
    check_observed_bounds,
    derive_bounds_from_csv,
    validate_supported_inputs,
    validate_target_hrc,
)
from materials_data_lab.recommender import recommend_recipe


def _base_4140() -> dict[str, float]:
    return {
        "c_wt": 0.40,
        "mn_wt": 0.85,
        "p_wt": 0.015,
        "s_wt": 0.015,
        "si_wt": 0.25,
        "ni_wt": 0.0,
        "cr_wt": 1.0,
        "mo_wt": 0.20,
        "v_wt": 0.0,
        "al_wt": 0.0,
        "cu_wt": 0.0,
        "temper_temp_c": 450.0,
        "temper_time_s": 3600.0,
    }


def test_observed_bounds_provenance_and_real_dataset_consistency():
    assert BOUNDS_PROVENANCE["csv_sha256"] == "3f88fdccf4bb2f3c03be6eeca03001b4bd8a2b0de5cd4818195486182a4762b0"
    assert BOUNDS_PROVENANCE["row_scope"]
    assert BOUNDS_PROVENANCE["missing_policy"]
    assert "Ac1" in BOUNDS_PROVENANCE["supported_scope_note"]
    assert "application" in BOUNDS_PROVENANCE["supported_scope_note"].lower()
    assert SUPPORTED_INPUT_BOUNDS["temper_temp_c"]["max"] == 800.0

    # High-carbon, high-Ni, high-Cr, nitriding, and low-Si steel chemistries present in dataset
    for sample_override in (
        {"c_wt": 0.80, "mn_wt": 0.75, "cr_wt": 0.0, "mo_wt": 0.0},  # 1080
        {"ni_wt": 3.41, "c_wt": 0.40},  # 2340
        {"c_wt": 1.02, "cr_wt": 1.42},  # E52100
        {"al_wt": 1.26, "cr_wt": 1.57},  # Nitriding
        {"si_wt": 0.06},  # low-Si plain carbon
    ):
        row = _base_4140()
        row.update(sample_override)
        within, oob = check_observed_bounds(row)
        assert within is True, f"Expected in-bounds for {sample_override}, got {oob}"

    # Verify against actual local CSV when present on disk (skipped nowhere)
    raw_csv = Path("data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv")
    if raw_csv.is_file():
        derived = derive_bounds_from_csv(raw_csv)
        assert derived["csv_sha256"] == BOUNDS_PROVENANCE["csv_sha256"]
        for feat, (lo, hi) in OBSERVED_UNIVARIATE_BOUNDS.items():
            assert derived["bounds"][feat]["min"] == pytest.approx(lo)
            assert derived["bounds"][feat]["max"] == pytest.approx(hi)


def test_derive_bounds_from_synthetic_csv_and_missing_policy(tmp_path: Path):
    csv_path = tmp_path / "synth_bounds.csv"
    df = pd.DataFrame({
        "Source": ["S1", "S2"],
        "Steel type": ["T1", "T2"],
        "Initial hardness (HRC) - post quenching": ["?", "58.0"],
        "Tempering time (s)": [120, 7200],
        "Tempering temperature (ºC)": [200.0, 650.0],
        "C (%wt)": [0.30, 0.90],
        "Mn (%wt)": [0.40, 1.20],
        "P (%wt)": [0.010, 0.030],
        "S (%wt)": [0.010, 0.025],
        "Si (%wt)": [0.10, 1.50],
        "Ni (%wt)": [0.0, 2.50],
        "Cr (%wt)": [0.0, 1.40],
        "Mo (%wt)": [0.0, 0.30],
        "V (%wt)": [0.0, 0.12],
        "Al (%wt)": [0.0, 0.05],
        "Cu (%wt)": [0.0, 0.04],
        "Final hardness (HRC) - post tempering": [55.0, 22.0],
    })
    df.to_csv(csv_path, index=False)

    derived = derive_bounds_from_csv(csv_path)
    assert derived["n_rows"] == 2
    assert derived["bounds"]["initial_hrc"]["non_null_count"] == 1
    assert derived["bounds"]["initial_hrc"]["missing_count"] == 1
    assert derived["bounds"]["c_wt"]["min"] == 0.30
    assert derived["bounds"]["c_wt"]["max"] == 0.90


def test_exact_boundary_and_out_of_bounds_edges():
    # Exact observed min and max across all 13 features must pass 1D bounds
    min_row = {k: bounds[0] for k, bounds in OBSERVED_UNIVARIATE_BOUNDS.items()}
    max_row = {k: bounds[1] for k, bounds in OBSERVED_UNIVARIATE_BOUNDS.items()}
    assert check_observed_bounds(validate_supported_inputs(min_row))[0] is True
    assert check_observed_bounds(validate_supported_inputs(max_row))[0] is True

    # Out-of-bounds edges (including previously loose Mo=0.50, Cu=0.20, C=0.20)
    for bad_key, bad_val in (
        ("c_wt", 0.249),
        ("c_wt", 1.151),
        ("mo_wt", 0.361),
        ("cu_wt", 0.081),
        ("mn_wt", 0.299),
        ("s_wt", 0.0049),
        ("temper_temp_c", 704.5),
        ("temper_time_s", 9.9),
    ):
        row = _base_4140()
        row[bad_key] = bad_val
        within, details = check_observed_bounds(validate_supported_inputs(row))
        assert within is False
        assert any(bad_key in d for d in details)


@pytest.mark.parametrize(
    "field,bad_val",
    [
        ("c_wt", float("nan")),
        ("temper_temp_c", float("inf")),
        ("temper_time_s", float("-inf")),
        ("c_wt", True),
        ("si_wt", "0.25"),
        ("c_wt", -0.01),
        ("c_wt", 100.1),
        ("temper_temp_c", 0.0),
        ("temper_temp_c", -50.0),
        ("temper_temp_c", 800.1),
        ("temper_time_s", 0.0),
        ("temper_time_s", -10.0),
    ],
)
def test_supported_input_validation_rejects_invalid_inputs(field, bad_val):
    row = _base_4140()
    row[field] = bad_val
    with pytest.raises(ValueError) as exc_info:
        validate_supported_inputs(row)
    if isinstance(bad_val, (int, float)) and not isinstance(bad_val, bool) and math.isfinite(bad_val):
        assert "application supported scope" in str(exc_info.value).lower()


def test_supported_input_validation_rejects_composition_sum_over_100_and_bad_target():
    row = _base_4140()
    row["c_wt"] = 60.0
    row["cr_wt"] = 45.0
    with pytest.raises(ValueError, match="exceeds 100"):
        validate_supported_inputs(row)

    for bad_target in (float("nan"), float("inf"), True, -1.0, 100.5):
        with pytest.raises(ValueError):
            validate_target_hrc(bad_target)
    assert math.isclose(validate_target_hrc(45.0), 45.0)


def test_recommender_refuses_candidates_when_chemistry_out_of_observed_bounds():
    class SpyModel:
        def __init__(self):
            self.called = False

        def predict(self, X):
            self.called = True
            return [42.5] * len(X)

    spy = SpyModel()
    comp = {k: v for k, v in _base_4140().items() if k not in ("temper_temp_c", "temper_time_s")}
    comp["mo_wt"] = 0.50  # Within application supported scope, but outside observed [0.0, 0.36]

    res = recommend_recipe(
        models={"xgb": spy, "b2": spy},
        meta={"features": list(OBSERVED_UNIVARIATE_BOUNDS.keys())[:-2] + ["temper_temp_c", "log_time"], "conformal_q90": 4.97},
        comp_dict=comp,
        target_hrc=42.5,
    )
    assert spy.called is False
    assert res["recommendable"] is False
    assert res["status"] == "OUT_OF_OBSERVED_SCOPE"
    assert res["is_extrapolated"] is True
    assert res["high_uncertainty"] is True
    assert res["multivariate_support_verified"] is False
    assert res["candidates"] == []
    assert res["n_solutions"] == 0
