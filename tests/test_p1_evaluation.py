"""Tests for P1 grouped nested CV + separate split-calibration evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor

from materials_data_lab.modeling import COMP_COLS, RF_FEATURES
from materials_data_lab.p1_evaluation import (
    DEFAULT_CSV_PATH,
    XGB_PARAM_SPACE,
    _VerifiedGroupKFold,
    build_split_manifest,
    compute_conformal_quantile,
    generate_xgb_candidates,
    load_and_prepare_p1_data,
    main as p1_main,
    run_p1_evaluation,
    run_p1_fold,
    summarize_evaluation_results,
)

FAST_SPACE = {
    "max_depth": [1, 2, 3, 4],
    "min_samples_leaf": [1, 2, 3],
    "min_samples_split": [2, 4],
}


def _fast_tree_factory(seed: int) -> BaseEstimator:
    return DecisionTreeRegressor(random_state=seed)


def _make_synthetic_raw_csv(path: Path, n_groups: int = 12, rows_per_group: int = 5) -> Path:
    rng = np.random.default_rng(123)
    rows = []
    for g_idx in range(n_groups):
        c_val = 0.30 + 0.05 * g_idx
        for r_idx in range(rows_per_group):
            temp_c = 200.0 + 90.0 * r_idx
            time_s = 1800.0 * (r_idx + 1)
            hardness = 62.0 - 0.05 * temp_c + 5.0 * c_val + float(rng.normal(0.0, 0.3))
            rows.append({
                "Source": f"Src{g_idx % 3}",
                "Steel type": f"Grade_{g_idx:02d}",
                "Initial hardness (HRC) - post quenching": "?" if g_idx % 2 == 0 else "58.0",
                "Tempering time (s)": time_s,
                "Tempering temperature (ºC)": temp_c,
                "C (%wt)": c_val,
                "Mn (%wt)": 0.70,
                "P (%wt)": 0.015,
                "S (%wt)": 0.015,
                "Si (%wt)": 0.25,
                "Ni (%wt)": 0.10,
                "Cr (%wt)": 0.50,
                "Mo (%wt)": 0.10,
                "V (%wt)": 0.0,
                "Al (%wt)": 0.0,
                "Cu (%wt)": 0.02,
                "Final hardness (HRC) - post tempering": round(hardness, 2),
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_feature_df(n_groups: int = 15, rows_per_group: int = 4) -> pd.DataFrame:
    rows = []
    orig_id = 0
    for g_idx in range(n_groups):
        c_val = 0.25 + 0.04 * g_idx
        for r_idx in range(rows_per_group):
            temp_c = 200.0 + 100.0 * r_idx
            time_s = 1800.0 + 600.0 * r_idx
            rows.append({
                "orig_row_id": orig_id,
                "steel_type": f"S_{g_idx:02d}",
                "source": f"Src_{g_idx % 3}",
                **{col: 0.1 for col in COMP_COLS},
                "c_wt": c_val,
                "temper_temp_c": temp_c,
                "temper_time_s": time_s,
                "log_time": math.log10(time_s),
                "p_hj": ((temp_c + 273.15) / 1000.0) * (19.5 + math.log10(time_s / 3600.0)),
                "final_hrc": 55.0 - 0.04 * temp_c + 2.0 * c_val,
            })
            orig_id += 1
    return pd.DataFrame(rows)


def test_split_manifest_group_disjointness_and_target_blindness():
    df = _make_feature_df(n_groups=16, rows_per_group=5)
    manifest_a = build_split_manifest(df.drop(columns=["final_hrc"]), n_outer_splits=4, n_inner_splits=3, seed=42)

    df_shuffled_targets = df.copy()
    df_shuffled_targets["final_hrc"] = np.linspace(5.0, 95.0, len(df))
    manifest_b = build_split_manifest(df_shuffled_targets, n_outer_splits=4, n_inner_splits=3, seed=42)
    assert manifest_a == manifest_b

    all_groups = set(df["steel_type"].tolist())
    all_rows = set(df["orig_row_id"].tolist())
    seen_test_rows: set[int] = set()

    for fold in manifest_a["folds"]:
        pt_g = set(fold["groups"]["proper_train"])
        cal_g = set(fold["groups"]["calibration"])
        te_g = set(fold["groups"]["outer_test"])
        assert pt_g.isdisjoint(cal_g)
        assert pt_g.isdisjoint(te_g)
        assert cal_g.isdisjoint(te_g)
        assert pt_g | cal_g | te_g == all_groups

        pt_r = set(fold["proper_train_orig_row_ids"])
        cal_r = set(fold["calibration_orig_row_ids"])
        te_r = set(fold["outer_test_orig_row_ids"])
        assert pt_r.isdisjoint(cal_r)
        assert pt_r.isdisjoint(te_r)
        assert cal_r.isdisjoint(te_r)
        assert pt_r | cal_r | te_r == all_rows
        seen_test_rows.update(te_r)

        assert len(fold["inner_folds"]) == 3
        for in_fold in fold["inner_folds"]:
            in_tr = set(in_fold["train_groups"])
            in_val = set(in_fold["val_groups"])
            assert in_tr.isdisjoint(in_val)
            assert in_tr.isdisjoint(cal_g)
            assert in_tr.isdisjoint(te_g)
            assert in_val.isdisjoint(cal_g)
            assert in_val.isdisjoint(te_g)
            assert in_tr | in_val == pt_g

    assert seen_test_rows == all_rows


def test_conformal_quantile_finite_sample_rank_and_boundary_cases():
    # n = 8, alpha = 0.10 -> ceil(9 * 0.9) = 9 > 8 (overflow)
    res_n8_inf = compute_conformal_quantile([3.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0], alpha=0.10, overflow_policy="inf")
    assert res_n8_inf["rank_1based"] == 9
    assert res_n8_inf["rank_exceeded_n"] is True
    assert res_n8_inf["finite_sample_feasible"] is False
    assert math.isinf(res_n8_inf["q"])
    assert "no group-level" in res_n8_inf["coverage_note"].lower()

    res_n8_cap = compute_conformal_quantile([3.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0], alpha=0.10, overflow_policy="cap_max")
    assert res_n8_cap["rank_exceeded_n"] is True
    assert math.isclose(res_n8_cap["q"], 8.0)

    with pytest.raises(ValueError, match="too small"):
        compute_conformal_quantile([1.0, 2.0, 3.0], alpha=0.10, overflow_policy="raise")

    # Exact boundary: n = 9, alpha = 0.10 -> ceil(10 * 0.9) = 9 <= 9 (exact max element)
    res_n9 = compute_conformal_quantile([9.0, 1.0, 5.0, 3.0, 2.0, 8.0, 7.0, 4.0, 6.0], alpha=0.10)
    assert res_n9["rank_1based"] == 9
    assert res_n9["rank_exceeded_n"] is False
    assert res_n9["finite_sample_feasible"] is True
    assert math.isclose(res_n9["q"], 9.0)

    # n = 10 -> ceil(11 * 0.9) = ceil(9.9) = 10 -> 10th element
    res_n10 = compute_conformal_quantile([10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], alpha=0.10)
    assert res_n10["rank_1based"] == 10
    assert math.isclose(res_n10["q"], 10.0)

    # n = 19 -> ceil(20 * 0.9) = 18 <= 19 -> 18th sorted value
    vals_19 = list(reversed(range(1, 20)))
    res_n19 = compute_conformal_quantile(vals_19, alpha=0.10)
    assert res_n19["rank_1based"] == 18
    assert res_n19["rank_exceeded_n"] is False
    assert math.isclose(res_n19["q"], 18.0)

    for bad_input in ([], [float("nan"), 1.0], [-0.5, 1.0]):
        with pytest.raises(ValueError):
            compute_conformal_quantile(bad_input, alpha=0.10)


def test_verified_group_kfold_rejects_missing_or_leaked_groups():
    splitter = _VerifiedGroupKFold(n_splits=2, expected_groups={"A", "B"})
    X = np.zeros((4, 2))
    with pytest.raises(ValueError, match="requires explicit `groups`"):
        list(splitter.split(X, groups=None))
    with pytest.raises(ValueError, match="outside proper_train"):
        list(splitter.split(X, groups=np.array(["A", "A", "CAL", "CAL"])))


def test_cal_and_test_targets_never_enter_fit_or_search_and_no_double_refit():
    df = _make_feature_df(n_groups=16, rows_per_group=4)
    manifest = build_split_manifest(df.drop(columns=["final_hrc"]), n_outer_splits=4, n_inner_splits=3, seed=42)
    fold0 = manifest["folds"][0]

    # Place unmistakable sentinel targets in calibration (9999.0) and outer_test (-9999.0)
    df.loc[fold0["proper_train_pos"], "final_hrc"] = 40.0
    df.loc[fold0["calibration_pos"], "final_hrc"] = 9999.0
    df.loc[fold0["outer_test_pos"], "final_hrc"] = -9999.0

    fit_log: list[dict[str, float]] = []

    class SentinelSpyEstimator(BaseEstimator, RegressorMixin):
        def __init__(self, max_depth: int = 1, min_samples_leaf: int = 1, min_samples_split: int = 2):
            self.max_depth = max_depth
            self.min_samples_leaf = min_samples_leaf
            self.min_samples_split = min_samples_split

        def fit(self, X, y):
            y_arr = np.asarray(y, dtype=float)
            assert np.all(np.isclose(y_arr, 40.0)), "Calibration or outer_test target leaked into estimator.fit!"
            fit_log.append({"n": float(len(y_arr)), "max_abs_y": float(np.max(np.abs(y_arr)))})
            return self

        def predict(self, X):
            return np.full(len(X), 40.0)

    cands, _ = generate_xgb_candidates(5, seed=42, canonical_full_n_iter=20, param_space=FAST_SPACE)
    rec, pred_df = run_p1_fold(
        df,
        fold0,
        candidates=cands,
        n_inner_splits=3,
        alpha=0.10,
        seed=42,
        estimator_factory=lambda _s: SentinelSpyEstimator(),
    )

    # 5 candidates * 3 inner folds + 1 refit via RandomizedSearchCV(refit=True) == 16 (no double refit)
    assert len(fit_log) == 16
    assert rec["fit_counts"]["xgb_total_fits"] == 16
    assert len(rec["verified_inner_folds"]) == 3
    # Calibration quantile comes from |9999.0 - 40.0| = 9959.0 (calibration untouched by fit, used only for q90)
    assert math.isclose(rec["calibration"]["q"], 959.0 + 9000.0)
    # Predictions on outer_test carry exact sentinel y_true without touching baselines
    assert np.all(np.isclose(pred_df["y_true"].to_numpy(), -9999.0))
    assert np.all(np.isclose(pred_df["y_pred_b1"].to_numpy(), 40.0))


def test_original_row_id_mapping_after_invalid_row_filtering(tmp_path: Path):
    raw_csv = _make_synthetic_raw_csv(tmp_path / "raw_with_bad.csv", n_groups=9, rows_per_group=4)
    raw_df = pd.read_csv(raw_csv)
    raw_df["Final hardness (HRC) - post tempering"] = raw_df["Final hardness (HRC) - post tempering"].astype(object)
    raw_df["C (%wt)"] = raw_df["C (%wt)"].astype(object)
    # Inject invalid rows at known raw indices 0, 5, 11 (must be dropped)
    raw_df.loc[0, "Final hardness (HRC) - post tempering"] = "?"
    raw_df.loc[5, "Tempering time (s)"] = 0.0
    raw_df.loc[11, "C (%wt)"] = "?"
    # Inject valid <20 HRC row at raw index 10 (must be preserved per D-02)
    raw_df.loc[10, "Final hardness (HRC) - post tempering"] = 14.5
    raw_df.to_csv(raw_csv, index=False)

    filtered, prep = load_and_prepare_p1_data(raw_csv)
    assert prep["raw_rows"] == 36
    assert prep["dropped_rows"] == 3
    assert prep["dropped_orig_row_ids"] == [0, 5, 11]
    assert prep["preserved_below_20_hrc_rows"] >= 1
    assert 10 in filtered["orig_row_id"].tolist()

    outdir = tmp_path / "p1_out"
    run_p1_evaluation(
        csv_path=raw_csv,
        outdir=outdir,
        mode="full",
        n_iter=3,
        canonical_full_n_iter=5,
        n_outer_splits=3,
        n_inner_splits=2,
        param_space=FAST_SPACE,
        estimator_factory=_fast_tree_factory,
    )
    preds = pd.read_csv(outdir / "predictions.csv")
    assert len(preds) == 33
    assert set(preds["orig_row_id"].tolist()).isdisjoint({0, 5, 11})
    for _, row in preds.iterrows():
        oid = int(row["orig_row_id"])
        assert str(row["steel_type"]) == str(raw_df.loc[oid, "Steel type"]).strip()
        assert math.isclose(float(row["y_true"]), float(raw_df.loc[oid, "Final hardness (HRC) - post tempering"]))


def test_metrics_known_example_fold_mean_vs_pooled():
    fold0 = {
        "fold": 0,
        "models": {
            "B1_Naive": {"MAE": 5.0, "RMSE": 5.0, "R2": 0.0},
            "B2_Physics": {"MAE": 4.0, "RMSE": 4.0, "R2": 0.1},
            "XGBoost": {"MAE": 2.0, "RMSE": math.sqrt(5.0), "R2": 0.5},
        },
        "uncertainty": {"q90_k": 2.0, "full_width": 4.0, "empirical_coverage": 0.5},
    }
    fold1 = {
        "fold": 1,
        "models": {
            "B1_Naive": {"MAE": 5.0, "RMSE": 5.0, "R2": 0.0},
            "B2_Physics": {"MAE": 4.0, "RMSE": 4.0, "R2": 0.1},
            "XGBoost": {"MAE": 1.0, "RMSE": 2.0, "R2": 0.8},
        },
        "uncertainty": {"q90_k": 4.0, "full_width": 8.0, "empirical_coverage": 1.0},
    }
    preds_df = pd.DataFrame({
        "orig_row_id": [0, 1, 2, 3, 4, 5],
        "fold": [0, 0, 1, 1, 1, 1],
        "steel_type": ["G0", "G0", "G1", "G1", "G1", "G1"],
        "y_true": [10.0, 10.0, 20.0, 20.0, 20.0, 20.0],
        "y_pred_b1": [5.0, 15.0, 15.0, 25.0, 15.0, 25.0],
        "y_pred_b2": [6.0, 14.0, 16.0, 24.0, 16.0, 24.0],
        "y_pred_xgb": [11.0, 7.0, 20.0, 20.0, 20.0, 24.0],
        "covered": [True, False, True, True, True, True],
    })
    agg = summarize_evaluation_results([fold0, fold1], preds_df)
    xgb_sum = agg["models"]["XGBoost"]
    # Unweighted fold mean: (2.0 + 1.0)/2 = 1.5; pooled weighted by 2 vs 4 rows: (1+3+0+0+0+4)/6 = 8/6
    assert math.isclose(xgb_sum["fold_mean"]["MAE_mean"], 1.5)
    assert math.isclose(xgb_sum["pooled_outer_test"]["MAE"], 8.0 / 6.0)
    assert math.isclose(xgb_sum["fold_mean"]["RMSE_mean"], (math.sqrt(5.0) + 2.0) / 2.0)
    assert math.isclose(xgb_sum["pooled_outer_test"]["RMSE"], math.sqrt(26.0 / 6.0))
    assert math.isclose(agg["uncertainty"]["fold_mean_coverage"], 0.75)
    assert math.isclose(agg["uncertainty"]["pooled_empirical_coverage"], 5.0 / 6.0)
    assert math.isclose(agg["per_group_xgb"]["G0"]["mean_signed_error"], -1.0)
    assert math.isclose(agg["per_group_xgb"]["G1"]["mean_signed_error"], 1.0)


def test_pilot_runs_only_fold_0_with_5_prefix_candidates_and_models_guard(tmp_path: Path):
    raw_csv = _make_synthetic_raw_csv(tmp_path / "pilot_synth.csv", n_groups=16, rows_per_group=5)
    outdir = tmp_path / "pilot_out"

    with pytest.raises(ValueError, match="protected models directory"):
        run_p1_evaluation(csv_path=raw_csv, outdir="models/should_block", mode="pilot")

    res = run_p1_evaluation(
        csv_path=raw_csv,
        outdir=outdir,
        mode="pilot",
        param_space=FAST_SPACE,
        estimator_factory=_fast_tree_factory,
    )
    assert res["mode"] == "pilot"
    assert res["timed_out"] is False
    assert [f["fold"] for f in res["folds"]] == [0]
    assert res["fit_counts"] == {"xgb_fits": 16, "baseline_fits": 2, "total_fits": 18}

    run_manifest = json.loads((outdir / "run_manifest.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(outdir / "predictions.csv")
    assert preds["fold"].unique().tolist() == [0]
    assert run_manifest["candidates"]["requested_n_iter"] == 5
    assert run_manifest["candidates"]["canonical_full_n_iter"] == 20
    assert run_manifest["candidates"]["pilot_is_exact_prefix_of_full"] is True
    assert run_manifest["candidates"]["selected_candidates"] == run_manifest["candidates"]["canonical_full_candidates"][:5]

    # Verify timeout guard stops gracefully without touching models/
    timeout_dir = tmp_path / "pilot_timeout"
    exit_code = p1_main(["--csv", str(raw_csv), "--outdir", str(timeout_dir), "--mode", "pilot", "--max-seconds", "-1"])
    assert exit_code == 1
    assert json.loads((timeout_dir / "results_summary.json").read_text(encoding="utf-8"))["timed_out"] is True


def test_candidate_sequence_consistency_with_real_xgb_space():
    c5, meta5 = generate_xgb_candidates(5, seed=42, canonical_full_n_iter=20)
    c20, meta20 = generate_xgb_candidates(20, seed=42, canonical_full_n_iter=20)
    assert len(c5) == 5
    assert len(c20) == 20
    assert c5 == c20[:5]
    assert meta5["pilot_is_exact_prefix_of_full"] is True
    assert set(c5[0].keys()) == set(XGB_PARAM_SPACE.keys())
    assert DEFAULT_CSV_PATH.name.endswith("Raiipa.csv")
    assert "temper_temp_c" in RF_FEATURES
