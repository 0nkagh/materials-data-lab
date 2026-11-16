"""Tests for reproducible P1 reporting metrics, sanitization, and figure generation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from materials_data_lab.p1_reporting import (
    compute_reporting_metrics,
    generate_p1_report,
    main as reporting_main,
    sanitize_run_provenance,
    validate_predictions_frame,
)


def _known_small_predictions() -> pd.DataFrame:
    # 3 steel groups across 2 folds with deliberately unequal row counts:
    # Fold 0 (q90=3.0, width=6.0):
    #   G_small (n=1): err=+4.0, covered=False
    #   G_big   (n=3): err=-1.0,-1.0,+1.0, covered=True,True,True
    # Fold 1 (q90=1.0, width=2.0):
    #   G_med   (n=2): err=0.0,+2.0, covered=True,False
    return pd.DataFrame({
        "orig_row_id": [10, 11, 12, 13, 20, 21],
        "fold": [0, 0, 0, 0, 1, 1],
        "steel_type": ["G_small", "G_big", "G_big", "G_big", "G_med", "G_med"],
        "source": ["SrcA", "SrcA", "SrcA", "SrcA", "SrcB", "SrcB"],
        "y_true": [30.0, 40.0, 42.0, 44.0, 20.0, 22.0],
        "y_pred_b1": [20.0, 40.0, 40.0, 40.0, 20.0, 20.0],
        "y_pred_b2": [28.0, 39.0, 41.0, 43.0, 20.0, 21.0],
        "y_pred_xgb": [34.0, 39.0, 41.0, 45.0, 20.0, 24.0],
        "q90_k": [3.0, 3.0, 3.0, 3.0, 1.0, 1.0],
        "abs_error_xgb": [4.0, 1.0, 1.0, 1.0, 0.0, 2.0],
        "covered": [False, True, True, True, True, False],
    })


def test_compute_reporting_metrics_known_small_examples():
    df = _known_small_predictions()
    m = compute_reporting_metrics(df)

    assert m["integrity"]["n_prediction_rows"] == 6
    assert m["integrity"]["n_unique_orig_row_ids"] == 6
    assert m["integrity"]["n_unique_groups"] == 3

    xgb = m["models"]["XGBoost"]
    # Pooled row-weighted MAE: (4 + 1 + 1 + 1 + 0 + 2) / 6 = 1.5
    assert math.isclose(xgb["pooled_row_weighted"]["MAE"], 1.5)
    # Fold MAE: fold0=(4+1+1+1)/4=1.75, fold1=(0+2)/2=1.0 -> fold_mean=1.375
    assert math.isclose(xgb["fold_mean"]["MAE_mean"], 1.375)
    # Group-macro MAE: G_small=4.0, G_big=1.0, G_med=1.0 -> mean=2.0
    assert math.isclose(xgb["group_macro"]["MAE_mean"], 2.0)

    # Pooled RMSE: sqrt((16 + 1 + 1 + 1 + 0 + 4) / 6) = sqrt(23/6)
    assert math.isclose(xgb["pooled_row_weighted"]["RMSE"], math.sqrt(23.0 / 6.0))

    unc = m["uncertainty"]
    # Global row-weighted coverage: 4/6 = 2/3
    assert math.isclose(unc["global_row_weighted_coverage"], 4.0 / 6.0)
    # Fold-mean coverage: (0.75 + 0.50)/2 = 0.625
    assert math.isclose(unc["fold_mean_coverage"], 0.625)
    # Group-macro coverage: (0.0 + 1.0 + 0.5)/3 = 0.50
    assert math.isclose(unc["group_macro_coverage"], 0.50)
    # Fold-mean width: (6.0 + 2.0)/2 = 4.0; row-weighted width: (4*6.0 + 2*2.0)/6 = 28/6
    assert math.isclose(unc["fold_mean_full_width"], 4.0)
    assert math.isclose(unc["row_weighted_mean_full_width"], 28.0 / 6.0)

    # Small groups must remain preserved in per_steel breakdown
    assert len(m["per_steel"]) == 3
    assert all(row["small_group_n_lt_30"] is True for row in m["per_steel"])


def test_validate_predictions_frame_rejects_bad_inputs():
    df = _known_small_predictions()
    with pytest.raises(ValueError, match="missing required columns"):
        validate_predictions_frame(df.drop(columns=["covered"]))

    dup_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate orig_row_id"):
        validate_predictions_frame(dup_df)

    nan_df = df.copy()
    nan_df.loc[0, "y_pred_xgb"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        validate_predictions_frame(nan_df)


def test_sanitize_run_provenance_excludes_os_paths_and_emails():
    raw = {
        "schema_version": 1,
        "mode": "full",
        "package_version": "1.5.0",
        "environment": {
            "platform": "Windows-11-PrivateHost",
            "python": "3.12.10",
            "search_n_jobs": 1,
            "xgb_n_jobs": 1,
        },
        "git": {"git_commit": "abc123", "git_dirty": False},
        "data_preparation": {
            "csv_path": r"C:\Users\SecretUser\data\raw\file.csv",
            "csv_sha256": "deadbeef",
            "raw_rows": 1466,
            "kept_rows": 1466,
            "dropped_rows": 0,
            "n_unique_groups": 34,
        },
        "parameters": {"seed": 42, "outer_splits": 5},
        "candidates": {
            "seed": 42,
            "requested_n_iter": 20,
            "canonical_full_n_iter": 20,
            "pilot_is_exact_prefix_of_full": True,
            "selected_candidates": [{"max_depth": 5}],
        },
        "fit_counts": {"xgb_fits": 305, "baseline_fits": 10, "total_fits": 315},
        "timing": {"total_elapsed_seconds": 53.5, "timed_out": False},
        "coverage_note": "No guarantee.",
    }
    clean = sanitize_run_provenance(raw)
    serialized = json.dumps(clean)
    assert "SecretUser" not in serialized
    assert "Windows-11" not in serialized
    assert clean["csv_sha256"] == "deadbeef"
    assert clean["git_commit"] == "abc123"
    assert clean["parameters"]["seed"] == 42


def test_generate_p1_report_creates_plots_markdown_and_json(tmp_path: Path):
    in_dir = tmp_path / "p1_in"
    out_dir = tmp_path / "p1_rep"
    in_dir.mkdir(parents=True)
    _known_small_predictions().to_csv(in_dir / "predictions.csv", index=False)

    metrics = generate_p1_report(in_dir, out_dir)
    assert math.isclose(metrics["models"]["XGBoost"]["pooled_row_weighted"]["MAE"], 1.5)

    for fname in (
        "summary.json",
        "README.md",
        "fig1_predicted_vs_actual.png",
        "fig2_steel_mae.png",
        "fig3_steel_coverage.png",
    ):
        fpath = out_dir / fname
        assert fpath.is_file()
        assert fpath.stat().st_size > 0
        if fname.endswith(".png"):
            assert fpath.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    report_text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "G_small" in report_text
    assert "No theoretical 90% coverage guarantee" in report_text or "no theoretical 90% coverage guarantee" in report_text

    # --no-overwrite should fail safely on second invocation
    assert reporting_main(["--input-dir", str(in_dir), "--outdir", str(out_dir)]) == 0
    with pytest.raises(FileExistsError):
        generate_p1_report(in_dir, out_dir, overwrite=False)
