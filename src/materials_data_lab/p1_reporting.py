"""Reproducible reporting and figure generation for completed P1 evaluation outputs.

Reads pre-computed P1 artifacts (`predictions.csv`, and optionally `results_summary.json`
and sanitized metadata from `run_manifest.json`) without training models, loading `.joblib`
pickles, or accessing raw datasets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REQUIRED_PRED_COLUMNS = {
    "orig_row_id",
    "fold",
    "steel_type",
    "source",
    "y_true",
    "y_pred_b1",
    "y_pred_b2",
    "y_pred_xgb",
    "q90_k",
    "abs_error_xgb",
    "covered",
}

NUMERIC_PRED_COLUMNS = [
    "orig_row_id",
    "fold",
    "y_true",
    "y_pred_b1",
    "y_pred_b2",
    "y_pred_xgb",
    "q90_k",
    "abs_error_xgb",
]


def _coerce_bool_series(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        raise ValueError("Column 'covered' contains non-boolean values.")
    return mapped.to_numpy(dtype=bool)


def validate_predictions_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required P1 prediction columns, numeric finiteness, and unique orig_row_id."""
    missing = sorted(REQUIRED_PRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"predictions.csv is missing required columns: {missing}")
    if df.empty:
        raise ValueError("predictions.csv contains zero rows.")

    clean = df.copy()
    for col in NUMERIC_PRED_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="raise")
        vals = clean[col].to_numpy(dtype=float)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Non-finite values found in column {col!r}.")

    clean["orig_row_id"] = clean["orig_row_id"].astype(int)
    clean["fold"] = clean["fold"].astype(int)
    clean["steel_type"] = clean["steel_type"].astype(str)
    clean["source"] = clean["source"].astype(str)
    clean["covered"] = _coerce_bool_series(clean["covered"])

    if clean["orig_row_id"].nunique() != len(clean):
        n_dups = len(clean) - clean["orig_row_id"].nunique()
        raise ValueError(f"Duplicate orig_row_id entries detected in predictions.csv ({n_dups} duplicates).")

    return clean


def _regression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    err = y_pred - y_true
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(np.mean(err))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 and float(np.var(y_true)) > 0.0 else None
    return {
        "n_rows": int(len(y_true)),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "mean_signed_error": bias,
    }


def sanitize_run_provenance(run_manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Extract share-safe reproducibility metadata without OS dumps, emails, or local paths."""
    if not run_manifest:
        return {}
    git_raw = run_manifest.get("git", {})
    params = run_manifest.get("parameters", {})
    cands = run_manifest.get("candidates", {})
    data_prep = run_manifest.get("data_preparation", {})
    timing = run_manifest.get("timing", {})
    return {
        "schema_version": run_manifest.get("schema_version"),
        "mode": run_manifest.get("mode"),
        "package_version": run_manifest.get("package_version"),
        "git_commit": git_raw.get("git_commit"),
        "git_dirty": git_raw.get("git_dirty"),
        "csv_sha256": data_prep.get("csv_sha256"),
        "raw_rows": data_prep.get("raw_rows"),
        "kept_rows": data_prep.get("kept_rows"),
        "dropped_rows": data_prep.get("dropped_rows"),
        "n_unique_groups": data_prep.get("n_unique_groups"),
        "parameters": params,
        "fit_counts": run_manifest.get("fit_counts", {}),
        "total_elapsed_seconds": timing.get("total_elapsed_seconds"),
        "timed_out": timing.get("timed_out"),
        "search_n_jobs": run_manifest.get("environment", {}).get("search_n_jobs"),
        "xgb_n_jobs": run_manifest.get("environment", {}).get("xgb_n_jobs"),
        "candidates": {
            "seed": cands.get("seed"),
            "requested_n_iter": cands.get("requested_n_iter"),
            "canonical_full_n_iter": cands.get("canonical_full_n_iter"),
            "pilot_is_exact_prefix_of_full": cands.get("pilot_is_exact_prefix_of_full"),
            "selected_candidates": cands.get("selected_candidates"),
        },
        "coverage_note": run_manifest.get("coverage_note"),
    }


def compute_reporting_metrics(
    pred_df: pd.DataFrame,
    *,
    results_summary: dict[str, Any] | None = None,
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute fold-level, fold-mean, pooled (row-weighted), group-macro, and per-steel metrics."""
    df = validate_predictions_frame(pred_df)
    summary_by_fold = {int(f["fold"]): f for f in (results_summary or {}).get("folds", [])}

    folds_out: list[dict[str, Any]] = []
    for fold_id, fdf in df.groupby("fold", sort=True):
        fold_id_int = int(fold_id)
        q90_vals = fdf["q90_k"].unique()
        if len(q90_vals) != 1:
            raise ValueError(f"Fold {fold_id_int} has non-constant q90_k values: {q90_vals}")
        q90_k = float(q90_vals[0])
        full_width = 2.0 * q90_k if math.isfinite(q90_k) else float("inf")

        y_t = fdf["y_true"].to_numpy(dtype=float)
        fold_rec: dict[str, Any] = {
            "fold": fold_id_int,
            "n_test_rows": int(len(fdf)),
            "n_test_groups": int(fdf["steel_type"].nunique()),
            "models": {
                "B1_Naive": _regression_stats(y_t, fdf["y_pred_b1"].to_numpy(dtype=float)),
                "B2_Physics": _regression_stats(y_t, fdf["y_pred_b2"].to_numpy(dtype=float)),
                "XGBoost": _regression_stats(y_t, fdf["y_pred_xgb"].to_numpy(dtype=float)),
            },
            "uncertainty": {
                "q90_k": q90_k,
                "full_width": full_width,
                "empirical_coverage": float(np.mean(fdf["covered"].to_numpy(dtype=bool))),
                "n_covered": int(fdf["covered"].sum()),
            },
        }
        if fold_id_int in summary_by_fold:
            sf = summary_by_fold[fold_id_int]
            cal = sf.get("calibration", {})
            fold_rec["best_params"] = sf.get("best_params")
            fold_rec["best_inner_cv_mae"] = sf.get("best_inner_cv_mae")
            fold_rec["calibration"] = {
                "n_cal": cal.get("n_cal"),
                "rank_1based": cal.get("rank_1based"),
                "rank_exceeded_n": cal.get("rank_exceeded_n"),
                "finite_sample_feasible": cal.get("finite_sample_feasible"),
            }
            fold_rec["timing_seconds"] = sf.get("timing_seconds", {}).get("fold_total")
            fold_rec["fit_counts"] = sf.get("fit_counts")
        folds_out.append(fold_rec)

    per_steel: list[dict[str, Any]] = []
    for grp_name, gdf in df.groupby("steel_type", sort=True):
        y_t = gdf["y_true"].to_numpy(dtype=float)
        xgb_stats = _regression_stats(y_t, gdf["y_pred_xgb"].to_numpy(dtype=float))
        b2_stats = _regression_stats(y_t, gdf["y_pred_b2"].to_numpy(dtype=float))
        b1_stats = _regression_stats(y_t, gdf["y_pred_b1"].to_numpy(dtype=float))
        srcs = {str(k): int(v) for k, v in gdf["source"].value_counts().sort_index().items()}
        per_steel.append({
            "steel_type": str(grp_name),
            "folds": sorted(int(v) for v in gdf["fold"].unique().tolist()),
            "n_rows": int(len(gdf)),
            "small_group_n_lt_30": bool(len(gdf) < 30),
            "sources": srcs,
            "xgb_MAE": xgb_stats["MAE"],
            "xgb_RMSE": xgb_stats["RMSE"],
            "xgb_bias": xgb_stats["mean_signed_error"],
            "b2_MAE": b2_stats["MAE"],
            "b1_MAE": b1_stats["MAE"],
            "q90_k": float(gdf["q90_k"].iloc[0]),
            "empirical_coverage": float(np.mean(gdf["covered"].to_numpy(dtype=bool))),
            "n_covered": int(gdf["covered"].sum()),
        })

    per_steel.sort(key=lambda r: (-r["xgb_MAE"], r["steel_type"]))

    models_summary: dict[str, Any] = {}
    y_all = df["y_true"].to_numpy(dtype=float)
    for model_key, pred_col, steel_mae_key in (
        ("B1_Naive", "y_pred_b1", "b1_MAE"),
        ("B2_Physics", "y_pred_b2", "b2_MAE"),
        ("XGBoost", "y_pred_xgb", "xgb_MAE"),
    ):
        fold_maes = [f["models"][model_key]["MAE"] for f in folds_out]
        fold_rmses = [f["models"][model_key]["RMSE"] for f in folds_out]
        fold_r2s = [f["models"][model_key]["R2"] for f in folds_out if f["models"][model_key]["R2"] is not None]
        grp_maes = [s[steel_mae_key] for s in per_steel]
        models_summary[model_key] = {
            "fold_mean": {
                "MAE_mean": float(np.mean(fold_maes)),
                "MAE_std_ddof0": float(np.std(fold_maes, ddof=0)),
                "RMSE_mean": float(np.mean(fold_rmses)),
                "RMSE_std_ddof0": float(np.std(fold_rmses, ddof=0)),
                "R2_mean": float(np.mean(fold_r2s)) if fold_r2s else None,
                "R2_std_ddof0": float(np.std(fold_r2s, ddof=0)) if fold_r2s else None,
            },
            "pooled_row_weighted": _regression_stats(y_all, df[pred_col].to_numpy(dtype=float)),
            "group_macro": {
                "n_groups": int(len(per_steel)),
                "MAE_mean": float(np.mean(grp_maes)),
                "MAE_std_ddof0": float(np.std(grp_maes, ddof=0)),
                "MAE_median": float(np.median(grp_maes)),
                "MAE_min": float(np.min(grp_maes)),
                "MAE_max": float(np.max(grp_maes)),
            },
        }

    fold_covs = [f["uncertainty"]["empirical_coverage"] for f in folds_out]
    fold_q90s = [f["uncertainty"]["q90_k"] for f in folds_out]
    fold_widths = [f["uncertainty"]["full_width"] for f in folds_out]
    grp_covs = [s["empirical_coverage"] for s in per_steel]

    regimes: dict[str, Any] = {}
    for label, mask in (
        ("final_hrc_lt_20", y_all < 20.0),
        ("final_hrc_lt_25", y_all < 25.0),
    ):
        sub = df.loc[mask]
        if not sub.empty:
            regimes[label] = {
                "n_rows": int(len(sub)),
                "xgb_MAE": float(np.mean(np.abs(sub["y_true"] - sub["y_pred_xgb"]))),
                "empirical_coverage": float(np.mean(sub["covered"])),
                "n_covered": int(sub["covered"].sum()),
            }

    by_source: dict[str, Any] = {}
    for src_name, sdf in df.groupby("source", sort=True):
        by_source[str(src_name)] = {
            "n_rows": int(len(sdf)),
            "n_groups": int(sdf["steel_type"].nunique()),
            "xgb_MAE": float(np.mean(np.abs(sdf["y_true"] - sdf["y_pred_xgb"]))),
            "empirical_coverage": float(np.mean(sdf["covered"])),
            "n_covered": int(sdf["covered"].sum()),
        }

    return {
        "integrity": {
            "n_prediction_rows": int(len(df)),
            "n_unique_orig_row_ids": int(df["orig_row_id"].nunique()),
            "n_unique_groups": int(len(per_steel)),
            "evaluated_folds": [f["fold"] for f in folds_out],
        },
        "models": models_summary,
        "uncertainty": {
            "global_row_weighted_coverage": float(np.mean(df["covered"].to_numpy(dtype=bool))),
            "n_covered_rows": int(df["covered"].sum()),
            "n_total_rows": int(len(df)),
            "fold_mean_coverage": float(np.mean(fold_covs)),
            "fold_std_coverage_ddof0": float(np.std(fold_covs, ddof=0)),
            "group_macro_coverage": float(np.mean(grp_covs)),
            "group_macro_coverage_std_ddof0": float(np.std(grp_covs, ddof=0)),
            "group_macro_coverage_median": float(np.median(grp_covs)),
            "fold_mean_q90": float(np.mean(fold_q90s)),
            "fold_mean_full_width": float(np.mean(fold_widths)),
            "row_weighted_mean_q90": float(np.mean(df["q90_k"].to_numpy(dtype=float))),
            "row_weighted_mean_full_width": float(np.mean(2.0 * df["q90_k"].to_numpy(dtype=float))),
            "disclaimer": (
                "No group-level 90% coverage guarantee for unseen steel groups and no external test set."
            ),
        },
        "folds": folds_out,
        "per_steel": per_steel,
        "weak_regimes": regimes,
        "by_source": by_source,
        "provenance": sanitize_run_provenance(run_manifest),
    }


def _plot_pred_vs_actual(df: pd.DataFrame, metrics: dict[str, Any], out_path: Path) -> None:
    y_t = df["y_true"].to_numpy(dtype=float)
    y_p = df["y_pred_xgb"].to_numpy(dtype=float)
    cov = df["covered"].to_numpy(dtype=bool)

    lo = min(float(np.min(y_t)), float(np.min(y_p)))
    hi = max(float(np.max(y_t)), float(np.max(y_p)))
    pad = max(1.0, 0.04 * (hi - lo))
    lims = [lo - pad, hi + pad]

    xgb_pooled = metrics["models"]["XGBoost"]["pooled_row_weighted"]
    unc = metrics["uncertainty"]

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=160)
    ax.scatter(
        y_t[cov],
        y_p[cov],
        s=14,
        alpha=0.55,
        color="#1f77b4",
        edgecolors="none",
        label=f"Within ±q90_k ({int(cov.sum())}/{len(cov)})",
    )
    if (~cov).any():
        ax.scatter(
            y_t[~cov],
            y_p[~cov],
            s=22,
            alpha=0.80,
            marker="x",
            color="#d62728",
            label=f"Outside ±q90_k ({int((~cov).sum())}/{len(cov)})",
        )
    ax.plot(lims, lims, linestyle="--", linewidth=1.4, color="#222222", label="Ideal y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlabel("Actual Post-Tempering Hardness (HRC)")
    ax.set_ylabel("Outer-Test Predicted Hardness — XGBoost (HRC)")
    ax.set_title(
        "P1 Outer-Test Predicted vs. Actual Hardness\n"
        f"Pooled MAE = {xgb_pooled['MAE']:.2f} HRC | RMSE = {xgb_pooled['RMSE']:.2f} HRC | "
        f"Coverage = {unc['global_row_weighted_coverage'] * 100:.2f}% (n = {len(df)})",
        fontsize=10.5,
    )
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_steel_mae(metrics: dict[str, Any], out_path: Path) -> None:
    steels = sorted(metrics["per_steel"], key=lambda r: (r["xgb_MAE"], r["steel_type"]))
    labels = [
        f"{s['steel_type']} (n={s['n_rows']}{'*' if s['small_group_n_lt_30'] else ''})"
        for s in steels
    ]
    maes = [s["xgb_MAE"] for s in steels]
    colors = ["#c44e52" if s["small_group_n_lt_30"] else "#4c72b0" for s in steels]

    pooled_mae = metrics["models"]["XGBoost"]["pooled_row_weighted"]["MAE"]
    macro_mae = metrics["models"]["XGBoost"]["group_macro"]["MAE_mean"]

    fig_h = max(6.0, 0.30 * len(steels) + 1.8)
    fig, ax = plt.subplots(figsize=(9.8, fig_h), dpi=160)
    y_pos = np.arange(len(steels))
    ax.barh(y_pos, maes, color=colors, height=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(pooled_mae, color="#222222", linestyle="--", linewidth=1.3, label=f"Pooled MAE ({pooled_mae:.2f} HRC)")
    ax.axvline(macro_mae, color="#dd8452", linestyle="-.", linewidth=1.3, label=f"Group-Macro MAE ({macro_mae:.2f} HRC)")

    x_pad = max(maes) * 0.16 if maes else 1.0
    ax.set_xlim(0.0, max(maes) + x_pad)
    for idx, mae_val in enumerate(maes):
        ax.text(
            mae_val + 0.08,
            idx,
            f"{mae_val:.2f}",
            va="center",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.6},
        )

    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_xlabel("Outer-Test MAE (HRC)")
    ax.set_title(
        "P1 Outer-Test MAE by Steel Group (All 34 Grades; * marks n < 30 small groups)\n"
        "Models trained on proper_train (20–21 groups); no groups or weak cases excluded",
        fontsize=10,
        pad=26,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=True,
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_steel_coverage(metrics: dict[str, Any], out_path: Path) -> None:
    steels = sorted(metrics["per_steel"], key=lambda r: (r["empirical_coverage"], -r["xgb_MAE"], r["steel_type"]), reverse=True)
    labels = [
        f"{s['steel_type']} (n={s['n_rows']}{'*' if s['small_group_n_lt_30'] else ''})"
        for s in steels
    ]
    cov_pcts = [s["empirical_coverage"] * 100.0 for s in steels]
    colors = ["#2ca02c" if c >= 90.0 else ("#ff7f0e" if c >= 70.0 else "#d62728") for c in cov_pcts]

    unc = metrics["uncertainty"]
    glob_pct = unc["global_row_weighted_coverage"] * 100.0
    macro_pct = unc["group_macro_coverage"] * 100.0

    fig_h = max(6.0, 0.30 * len(steels) + 2.2)
    fig, ax = plt.subplots(figsize=(9.8, fig_h), dpi=160)
    y_pos = np.arange(len(steels))
    ax.barh(y_pos, cov_pcts, color=colors, height=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(90.0, color="#111111", linestyle="--", linewidth=1.4, label="Nominal 90% Target (no guarantee)")
    ax.axvline(glob_pct, color="#1f77b4", linestyle="-.", linewidth=1.2, label=f"Global Row-Weighted ({glob_pct:.2f}%)")
    ax.axvline(macro_pct, color="#9467bd", linestyle=":", linewidth=1.3, label=f"Group-Macro ({macro_pct:.2f}%)")

    ax.set_xlim(0.0, 121.0)
    for idx, (pct, s) in enumerate(zip(cov_pcts, steels)):
        ax.text(
            pct + 1.1,
            idx,
            f"{pct:.1f}% ({s['n_covered']}/{s['n_rows']})",
            va="center",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.6},
        )

    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_xlabel("Empirical Outer-Test Coverage (%)")
    ax.set_title(
        "P1 Per-Steel Empirical ±q90_k Coverage vs. 90% Reference Line (All 34 Grades)\n"
        "* Small groups (n < 30) have coarse step resolution; no external or per-family guarantee",
        fontsize=10,
        pad=26,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=True,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _format_technical_report(metrics: dict[str, Any]) -> str:
    integ = metrics["integrity"]
    xgb_m = metrics["models"]["XGBoost"]
    b2_m = metrics["models"]["B2_Physics"]
    b1_m = metrics["models"]["B1_Naive"]
    unc = metrics["uncertainty"]
    prov = metrics.get("provenance", {})

    lines = [
        "# P1 Grouped Evaluation Report (Held-Out Calibration Benchmark)",
        "",
        "> **Scope & Limitations Notice:** This report summarizes the completed P1 unbiased grouped evaluation (`outer GroupKFold(5)` with target-blind `proper_train` / `calibration` separation at `seed=42`). These metrics characterize `proper_train`-fitted models (`20–21` groups per fold) within the 3 literature compilations; **there is no independent external dataset and no theoretical 90% coverage guarantee for unseen steel groups**. The interactive demo/API (`app.py`, `api.py`) continues to serve legacy pre-built `models/` artifacts (`artifact_meta.json`, `q90 = 4.9673 HRC`), **not** models from this P1 benchmark.",
        "",
        "## 1. Aggregate Benchmark Comparison",
        "",
        "| Model | Fold-Mean MAE ± Std (HRC) | Pooled Row-Weighted MAE (`n={n}`) | Pooled RMSE | Pooled R² | Group-Macro MAE ± Std (`G={g}`) |".format(
            n=integ["n_prediction_rows"], g=integ["n_unique_groups"]
        ),
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **B1_Naive** (`DummyRegressor` median) | `{b1_m['fold_mean']['MAE_mean']:.4f} ± {b1_m['fold_mean']['MAE_std_ddof0']:.4f}` | `{b1_m['pooled_row_weighted']['MAE']:.4f}` | `{b1_m['pooled_row_weighted']['RMSE']:.4f}` | `{b1_m['pooled_row_weighted']['R2']:.4f}` | `{b1_m['group_macro']['MAE_mean']:.4f} ± {b1_m['group_macro']['MAE_std_ddof0']:.4f}` |",
        f"| **B2_Physics** (`LinearRegression` on `p_hj`) | `{b2_m['fold_mean']['MAE_mean']:.4f} ± {b2_m['fold_mean']['MAE_std_ddof0']:.4f}` | `{b2_m['pooled_row_weighted']['MAE']:.4f}` | `{b2_m['pooled_row_weighted']['RMSE']:.4f}` | `{b2_m['pooled_row_weighted']['R2']:.4f}` | `{b2_m['group_macro']['MAE_mean']:.4f} ± {b2_m['group_macro']['MAE_std_ddof0']:.4f}` |",
        f"| **XGBoost (P1)** (`3-fold` inner × `20` cand.) | **`{xgb_m['fold_mean']['MAE_mean']:.4f} ± {xgb_m['fold_mean']['MAE_std_ddof0']:.4f}`** | **`{xgb_m['pooled_row_weighted']['MAE']:.4f}`** | **`{xgb_m['pooled_row_weighted']['RMSE']:.4f}`** | **`{xgb_m['pooled_row_weighted']['R2']:.4f}`** | **`{xgb_m['group_macro']['MAE_mean']:.4f} ± {xgb_m['group_macro']['MAE_std_ddof0']:.4f}`** |",
        "",
        f"- **Uncertainty Band Summary (`outer_test`):** Global row-weighted empirical coverage = **`{unc['global_row_weighted_coverage'] * 100:.2f}%`** (`{unc['n_covered_rows']}/{unc['n_total_rows']}`); Fold-mean coverage = **`{unc['fold_mean_coverage'] * 100:.2f}% ± {unc['fold_std_coverage_ddof0'] * 100:.2f}%`**; Group-macro coverage (`{integ['n_unique_groups']}` grades equally weighted) = **`{unc['group_macro_coverage'] * 100:.2f}% ± {unc['group_macro_coverage_std_ddof0'] * 100:.2f}%`**. Fold-mean `q90_k` = `{unc['fold_mean_q90']:.4f} HRC`, mean full band width = **`{unc['fold_mean_full_width']:.4f} HRC`** (`{unc['row_weighted_mean_full_width']:.4f} HRC` row-weighted).",
        "- **Comparison with Historical `V2-C2` Scores (`2.35` fold-mean / `2.48` pooled MAE):** Higher P1 error (`~+0.50` to `+0.63 HRC`) reflects both removal of selection/calibration overlap **and** structural changes in P1—holding out `7` calibration groups shrinks training data from `27–28` groups (`~80%`) to `20–21` groups (`~60%`), alongside a `3`-fold × `20`-candidate inner search budget. The gap must therefore **not** be attributed solely to historical evaluation optimism.",
        "",
        "## 2. Per-Fold Metrics & Held-Out Calibration (`q90_k`)",
        "",
        "| Fold | `n_test` (`groups`) | XGB MAE (HRC) | XGB RMSE | XGB R² | `B2_Physics` MAE | `n_cal` | `q90_k` (HRC) | Width (`2×q90_k`) | Coverage |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for f in metrics["folds"]:
        xm = f["models"]["XGBoost"]
        bm = f["models"]["B2_Physics"]
        fu = f["uncertainty"]
        n_cal = f.get("calibration", {}).get("n_cal", "-")
        lines.append(
            f"| {f['fold']} | {f['n_test_rows']} ({f['n_test_groups']}) | {xm['MAE']:.4f} | {xm['RMSE']:.4f} | {xm['R2']:.4f} | "
            f"{bm['MAE']:.4f} | {n_cal} | {fu['q90_k']:.4f} | {fu['full_width']:.4f} | {fu['empirical_coverage'] * 100:.2f}% |"
        )

    lines.extend([
        "",
        "## 3. Visualizations",
        "",
        "![Predicted vs Actual Hardness](fig1_predicted_vs_actual.png)",
        "",
        "![Outer-Test MAE by Steel Group](fig2_steel_mae.png)",
        "",
        "![Empirical Coverage by Steel Group](fig3_steel_coverage.png)",
        "",
        "## 4. Per-Steel Breakdown (All 34 Grades; No Cases Excluded)",
        "",
        "| Steel Grade | Fold | `n` | XGB MAE | XGB RMSE | XGB Bias | `B2` MAE | `q90_k` | Coverage (`hit/n`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for s in metrics["per_steel"]:
        flag = " *" if s["small_group_n_lt_30"] else ""
        folds_str = ",".join(str(v) for v in s["folds"])
        lines.append(
            f"| `{s['steel_type']}`{flag} | {folds_str} | {s['n_rows']} | {s['xgb_MAE']:.3f} | {s['xgb_RMSE']:.3f} | "
            f"{s['xgb_bias']:+.3f} | {s['b2_MAE']:.3f} | {s['q90_k']:.3f} | {s['empirical_coverage'] * 100:.1f}% ({s['n_covered']}/{s['n_rows']}) |"
        )

    lines.extend([
        "",
        "*(Asterisk `*` marks small groups with `n < 30`; single-row shifts in `n = 9–19` groups change empirical coverage by `5.3–11.1` percentage points, so strong conclusions should not be drawn from small groups alone.)*",
        "",
        "## 5. Provenance, Sharing Scope & Reproducibility",
        "",
        f"- **Dataset SHA-256 (verified local raw CSV):** `{prov.get('csv_sha256')}`",
        f"- **Evaluation Code Commit:** `{prov.get('git_commit')}` | **Fit Counts:** `{prov.get('fit_counts')}` | **Elapsed Time:** `{prov.get('total_elapsed_seconds'):.3f} s`"
        if isinstance(prov.get("total_elapsed_seconds"), (int, float))
        else f"- **Evaluation Code Commit:** `{prov.get('git_commit')}`",
        "- **Sharing Policy for Per-Row Outputs & Manifests (Why `predictions.csv` / `split_manifest.json` / `run_manifest.json` Are Omitted Here):**",
        "  - Excluding `data/raw/` via `.gitignore` is a repository workflow practice, **not** a license determination or legal sharing ban; similarly, row-index lists in `split_manifest.json` are not confidential or inherently prohibited.",
        "  - General redistribution with attribution under `CC BY 4.0` (the Kaggle dataset license) is distinct from verifying what underlying rights the dataset uploader holds over the compiled historical literature sources (`Hollomon & Jaffe 1945`, `Grange & Baughman 1956`, and ambiguous `Penha 2010`).",
        "  - Because committing per-row Kaggle target outputs (`predictions.csv`), raw split indices (`split_manifest.json`), or local OS runtime dumps (`run_manifest.json`) is not required for this publication package, source provenance over underlying compilations has not been separately cleared, and every omitted file is deterministically reproducible locally (`seed=42`), those three files remain outside Git.",
        "- **1. Full Reproduction From a Clean GitHub Clone (Requires External Source Dataset):**",
        "  A clean GitHub clone alone does **not** include the raw CSV or local P1 run outputs; you must first obtain the dataset independently, verify its SHA-256 checksum, run `p1_evaluation`, and then run `p1_reporting`:",
        "```bash",
        "# 1. Place the Raiipa CSV at data/raw/ and verify its SHA-256 matches:",
        "#    3f88fdccf4bb2f3c03be6eeca03001b4bd8a2b0de5cd4818195486182a4762b0",
        "python -c \"import hashlib; print(hashlib.sha256(open('data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv', 'rb').read()).hexdigest())\"",
        "# 2. Run the full P1 evaluation (writes predictions.csv & manifests outside models/):",
        "python -m materials_data_lab.p1_evaluation --csv \"data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv\" --outdir <local_p1_run_dir> --mode full --max-seconds 900",
        "# 3. Generate summary.json, technical report, and PNG plots from that run:",
        "python -m materials_data_lab.p1_reporting --input-dir <local_p1_run_dir> --outdir reports/p1_grouped_evaluation",
        "```",
        "- **2. Regenerate Report & Charts Only From an Existing Local Run Directory (No Training or CSV Access):**",
        "```bash",
        "python -m materials_data_lab.p1_reporting --input-dir <local_p1_run_dir> --outdir reports/p1_grouped_evaluation",
        "```",
    ])
    return "\n".join(lines) + "\n"


def generate_p1_report(
    input_dir: str | Path,
    outdir: str | Path,
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Read saved P1 artifacts in `input_dir` and write plots, README.md, and summary.json."""
    in_path = Path(input_dir)
    out_path = Path(outdir)

    pred_path = in_path / "predictions.csv"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Required predictions.csv not found in {in_path}")
    pred_df = pd.read_csv(pred_path)

    results_summary = None
    summary_in_path = in_path / "results_summary.json"
    if summary_in_path.is_file():
        results_summary = json.loads(summary_in_path.read_text(encoding="utf-8"))

    run_manifest = None
    run_manifest_path = in_path / "run_manifest.json"
    if run_manifest_path.is_file():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    metrics = compute_reporting_metrics(pred_df, results_summary=results_summary, run_manifest=run_manifest)

    out_path.mkdir(parents=True, exist_ok=True)
    targets = [
        out_path / "summary.json",
        out_path / "README.md",
        out_path / "fig1_predicted_vs_actual.png",
        out_path / "fig2_steel_mae.png",
        out_path / "fig3_steel_coverage.png",
    ]
    if not overwrite:
        for p in targets:
            if p.exists():
                raise FileExistsError(f"Output file already exists (pass overwrite=True): {p}")

    valid_df = validate_predictions_frame(pred_df)
    _plot_pred_vs_actual(valid_df, metrics, out_path / "fig1_predicted_vs_actual.png")
    _plot_steel_mae(metrics, out_path / "fig2_steel_mae.png")
    _plot_steel_coverage(metrics, out_path / "fig3_steel_coverage.png")

    (out_path / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_path / "README.md").write_text(_format_technical_report(metrics), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate reproducible P1 evaluation report and figures.")
    parser.add_argument("--input-dir", type=str, required=True, help="Local directory holding P1 outputs.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="reports/p1_grouped_evaluation",
        help="Target directory for report, summary.json, and PNG figures.",
    )
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if target output files already exist.")
    args = parser.parse_args(argv)

    metrics = generate_p1_report(args.input_dir, args.outdir, overwrite=not args.no_overwrite)
    print(json.dumps({
        "n_rows": metrics["integrity"]["n_prediction_rows"],
        "n_groups": metrics["integrity"]["n_unique_groups"],
        "xgb_pooled_mae": round(metrics["models"]["XGBoost"]["pooled_row_weighted"]["MAE"], 4),
        "global_coverage": round(metrics["uncertainty"]["global_row_weighted_coverage"], 4),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
