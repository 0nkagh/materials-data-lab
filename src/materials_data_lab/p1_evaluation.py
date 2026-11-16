"""P1 scientific evaluation and pilot timing module for Materials Data Lab.

Implements the strict, leakage-free grouped evaluation flow defined in
P1-DEGERLENDIRME-PROTOKOLU.md without touching production API/demo code or models/:
    outer GroupKFold(5) over steel_type
    -> target-blind deterministic GroupShuffleSplit (proper_train / calibration)
    -> inner GroupKFold(3) RandomizedSearchCV on proper_train only (refit=True)
    -> finite-sample ceil((n + 1) * (1 - alpha)) calibration quantile on calibration only
    -> evaluation of that exact proper_train model and band on outer_test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterSampler, RandomizedSearchCV

from materials_data_lab import __version__
from materials_data_lab.clean_loader import load_clean
from materials_data_lab.modeling import COMP_COLS, RF_FEATURES

DEFAULT_CSV_PATH = Path("data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv")
DEFAULT_OUTER_SPLITS = 5
DEFAULT_INNER_SPLITS = 3
DEFAULT_CAL_FRACTION = 0.25
DEFAULT_ALPHA = 0.10
DEFAULT_SEED = 42
DEFAULT_FULL_N_ITER = 20
DEFAULT_PILOT_N_ITER = 5
DEFAULT_MAX_SECONDS = 600.0
GROUP_COVERAGE_DISCLAIMER = (
    "No group-level theoretical coverage guarantee exists for unseen steel_type groups; "
    "finite-sample order statistic calibration is evaluated strictly as an empirical benchmark."
)

XGB_PARAM_SPACE: dict[str, list[Any]] = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1, 2, 5],
}


class PilotTimeoutError(TimeoutError):
    """Raised when P1 evaluation exceeds its configured wall-clock budget."""


class _VerifiedGroupKFold(GroupKFold):
    """GroupKFold wrapper that verifies non-empty groups arrive and stay disjoint inside proper_train."""

    def __init__(self, n_splits: int, expected_groups: set[str]):
        super().__init__(n_splits=n_splits)
        self.expected_groups = expected_groups
        self.verified_folds: list[dict[str, Any]] = []
        self._recorded = False

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Inner GroupKFold requires explicit `groups`; received None.")
        groups_arr = np.asarray(groups, dtype=str)
        observed_set = set(groups_arr.tolist())
        if not observed_set.issubset(self.expected_groups):
            unexpected = sorted(observed_set - self.expected_groups)
            raise ValueError(f"Inner CV received groups outside proper_train: {unexpected}")

        record_now = not self._recorded
        for inner_idx, (tr_idx, val_idx) in enumerate(super().split(X, y=y, groups=groups_arr)):
            tr_groups = set(groups_arr[tr_idx].tolist())
            val_groups = set(groups_arr[val_idx].tolist())
            overlap = tr_groups & val_groups
            if overlap:
                raise AssertionError(f"Inner fold {inner_idx} leaked groups: {sorted(overlap)}")
            if tr_groups | val_groups != observed_set:
                raise AssertionError(f"Inner fold {inner_idx} dropped or altered proper_train groups.")
            if record_now:
                self.verified_folds.append({
                    "inner_fold": inner_idx,
                    "n_train_rows": int(len(tr_idx)),
                    "n_val_rows": int(len(val_idx)),
                    "n_train_groups": int(len(tr_groups)),
                    "n_val_groups": int(len(val_groups)),
                    "train_groups": sorted(tr_groups),
                    "val_groups": sorted(val_groups),
                })
            yield tr_idx, val_idx
        self._recorded = True


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _safe_git_info(repo_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"git_commit": None, "git_dirty": None}
    try:
        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty_lines = [line for line in status_res.stdout.splitlines() if line.strip()]
        info["git_commit"] = commit_res.stdout.strip()
        info["git_dirty"] = len(dirty_lines) > 0
        info["git_dirty_paths"] = dirty_lines
    except Exception as exc:
        info["git_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _assert_safe_outdir(outdir: Path, repo_root: Path | None = None) -> None:
    resolved_out = outdir.resolve()
    candidate_models = [Path("models").resolve()]
    if repo_root is not None:
        candidate_models.append((repo_root / "models").resolve())
    for models_dir in candidate_models:
        if resolved_out == models_dir or models_dir in resolved_out.parents:
            raise ValueError(f"Refusing to write P1 evaluation outputs inside protected models directory: {outdir}")


def _write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output file without overwrite=True: {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def generate_xgb_candidates(
    n_iter: int,
    *,
    seed: int = DEFAULT_SEED,
    canonical_full_n_iter: int = DEFAULT_FULL_N_ITER,
    param_space: dict[str, list[Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate deterministic hyperparameter candidates where pilot draws equal full_run[:n_iter].

    In scikit-learn's ParameterSampler, changing n_iter changes the random_state draws.
    Sampling `max(canonical_full_n_iter, n_iter)` once and slicing `[:n_iter]` makes the
    pilot budget an exact deterministic prefix of the canonical full-run candidate sequence.
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    if canonical_full_n_iter < 1:
        raise ValueError(f"canonical_full_n_iter must be >= 1, got {canonical_full_n_iter}")

    space = dict(param_space) if param_space is not None else dict(XGB_PARAM_SPACE)
    base_budget = max(canonical_full_n_iter, n_iter)
    full_sequence = list(ParameterSampler(space, n_iter=base_budget, random_state=seed))
    selected = full_sequence[:n_iter]
    canonical_slice = full_sequence[:canonical_full_n_iter]

    meta = {
        "seed": seed,
        "requested_n_iter": int(n_iter),
        "actual_n_iter": int(len(selected)),
        "canonical_full_n_iter": int(canonical_full_n_iter),
        "pilot_is_exact_prefix_of_full": selected == canonical_slice[: len(selected)],
        "relationship_note": (
            "Selected candidates equal canonical_sequence[:actual_n_iter] generated once with "
            f"ParameterSampler(n_iter={base_budget}, random_state={seed}), so pilot (n_iter=5) "
            "is an exact prefix [0:5] of the canonical full-run (n_iter=20) sequence."
        ),
        "selected_candidates": selected,
        "canonical_full_candidates": canonical_slice,
    }
    return selected, meta


def load_and_prepare_p1_data(csv_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load raw CSV deterministically, preserve original row IDs, and filter invalid/missing modeling rows."""
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_file}")

    csv_sha256 = _sha256_file(csv_file)
    raw_df, clean_manifest = load_clean(csv_file)

    df = raw_df.copy()
    df["orig_row_id"] = np.arange(len(df), dtype=int)

    # D-08: exclude initial_hrc (and missingness marker unused by modeling features)
    if "initial_hrc" in df.columns:
        df = df.drop(columns=["initial_hrc"])

    numeric_cols = COMP_COLS + ["temper_temp_c", "temper_time_s", "final_hrc"]
    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Expected modeling column missing after clean_loader: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    finite_mask = np.ones(len(df), dtype=bool)
    for col in numeric_cols:
        finite_mask &= np.isfinite(df[col].to_numpy(dtype=float))

    steel_valid = df["steel_type"].notna() & (df["steel_type"].astype(str).str.strip() != "")
    time_valid = df["temper_time_s"].to_numpy(dtype=float) > 0.0
    temp_valid = df["temper_temp_c"].to_numpy(dtype=float) > -273.15

    keep_mask = finite_mask & steel_valid.to_numpy(dtype=bool) & time_valid & temp_valid
    dropped_df = df.loc[~keep_mask]

    filtered = df.loc[keep_mask].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No valid modeling rows remain after deterministic P1 filtering.")

    filtered["log_time"] = np.log10(filtered["temper_time_s"].to_numpy(dtype=float))
    temp_k = filtered["temper_temp_c"].to_numpy(dtype=float) + 273.15
    t_hours = filtered["temper_time_s"].to_numpy(dtype=float) / 3600.0
    filtered["p_hj"] = (temp_k / 1000.0) * (19.5 + np.log10(t_hours))
    filtered["steel_type"] = filtered["steel_type"].astype(str)
    if "source" not in filtered.columns:
        filtered["source"] = "UNKNOWN_SOURCE"
    else:
        filtered["source"] = filtered["source"].astype(str)

    prep_manifest = {
        "csv_path": str(csv_file),
        "csv_sha256": csv_sha256,
        "raw_rows": int(len(raw_df)),
        "kept_rows": int(len(filtered)),
        "dropped_rows": int(len(dropped_df)),
        "dropped_orig_row_ids": [int(i) for i in dropped_df["orig_row_id"].tolist()],
        "drop_reason_counts": {
            "non_finite_or_missing_numeric": int((~finite_mask).sum()),
            "missing_steel_type": int((~steel_valid).sum()),
            "non_positive_temper_time_s": int((finite_mask & ~time_valid).sum()),
            "non_positive_kelvin_temp": int((finite_mask & ~temp_valid).sum()),
        },
        "n_unique_groups": int(filtered["steel_type"].nunique()),
        "preserved_below_20_hrc_rows": int((filtered["final_hrc"] < 20.0).sum()),
        "clean_loader_manifest": clean_manifest,
    }
    return filtered, prep_manifest


def _assert_disjoint_groups(splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    names = list(splits.keys())
    for i in range(len(names)):
        idx_a, grp_a = splits[names[i]]
        set_idx_a = set(idx_a.tolist())
        set_grp_a = set(grp_a.tolist())
        for j in range(i + 1, len(names)):
            idx_b, grp_b = splits[names[j]]
            row_overlap = set_idx_a & set(idx_b.tolist())
            grp_overlap = set_grp_a & set(grp_b.tolist())
            if row_overlap:
                raise AssertionError(f"Row leakage between {names[i]} and {names[j]}: {len(row_overlap)} rows")
            if grp_overlap:
                raise AssertionError(
                    f"Group leakage between {names[i]} and {names[j]}: {sorted(grp_overlap)}"
                )


def _group_source_counts(df_slice: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df_slice["source"].value_counts().sort_index().items()}


def build_split_manifest(
    feature_df_no_target: pd.DataFrame,
    *,
    n_outer_splits: int = DEFAULT_OUTER_SPLITS,
    n_inner_splits: int = DEFAULT_INNER_SPLITS,
    cal_fraction: float = DEFAULT_CAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Pre-commit target-blind deterministic outer, calibration, and inner splits.

    Target values (`final_hrc`) are intentionally excluded from inputs so calibration
    or test targets can never influence any partition.
    """
    required_cols = {"orig_row_id", "steel_type"}
    if not required_cols.issubset(feature_df_no_target.columns):
        missing = sorted(required_cols - set(feature_df_no_target.columns))
        raise ValueError(f"Missing required manifest columns: {missing}")
    if "final_hrc" in feature_df_no_target.columns:
        feature_df_no_target = feature_df_no_target.drop(columns=["final_hrc"])
    if not (0.0 < cal_fraction < 1.0):
        raise ValueError(f"cal_fraction must be in (0, 1), got {cal_fraction}")

    groups = feature_df_no_target["steel_type"].to_numpy(dtype=str)
    orig_ids = feature_df_no_target["orig_row_id"].to_numpy(dtype=int)
    pos_indices = np.arange(len(feature_df_no_target), dtype=int)
    has_source = "source" in feature_df_no_target.columns

    n_total_groups = len(np.unique(groups))
    if n_total_groups < n_outer_splits:
        raise ValueError(f"Need at least {n_outer_splits} groups for outer GroupKFold, found {n_total_groups}.")

    outer_gkf = GroupKFold(n_splits=n_outer_splits)
    folds_manifest: list[dict[str, Any]] = []

    for fold_idx, (outer_tr_pos, outer_te_pos) in enumerate(outer_gkf.split(pos_indices, groups=groups)):
        outer_tr_groups = groups[outer_tr_pos]
        outer_te_groups = groups[outer_te_pos]

        gss = GroupShuffleSplit(n_splits=1, test_size=cal_fraction, random_state=seed)
        pt_rel, cal_rel = next(gss.split(outer_tr_pos, groups=outer_tr_groups))
        proper_tr_pos = outer_tr_pos[pt_rel]
        cal_pos = outer_tr_pos[cal_rel]

        _assert_disjoint_groups({
            "proper_train": (proper_tr_pos, groups[proper_tr_pos]),
            "calibration": (cal_pos, groups[cal_pos]),
            "outer_test": (outer_te_pos, outer_te_groups),
        })

        n_pt_groups = int(len(np.unique(groups[proper_tr_pos])))
        if n_pt_groups < n_inner_splits:
            raise ValueError(
                f"Outer fold {fold_idx} has only {n_pt_groups} groups in proper_train, "
                f"fewer than n_inner_splits={n_inner_splits}."
            )

        inner_gkf = GroupKFold(n_splits=n_inner_splits)
        inner_folds: list[dict[str, Any]] = []
        for in_idx, (in_tr_rel, in_val_rel) in enumerate(
            inner_gkf.split(proper_tr_pos, groups=groups[proper_tr_pos])
        ):
            in_tr_pos = proper_tr_pos[in_tr_rel]
            in_val_pos = proper_tr_pos[in_val_rel]
            _assert_disjoint_groups({
                "inner_train": (in_tr_pos, groups[in_tr_pos]),
                "inner_val": (in_val_pos, groups[in_val_pos]),
                "calibration": (cal_pos, groups[cal_pos]),
                "outer_test": (outer_te_pos, outer_te_groups),
            })
            inner_folds.append({
                "inner_fold": int(in_idx),
                "n_train_rows": int(len(in_tr_pos)),
                "n_val_rows": int(len(in_val_pos)),
                "n_train_groups": int(len(np.unique(groups[in_tr_pos]))),
                "n_val_groups": int(len(np.unique(groups[in_val_pos]))),
                "train_groups": sorted(set(groups[in_tr_pos].tolist())),
                "val_groups": sorted(set(groups[in_val_pos].tolist())),
            })

        fold_entry: dict[str, Any] = {
            "fold": int(fold_idx),
            "proper_train_pos": [int(i) for i in proper_tr_pos.tolist()],
            "calibration_pos": [int(i) for i in cal_pos.tolist()],
            "outer_test_pos": [int(i) for i in outer_te_pos.tolist()],
            "proper_train_orig_row_ids": [int(i) for i in orig_ids[proper_tr_pos].tolist()],
            "calibration_orig_row_ids": [int(i) for i in orig_ids[cal_pos].tolist()],
            "outer_test_orig_row_ids": [int(i) for i in orig_ids[outer_te_pos].tolist()],
            "counts": {
                "outer_train_rows": int(len(outer_tr_pos)),
                "outer_train_groups": int(len(np.unique(outer_tr_groups))),
                "proper_train_rows": int(len(proper_tr_pos)),
                "proper_train_groups": n_pt_groups,
                "calibration_rows": int(len(cal_pos)),
                "calibration_groups": int(len(np.unique(groups[cal_pos]))),
                "outer_test_rows": int(len(outer_te_pos)),
                "outer_test_groups": int(len(np.unique(outer_te_groups))),
            },
            "groups": {
                "proper_train": sorted(set(groups[proper_tr_pos].tolist())),
                "calibration": sorted(set(groups[cal_pos].tolist())),
                "outer_test": sorted(set(outer_te_groups.tolist())),
            },
            "inner_folds": inner_folds,
        }
        if has_source:
            fold_entry["sources"] = {
                "proper_train": _group_source_counts(feature_df_no_target.iloc[proper_tr_pos]),
                "calibration": _group_source_counts(feature_df_no_target.iloc[cal_pos]),
                "outer_test": _group_source_counts(feature_df_no_target.iloc[outer_te_pos]),
            }
        folds_manifest.append(fold_entry)

    return {
        "target_blind": True,
        "seed": int(seed),
        "n_outer_splits": int(n_outer_splits),
        "n_inner_splits": int(n_inner_splits),
        "cal_fraction": float(cal_fraction),
        "n_rows": int(len(feature_df_no_target)),
        "n_groups": int(n_total_groups),
        "folds": folds_manifest,
    }


def compute_conformal_quantile(
    abs_residuals: Iterable[float] | np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    overflow_policy: Literal["inf", "cap_max", "raise"] = "inf",
) -> dict[str, Any]:
    """Compute finite-sample conformal order statistic ceil((n + 1) * (1 - alpha)) on |y - y_hat|.

    When rank > n (i.e. n < (1 - alpha) / alpha; for alpha=0.10, n <= 8), no finite
    marginal distribution-free order statistic exists at level 1 - alpha:
    - overflow_policy="inf" (default): returns math.inf with finite_sample_feasible=False.
    - overflow_policy="cap_max": returns max(abs_residuals) while flagging rank_exceeded_n=True.
    - overflow_policy="raise": raises ValueError.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
    if overflow_policy not in ("inf", "cap_max", "raise"):
        raise ValueError(f"Unknown overflow_policy: {overflow_policy!r}")

    arr = np.asarray(list(abs_residuals), dtype=float)
    n = int(arr.size)
    if n == 0:
        raise ValueError("Cannot compute conformal quantile on empty calibration residuals.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Calibration residuals contain NaN or Infinity.")
    if np.any(arr < 0.0):
        raise ValueError("Calibration residuals must be non-negative absolute values.")

    sorted_r = np.sort(arr)
    rank_1based = int(math.ceil((n + 1) * (1.0 - alpha)))
    rank_exceeded = rank_1based > n

    if not rank_exceeded:
        q_val = float(sorted_r[rank_1based - 1])
    elif overflow_policy == "inf":
        q_val = float("inf")
    elif overflow_policy == "cap_max":
        q_val = float(sorted_r[-1])
    else:
        raise ValueError(
            f"Calibration sample size n={n} is too small for alpha={alpha}: "
            f"ceil((n+1)*(1-alpha))={rank_1based} > n."
        )

    return {
        "q": q_val,
        "alpha": float(alpha),
        "nominal_level": float(1.0 - alpha),
        "n_cal": n,
        "rank_1based": rank_1based,
        "rank_exceeded_n": rank_exceeded,
        "finite_sample_feasible": not rank_exceeded,
        "overflow_policy": overflow_policy,
        "max_cal_residual": float(sorted_r[-1]),
        "coverage_note": GROUP_COVERAGE_DISCLAIMER,
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute MAE, RMSE, R2 (when >= 2 samples), and mean signed bias."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if y_t.size == 0 or y_t.size != y_p.size:
        raise ValueError("y_true and y_pred must be non-empty arrays of identical length.")
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(math.sqrt(mean_squared_error(y_t, y_p)))
    bias = float(np.mean(y_p - y_t))
    r2 = float(r2_score(y_t, y_p)) if y_t.size >= 2 and np.var(y_t) > 0.0 else None
    return {
        "n_rows": int(y_t.size),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "mean_signed_error": bias,
    }


def summarize_evaluation_results(
    fold_records: list[dict[str, Any]],
    predictions_df: pd.DataFrame,
) -> dict[str, Any]:
    """Aggregate per-fold, fold-mean ± std, pooled outer-test, and per-group metrics."""
    if not fold_records or predictions_df.empty:
        raise ValueError("Cannot summarize empty fold records or predictions.")

    models_summary: dict[str, Any] = {}
    for model_key, pred_col in (("B1_Naive", "y_pred_b1"), ("B2_Physics", "y_pred_b2"), ("XGBoost", "y_pred_xgb")):
        fold_maes = [float(f["models"][model_key]["MAE"]) for f in fold_records]
        fold_rmses = [float(f["models"][model_key]["RMSE"]) for f in fold_records]
        fold_r2s = [f["models"][model_key]["R2"] for f in fold_records if f["models"][model_key]["R2"] is not None]
        pooled = compute_regression_metrics(predictions_df["y_true"].to_numpy(), predictions_df[pred_col].to_numpy())
        models_summary[model_key] = {
            "fold_mean": {
                "MAE_mean": float(np.mean(fold_maes)),
                "MAE_std_ddof0": float(np.std(fold_maes, ddof=0)),
                "RMSE_mean": float(np.mean(fold_rmses)),
                "RMSE_std_ddof0": float(np.std(fold_rmses, ddof=0)),
                "R2_mean": float(np.mean(fold_r2s)) if fold_r2s else None,
                "R2_std_ddof0": float(np.std(fold_r2s, ddof=0)) if fold_r2s else None,
            },
            "pooled_outer_test": pooled,
        }

    fold_covs = [float(f["uncertainty"]["empirical_coverage"]) for f in fold_records]
    fold_q90s = [float(f["uncertainty"]["q90_k"]) for f in fold_records]
    fold_widths = [float(f["uncertainty"]["full_width"]) for f in fold_records]

    per_group: dict[str, Any] = {}
    for grp_name, grp_df in predictions_df.groupby("steel_type", sort=True):
        m = compute_regression_metrics(grp_df["y_true"].to_numpy(), grp_df["y_pred_xgb"].to_numpy())
        m["empirical_coverage"] = float(np.mean(grp_df["covered"].to_numpy(dtype=bool)))
        per_group[str(grp_name)] = m

    return {
        "evaluated_folds": [int(f["fold"]) for f in fold_records],
        "n_evaluated_folds": int(len(fold_records)),
        "n_evaluated_rows": int(len(predictions_df)),
        "models": models_summary,
        "uncertainty": {
            "pooled_empirical_coverage": float(np.mean(predictions_df["covered"].to_numpy(dtype=bool))),
            "fold_mean_coverage": float(np.mean(fold_covs)),
            "fold_std_coverage_ddof0": float(np.std(fold_covs, ddof=0)),
            "mean_q90": float(np.mean(fold_q90s)),
            "mean_full_width": float(np.mean(fold_widths)),
            "coverage_note": GROUP_COVERAGE_DISCLAIMER,
        },
        "per_group_xgb": per_group,
    }


def _default_xgb_factory(seed: int) -> BaseEstimator:
    return xgb.XGBRegressor(random_state=seed, tree_method="hist", n_jobs=1)


def _check_timeout(start_mono: float, max_seconds: float | None, stage: str) -> None:
    if max_seconds is None:
        return
    elapsed = time.monotonic() - start_mono
    if elapsed > max_seconds:
        raise PilotTimeoutError(
            f"P1 evaluation exceeded time limit ({elapsed:.2f}s > {max_seconds:.2f}s) at stage '{stage}'."
        )


def run_p1_fold(
    feature_df: pd.DataFrame,
    fold_manifest: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    n_inner_splits: int = DEFAULT_INNER_SPLITS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    overflow_policy: Literal["inf", "cap_max", "raise"] = "inf",
    estimator_factory: Callable[[int], BaseEstimator] | None = None,
    start_monotonic: float | None = None,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Execute steps C through G strictly inside a pre-committed outer fold."""
    t_fold_start = time.monotonic()
    t_budget_ref = start_monotonic if start_monotonic is not None else t_fold_start
    _check_timeout(t_budget_ref, max_seconds, f"fold_{fold_manifest['fold']}_start")

    pt_pos = np.asarray(fold_manifest["proper_train_pos"], dtype=int)
    cal_pos = np.asarray(fold_manifest["calibration_pos"], dtype=int)
    te_pos = np.asarray(fold_manifest["outer_test_pos"], dtype=int)

    groups_all = feature_df["steel_type"].to_numpy(dtype=str)
    _assert_disjoint_groups({
        "proper_train": (pt_pos, groups_all[pt_pos]),
        "calibration": (cal_pos, groups_all[cal_pos]),
        "outer_test": (te_pos, groups_all[te_pos]),
    })

    X_rf = feature_df[RF_FEATURES].to_numpy(dtype=float)
    X_phys = feature_df[["p_hj"]].to_numpy(dtype=float)
    y = feature_df["final_hrc"].to_numpy(dtype=float)

    X_pt, y_pt, g_pt = X_rf[pt_pos], y[pt_pos], groups_all[pt_pos]
    X_cal, y_cal = X_rf[cal_pos], y[cal_pos]
    X_te, y_te = X_rf[te_pos], y[te_pos]

    # Baselines fit ONLY on proper_train (never touching calibration or outer_test targets)
    t_ref0 = time.monotonic()
    b1_model = DummyRegressor(strategy="median").fit(X_pt, y_pt)
    b1_preds = b1_model.predict(X_te)

    b2_model = LinearRegression().fit(X_phys[pt_pos], y_pt)
    b2_preds = b2_model.predict(X_phys[te_pos])
    t_baselines = time.monotonic() - t_ref0

    _check_timeout(t_budget_ref, max_seconds, f"fold_{fold_manifest['fold']}_after_baselines")

    # Inner grouped hyperparameter search strictly on proper_train using refit=True (single final proper_train fit)
    factory = estimator_factory if estimator_factory is not None else _default_xgb_factory
    base_estimator = clone(factory(seed))
    param_distributions = [{k: [v] for k, v in cand.items()} for cand in candidates]
    inner_cv = _VerifiedGroupKFold(n_splits=n_inner_splits, expected_groups=set(g_pt.tolist()))

    t_search0 = time.monotonic()
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=len(candidates),
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        refit=True,
        random_state=seed,
        n_jobs=1,
    )
    search.fit(X_pt, y_pt, groups=g_pt)
    t_search = time.monotonic() - t_search0

    _check_timeout(t_budget_ref, max_seconds, f"fold_{fold_manifest['fold']}_after_search")

    if len(inner_cv.verified_folds) != n_inner_splits:
        raise AssertionError(
            f"Expected {n_inner_splits} verified inner folds, observed {len(inner_cv.verified_folds)}"
        )

    # Use search.best_estimator_ directly (Step D refit already done via refit=True; NEVER refit on calibration)
    proper_model = search.best_estimator_

    # Step E: calibrate empirical quantile exclusively on calibration slice
    y_cal_pred = np.asarray(proper_model.predict(X_cal), dtype=float)
    abs_cal_res = np.abs(y_cal - y_cal_pred)
    q_info = compute_conformal_quantile(abs_cal_res, alpha=alpha, overflow_policy=overflow_policy)
    q90_k = float(q_info["q"])

    # Step F: evaluate exact same proper_train model and q90_k band on outer_test
    xgb_te_preds = np.asarray(proper_model.predict(X_te), dtype=float)
    abs_te_err = np.abs(y_te - xgb_te_preds)
    covered = abs_te_err <= q90_k
    full_width = 2.0 * q90_k if math.isfinite(q90_k) else float("inf")

    expected_xgb_fits = int(len(candidates) * n_inner_splits + 1)
    fold_elapsed = time.monotonic() - t_fold_start

    te_df = feature_df.iloc[te_pos]
    pred_df = pd.DataFrame({
        "orig_row_id": te_df["orig_row_id"].to_numpy(dtype=int),
        "fold": int(fold_manifest["fold"]),
        "steel_type": te_df["steel_type"].to_numpy(dtype=str),
        "source": te_df["source"].to_numpy(dtype=str),
        "y_true": y_te,
        "y_pred_b1": b1_preds,
        "y_pred_b2": b2_preds,
        "y_pred_xgb": xgb_te_preds,
        "q90_k": q90_k,
        "interval_low": xgb_te_preds - q90_k if math.isfinite(q90_k) else -float("inf"),
        "interval_high": xgb_te_preds + q90_k if math.isfinite(q90_k) else float("inf"),
        "abs_error_xgb": abs_te_err,
        "covered": covered,
    })

    fold_record = {
        "fold": int(fold_manifest["fold"]),
        "best_params": search.best_params_,
        "best_inner_cv_mae": float(-search.best_score_),
        "fit_counts": {
            "inner_cv_fits": int(len(candidates) * n_inner_splits),
            "proper_train_refit_via_search": 1,
            "xgb_total_fits": expected_xgb_fits,
            "baseline_fits": 2,
            "total_fits": int(expected_xgb_fits + 2),
        },
        "timing_seconds": {
            "baselines_fit_predict": float(t_baselines),
            "inner_search_plus_refit": float(t_search),
            "fold_total": float(fold_elapsed),
        },
        "verified_inner_folds": inner_cv.verified_folds,
        "calibration": q_info,
        "models": {
            "B1_Naive": compute_regression_metrics(y_te, b1_preds),
            "B2_Physics": compute_regression_metrics(y_te, b2_preds),
            "XGBoost": compute_regression_metrics(y_te, xgb_te_preds),
        },
        "uncertainty": {
            "q90_k": q90_k,
            "full_width": full_width,
            "empirical_coverage": float(np.mean(covered)),
            "n_test_rows": int(len(y_te)),
            "n_test_groups": int(fold_manifest["counts"]["outer_test_groups"]),
        },
    }
    return fold_record, pred_df


def run_p1_evaluation(
    csv_path: str | Path,
    outdir: str | Path,
    *,
    mode: Literal["pilot", "full"] = "pilot",
    fold_indices: list[int] | None = None,
    n_iter: int | None = None,
    canonical_full_n_iter: int = DEFAULT_FULL_N_ITER,
    n_outer_splits: int = DEFAULT_OUTER_SPLITS,
    n_inner_splits: int = DEFAULT_INNER_SPLITS,
    cal_fraction: float = DEFAULT_CAL_FRACTION,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    overflow_policy: Literal["inf", "cap_max", "raise"] = "inf",
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    overwrite: bool = False,
    param_space: dict[str, list[Any]] | None = None,
    estimator_factory: Callable[[int], BaseEstimator] | None = None,
) -> dict[str, Any]:
    """Run P1 grouped evaluation (`pilot` = fold 0 & 5 candidates; `full` = 5 folds & 20 candidates)."""
    out_path = Path(outdir)
    repo_root = Path(__file__).resolve().parents[2]
    _assert_safe_outdir(out_path, repo_root=repo_root)

    if mode not in ("pilot", "full"):
        raise ValueError(f"mode must be 'pilot' or 'full', got {mode!r}")

    selected_n_iter = n_iter if n_iter is not None else (DEFAULT_PILOT_N_ITER if mode == "pilot" else DEFAULT_FULL_N_ITER)
    selected_folds = list(fold_indices) if fold_indices is not None else ([0] if mode == "pilot" else list(range(n_outer_splits)))
    if not selected_folds or any(f < 0 or f >= n_outer_splits for f in selected_folds):
        raise ValueError(f"Invalid fold_indices {selected_folds} for n_outer_splits={n_outer_splits}")

    t_total0 = time.monotonic()
    feature_df, prep_manifest = load_and_prepare_p1_data(csv_path)
    split_manifest = build_split_manifest(
        feature_df.drop(columns=["final_hrc"]),
        n_outer_splits=n_outer_splits,
        n_inner_splits=n_inner_splits,
        cal_fraction=cal_fraction,
        seed=seed,
    )
    candidates, cand_meta = generate_xgb_candidates(
        selected_n_iter,
        seed=seed,
        canonical_full_n_iter=canonical_full_n_iter,
        param_space=param_space,
    )

    fold_records: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    timed_out = False
    timeout_error: str | None = None

    for fold_idx in selected_folds:
        try:
            fold_rec, pred_df = run_p1_fold(
                feature_df,
                split_manifest["folds"][fold_idx],
                candidates=candidates,
                n_inner_splits=n_inner_splits,
                alpha=alpha,
                seed=seed,
                overflow_policy=overflow_policy,
                estimator_factory=estimator_factory,
                start_monotonic=t_total0,
                max_seconds=max_seconds,
            )
            fold_records.append(fold_rec)
            pred_frames.append(pred_df)
        except PilotTimeoutError as exc:
            timed_out = True
            timeout_error = str(exc)
            break

    total_elapsed = time.monotonic() - t_total0
    out_path.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    preds_all = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    if not preds_all.empty:
        summary = summarize_evaluation_results(fold_records, preds_all)
        preds_path = out_path / "predictions.csv"
        if preds_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output file without overwrite=True: {preds_path}")
        preds_all.to_csv(preds_path, index=False)

    total_xgb_fits = int(sum(f["fit_counts"]["xgb_total_fits"] for f in fold_records))
    total_baseline_fits = int(sum(f["fit_counts"]["baseline_fits"] for f in fold_records))

    run_manifest = {
        "schema_version": 1,
        "mode": mode,
        "purpose_note": (
            "Timing and workflow verification pilot only; do not select models or search budgets "
            "using pilot outer-test scores."
            if mode == "pilot"
            else "Unbiased grouped nested CV + separate split-calibration evaluation."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "joblib": joblib.__version__,
            "search_n_jobs": 1,
            "xgb_n_jobs": 1,
        },
        "git": _safe_git_info(repo_root),
        "data_preparation": prep_manifest,
        "parameters": {
            "outer_splits": int(n_outer_splits),
            "inner_splits": int(n_inner_splits),
            "cal_fraction": float(cal_fraction),
            "alpha": float(alpha),
            "seed": int(seed),
            "overflow_policy": overflow_policy,
            "selected_folds": selected_folds,
            "max_seconds": max_seconds,
        },
        "candidates": cand_meta,
        "fit_counts": {
            "xgb_fits": total_xgb_fits,
            "baseline_fits": total_baseline_fits,
            "total_fits": int(total_xgb_fits + total_baseline_fits),
        },
        "timing": {
            "total_elapsed_seconds": float(total_elapsed),
            "timed_out": timed_out,
            "timeout_error": timeout_error,
        },
        "coverage_note": GROUP_COVERAGE_DISCLAIMER,
    }

    results_payload = {
        "mode": mode,
        "timed_out": timed_out,
        "timeout_error": timeout_error,
        "total_elapsed_seconds": float(total_elapsed),
        "fit_counts": run_manifest["fit_counts"],
        "folds": fold_records,
        "aggregate": summary,
    }

    _write_json(out_path / "split_manifest.json", split_manifest, overwrite=overwrite)
    _write_json(out_path / "run_manifest.json", run_manifest, overwrite=overwrite)
    _write_json(out_path / "results_summary.json", results_payload, overwrite=overwrite)
    return results_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run P1 grouped nested CV + split-calibration evaluation.")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV_PATH), help="Path to cleanable raw CSV dataset.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory outside models/.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot", help="Pilot (fold 0, 5 iter) or full.")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS, help="Wall-clock budget in seconds.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files if present.")
    args = parser.parse_args(argv)

    res = run_p1_evaluation(
        csv_path=args.csv,
        outdir=args.outdir,
        mode=args.mode,
        max_seconds=args.max_seconds,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "mode": res["mode"],
        "timed_out": res["timed_out"],
        "elapsed_seconds": round(res["total_elapsed_seconds"], 3),
        "fit_counts": res["fit_counts"],
    }, indent=2))
    return 1 if res["timed_out"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
