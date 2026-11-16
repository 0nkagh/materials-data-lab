"""Model artifact loading and explicit training utilities for Materials Data Lab.

Security & safety policy:
- Runtime loading (`load_artifact`) NEVER triggers automatic retraining when artifacts
  are missing, incomplete, mismatched, or fail to deserialize.
- Training (`train_and_save_artifact`) is strictly opt-in via the explicit training
  entrypoint and refuses to overwrite existing files unless `overwrite=True`.
- `.joblib` files use Python pickle serialization and can execute arbitrary code when
  deserialized: only load artifacts from trusted local `models/` directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from materials_data_lab import __version__
from materials_data_lab.clean_loader import load_clean
from materials_data_lab.modeling import RF_FEATURES, prepare_features

logger = logging.getLogger(__name__)

ARTIFACT_FILENAMES = ("artifact_meta.json", "xgb_champion.joblib", "b2_linear.joblib")


class ArtifactError(RuntimeError):
    """Base error for model artifact lifecycle failures."""


class ArtifactNotReadyError(ArtifactError):
    """Raised when required artifact files or metadata are missing/invalid."""


class ArtifactLoadError(ArtifactError):
    """Raised when an existing artifact fails integrity checks or deserialization."""


class ArtifactOverwriteError(ArtifactError):
    """Raised when training would overwrite existing artifacts without explicit approval."""


def _hash_file(filepath: Path) -> str:
    """Calculate SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _resolve_trusted_file(models_path: Path, filename: str) -> Path:
    """Ensure target artifact file is a regular non-symlink file inside models_path."""
    base_resolved = models_path.resolve()
    candidate = models_path / filename
    if candidate.is_symlink():
        raise ArtifactLoadError(
            f"Refusing to load symlinked artifact path: {candidate} (pickle trust check failed)."
        )
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ArtifactLoadError(
            f"Artifact path escapes trusted directory {base_resolved}: {resolved}"
        ) from exc
    if not candidate.is_file():
        raise ArtifactNotReadyError(f"Missing or non-regular artifact file: {candidate}")
    return candidate


def _collect_version_mismatches(meta: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Compare recorded metadata library versions with active runtime versions.

    Differences are logged as warnings only and do NOT by themselves block loading.
    """
    checks = {
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "xgboost_version": xgb.__version__,
    }
    mismatches: dict[str, dict[str, str]] = {}
    for key, runtime_ver in checks.items():
        art_ver = meta.get(key)
        if art_ver is not None and str(art_ver) != str(runtime_ver):
            mismatches[key] = {"artifact": str(art_ver), "runtime": str(runtime_ver)}
    return mismatches


def _safe_git_commit() -> str | None:
    """Best-effort local HEAD commit lookup for new training provenance."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if proc.returncode == 0:
            sha = proc.stdout.strip()
            return sha or None
    except Exception:
        return None
    return None


def load_artifact(
    models_dir: str | Path = "models",
    csv_path: str | Path | None = None,
    *,
    verify_csv_hash: bool = False,
    require_artifact_hashes: bool = False,
) -> Tuple[dict[str, Any], dict[str, Any]]:
    """Load pre-trained v2 modeling artifacts without any automatic retraining or writes.

    Note: Runtime startup does NOT require raw CSV files unless `verify_csv_hash=True`.
    """
    models_path = Path(models_dir)
    if not models_path.exists() or not models_path.is_dir():
        raise ArtifactNotReadyError(
            f"Models directory does not exist: {models_path}. "
            "Run explicit training: python -m materials_data_lab.model_artifact --csv <raw.csv>"
        )

    missing = [name for name in ARTIFACT_FILENAMES if not (models_path / name).exists()]
    if missing:
        raise ArtifactNotReadyError(
            f"Missing model artifact file(s) in {models_path}: {', '.join(missing)}. "
            "Automatic training is disabled; run explicit training: "
            "python -m materials_data_lab.model_artifact --csv <raw.csv>"
        )

    meta_path = _resolve_trusted_file(models_path, "artifact_meta.json")
    xgb_path = _resolve_trusted_file(models_path, "xgb_champion.joblib")
    b2_path = _resolve_trusted_file(models_path, "b2_linear.joblib")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        logger.exception("Failed to read artifact metadata at %s", meta_path)
        raise ArtifactLoadError(
            f"Could not parse artifact metadata at {meta_path}: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(meta, dict):
        raise ArtifactNotReadyError(f"Invalid metadata format in {meta_path}: expected JSON object.")
    if meta.get("schema_version") != 2 or "conformal_q90" not in meta:
        raise ArtifactNotReadyError(
            f"Incompatible or incomplete artifact schema in {meta_path} "
            f"(schema_version={meta.get('schema_version')!r}, has_conformal_q90={'conformal_q90' in meta}). "
            "Run explicit training with --overwrite to regenerate Schema v2 artifacts."
        )
    if meta.get("features") != RF_FEATURES:
        raise ArtifactNotReadyError(
            f"Artifact feature list mismatch in {meta_path}: {meta.get('features')!r} != {RF_FEATURES!r}."
        )

    if verify_csv_hash:
        if csv_path is None:
            raise ArtifactNotReadyError("verify_csv_hash=True requires csv_path.")
        csv_p = Path(csv_path)
        if not csv_p.is_file():
            raise ArtifactNotReadyError(f"Raw CSV not found for hash verification: {csv_p}")
        csv_sha256 = _hash_file(csv_p)
        if meta.get("csv_sha256") != csv_sha256:
            raise ArtifactNotReadyError(
                f"CSV SHA-256 mismatch: expected {meta.get('csv_sha256')}, got {csv_sha256}. "
                "Automatic retraining is disabled."
            )

    artifact_hashes = meta.get("artifact_hashes")
    if isinstance(artifact_hashes, dict):
        for key, path in (("xgb_champion.joblib", xgb_path), ("b2_linear.joblib", b2_path)):
            expected = artifact_hashes.get(key)
            if expected:
                actual = _hash_file(path)
                if actual != expected:
                    raise ArtifactLoadError(
                        f"Integrity check failed for {path}: expected SHA-256 {expected}, got {actual}."
                    )
    elif require_artifact_hashes:
        raise ArtifactLoadError(
            f"Refusing to load .joblib pickle files without SHA-256 digests in {meta_path}."
        )
    else:
        logger.warning(
            "Loading local .joblib artifacts from %s without artifact_hashes in metadata; "
            "joblib/pickle executes serialized code—ensure artifacts come from a trusted source.",
            models_path,
        )

    mismatches = _collect_version_mismatches(meta)
    if mismatches:
        logger.warning(
            "Artifact runtime version differences detected (proceeding with load): %s",
            mismatches,
        )

    try:
        xgb_model = joblib.load(xgb_path)
        b2_model = joblib.load(b2_path)
    except Exception as exc:
        logger.exception(
            "Failed to load joblib artifacts from %s (version_mismatches=%s)",
            models_path,
            mismatches,
        )
        mismatch_note = f"; runtime version differences: {mismatches}" if mismatches else ""
        raise ArtifactLoadError(
            f"Failed to deserialize model artifact in {models_path}: {type(exc).__name__}: {exc}{mismatch_note}"
        ) from exc

    meta_out = dict(meta)
    meta_out["retrained"] = False
    meta_out["version_mismatches"] = mismatches
    return {"xgb": xgb_model, "b2": b2_model}, meta_out


def train_and_save_artifact(
    csv_path: str | Path,
    models_dir: str | Path = "models",
    *,
    overwrite: bool = False,
) -> Tuple[dict[str, Any], dict[str, Any]]:
    """Explicitly train champion XGBoost + B2 physics baseline and write Schema v2 artifacts.

    Refuses to overwrite existing files in `models_dir` unless `overwrite=True`.
    Algorithm parameters are unchanged from the baseline V2-C2/V3-A specification.
    """
    models_path = Path(models_dir)
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"Training CSV not found: {csv_file}")

    existing = [name for name in ARTIFACT_FILENAMES if (models_path / name).exists()]
    if existing and not overwrite:
        raise ArtifactOverwriteError(
            f"Existing artifact file(s) already present in {models_path}: {', '.join(existing)}. "
            "Pass overwrite=True (or --overwrite) explicitly to replace them."
        )

    csv_sha256 = _hash_file(csv_file)
    raw_df, _ = load_clean(csv_file)
    df = prepare_features(raw_df, keep_initial=False)

    X_rf = df[RF_FEATURES].values
    X_phys = df[["p_hj"]].values
    y = df["final_hrc"].values
    groups = df["steel_type"].values

    best_params = dict(
        subsample=0.7,
        reg_lambda=1,
        n_estimators=800,
        max_depth=7,
        learning_rate=0.03,
        colsample_bytree=1.0,
        random_state=42,
        tree_method="hist",
        n_jobs=1,
    )

    xgb_model = xgb.XGBRegressor(**best_params)
    xgb_model.fit(X_rf, y)

    b2 = LinearRegression()
    b2.fit(X_phys, y)

    # Calculate OOF empirical 90th residual quantile (GroupKFold)
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros_like(y)
    for train_idx, test_idx in gkf.split(X_rf, y, groups):
        cv_model = xgb.XGBRegressor(**best_params)
        cv_model.fit(X_rf[train_idx], y[train_idx])
        oof_preds[test_idx] = cv_model.predict(X_rf[test_idx])

    residuals = np.abs(y - oof_preds)
    q90 = float(np.quantile(residuals, 0.90))

    models_path.mkdir(parents=True, exist_ok=True)
    meta_path = models_path / "artifact_meta.json"
    xgb_path = models_path / "xgb_champion.joblib"
    b2_path = models_path / "b2_linear.joblib"

    joblib.dump(xgb_model, xgb_path)
    joblib.dump(b2, b2_path)

    meta = {
        "schema_version": 2,
        "champion": "XGB_tuned",
        "best_params": best_params,
        "csv_sha256": csv_sha256,
        "package_version": __version__,
        "git_commit": _safe_git_commit(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "xgboost_version": xgb.__version__,
        "artifact_hashes": {
            "xgb_champion.joblib": _hash_file(xgb_path),
            "b2_linear.joblib": _hash_file(b2_path),
        },
        "features": RF_FEATURES,
        "n_rows": len(df),
        "conformal_q90": q90,
        "note": "performance estimates from V2-C2 CV study, not this artifact; OOF pooled quantile (not independent split)",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "retrained": True,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"xgb": xgb_model, "b2": b2}, meta


def build_or_load_artifact(
    csv_path: str | Path | None = None,
    models_dir: str | Path = "models",
    *,
    allow_train: bool = False,
    overwrite: bool = False,
) -> Tuple[dict[str, Any], dict[str, Any]]:
    """Load artifacts safely; training is only allowed when explicitly requested.

    Defaults to `allow_train=False` so callers never trigger silent retraining.
    """
    try:
        return load_artifact(
            models_dir=models_dir,
            csv_path=csv_path,
            verify_csv_hash=bool(allow_train and csv_path is not None),
        )
    except ArtifactNotReadyError:
        if not allow_train:
            raise
        if csv_path is None:
            raise ValueError("csv_path is required when allow_train=True.")
        return train_and_save_artifact(csv_path, models_dir=models_dir, overwrite=overwrite)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly train and save Schema v2 Materials Data Lab model artifacts."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv",
        help="Path to raw CSV dataset.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Target directory for trained artifacts (default: models).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Required to replace existing artifact files in models-dir.",
    )
    args = parser.parse_args(argv)

    _, meta = train_and_save_artifact(args.csv, models_dir=args.models_dir, overwrite=args.overwrite)
    print(f"Trained Schema v2 artifacts into {args.models_dir} (n_rows={meta['n_rows']}, q90={meta['conformal_q90']:.4f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
