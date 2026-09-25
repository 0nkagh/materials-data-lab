import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from materials_data_lab.clean_loader import load_clean
from materials_data_lab.modeling import RF_FEATURES, prepare_features


def _hash_file(filepath: Path) -> str:
    """Calculate SHA256 of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def build_or_load_artifact(csv_path: Path, models_dir: str | Path = "models") -> Tuple[dict[str, Any], dict[str, Any]]:
    """Builds or loads the modeling artifact (Schema v2)."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    meta_path = models_path / "artifact_meta.json"
    xgb_path = models_path / "xgb_champion.joblib"
    b2_path = models_path / "b2_linear.joblib"
    
    csv_sha256 = _hash_file(csv_path)
    
    # Try to load existing v2 artifact
    if meta_path.exists() and xgb_path.exists() and b2_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            # Check schema version, csv hash, and conformal field
            if meta.get("schema_version") == 2 and meta.get("csv_sha256") == csv_sha256 and "conformal_q90" in meta:
                # Valid artifact found
                models = {
                    "xgb": joblib.load(xgb_path),
                    "b2": joblib.load(b2_path)
                }
                meta["retrained"] = False
                return models, meta
        except Exception:
            pass # Fall back to training
            
    # Need to retrain (either v1 exists, missing files, or hash mismatch)
    raw_df, _ = load_clean(csv_path)
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
        n_jobs=1
    )
    
    xgb_model = xgb.XGBRegressor(**best_params)
    xgb_model.fit(X_rf, y)
    
    b2 = LinearRegression()
    b2.fit(X_phys, y)
    
    # Save artifacts
    joblib.dump(xgb_model, xgb_path)
    joblib.dump(b2, b2_path)
    
    # Calculate conformal q90
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros_like(y)
    for train_idx, test_idx in gkf.split(X_rf, y, groups):
        cv_model = xgb.XGBRegressor(**best_params)
        cv_model.fit(X_rf[train_idx], y[train_idx])
        oof_preds[test_idx] = cv_model.predict(X_rf[test_idx])
        
    residuals = np.abs(y - oof_preds)
    q90 = float(np.quantile(residuals, 0.90))
    
    meta = {
        "schema_version": 2,
        "champion": "XGB_tuned",
        "best_params": best_params,
        "csv_sha256": csv_sha256,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "xgboost_version": xgb.__version__,
        "features": RF_FEATURES,
        "n_rows": len(df),
        "conformal_q90": q90,
        "note": "performance estimates from V2-C2 CV study, not this artifact",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "retrained": True
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    models = {
        "xgb": xgb_model,
        "b2": b2
    }
    
    return models, meta
