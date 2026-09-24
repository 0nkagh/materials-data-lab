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
from sklearn.ensemble import RandomForestRegressor
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
    """Builds or loads the modeling artifact."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    meta_path = models_path / "artifact_meta.json"
    rf_path = models_path / "rf_final.joblib"
    b2_path = models_path / "b2_linear.joblib"
    
    csv_sha256 = _hash_file(csv_path)
    
    # Try to load existing artifact
    if meta_path.exists() and rf_path.exists() and b2_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            if meta.get("csv_sha256") == csv_sha256:
                # Valid artifact found
                models = {
                    "rf": joblib.load(rf_path),
                    "b2": joblib.load(b2_path)
                }
                meta["retrained"] = False
                return models, meta
        except Exception:
            pass # Fall back to training
            
    # Need to retrain
    raw_df, _ = load_clean(csv_path)
    df = prepare_features(raw_df, keep_initial=False)
    
    X_rf = df[RF_FEATURES].values
    X_phys = df[["p_hj"]].values
    y = df["final_hrc"].values
    
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_rf, y)
    
    b2 = LinearRegression()
    b2.fit(X_phys, y)
    
    # Save artifacts
    joblib.dump(rf, rf_path)
    joblib.dump(b2, b2_path)
    
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "csv_sha256": csv_sha256,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "features": RF_FEATURES,
        "n_rows": len(df),
        "model_type": "RandomForestRegressor + LinearRegression(p_hj)",
        "rf_params": rf.get_params(),
        "note": "trained on full data; performance estimates from Phase 4 CV study, not this artifact",
        "retrained": True
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    models = {
        "rf": rf,
        "b2": b2
    }
    
    return models, meta
