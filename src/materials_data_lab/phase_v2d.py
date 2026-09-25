"""Phase V2-D: Steel Class Analysis."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

# Must be before pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

from materials_data_lab.clean_loader import load_clean
from materials_data_lab.modeling import RF_FEATURES, prepare_features
from materials_data_lab.phase6a import get_n_splits
from materials_data_lab.steel_classes import classify_steel

def run_v2d(csv_path: Path, outdir: Path):
    raw_df, _ = load_clean(csv_path)
    df = prepare_features(raw_df, keep_initial=False).copy()
    
    # 1. Apply Classification
    df["steel_class"] = df["steel_type"].apply(classify_steel)
    
    # Class distribution (overall and by source)
    dist_overall = df["steel_class"].value_counts().to_dict()
    dist_source = df.groupby(["steel_class", "source"]).size().unstack(fill_value=0).to_dict(orient="index")
    
    unknown_count = dist_overall.get("UNKNOWN_CLASS", 0)
    print(f"Total UNKNOWN_CLASS: {unknown_count}")
    
    X = df[RF_FEATURES].values
    y = df["final_hrc"].values
    groups = df["steel_type"].values
    
    # Models to test
    # XGB_tuned from V2-C2
    xgb_tuned = xgb.XGBRegressor(
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
    
    # RF_tuned from V2-C1
    rf_tuned = RandomForestRegressor(
        n_estimators=800,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.5,
        max_depth=30,
        random_state=42
    )
    
    n_splits = get_n_splits(groups, desired=5)
    gkf = GroupKFold(n_splits=n_splits)
    
    def get_oof_predictions(model, X, y, groups):
        oof = np.zeros(len(y))
        for train_idx, test_idx in gkf.split(X, y, groups):
            m = sklearn.base.clone(model)
            m.fit(X[train_idx], y[train_idx])
            oof[test_idx] = m.predict(X[test_idx])
        return oof
        
    print("Generating OOF for XGB_tuned...")
    oof_xgb = get_oof_predictions(xgb_tuned, X, y, groups)
    print("Generating OOF for RF_tuned...")
    oof_rf = get_oof_predictions(rf_tuned, X, y, groups)
    
    df["oof_xgb"] = oof_xgb
    df["oof_rf"] = oof_rf
    
    df["err_xgb"] = np.abs(df["final_hrc"] - df["oof_xgb"])
    df["err_rf"] = np.abs(df["final_hrc"] - df["oof_rf"])
    
    class_mae = df.groupby("steel_class")[["err_xgb", "err_rf"]].mean().to_dict(orient="index")
    
    # A/B Test: XGB_tuned vs XGB_tuned + one-hot steel_class
    print("Running A/B experiment for one-hot encoding...")
    X_onehot = pd.get_dummies(df["steel_class"], prefix="class").values
    X_ab = np.hstack([X, X_onehot])
    
    oof_xgb_ab = get_oof_predictions(xgb_tuned, X_ab, y, groups)
    mae_ab = mean_absolute_error(y, oof_xgb_ab)
    mae_base = mean_absolute_error(y, oof_xgb)
    
    gain = mae_base - mae_ab
    if gain < 0.1:
        ab_result = f"Adding one-hot class improved MAE by {gain:.4f} HRC. This is <0.1 HRC, so the gain is not material."
    else:
        ab_result = f"Adding one-hot class improved MAE by {gain:.4f} HRC. This is material."
        
    inv_dir = csv_path.parent.parent / "inventory"
    inv_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare JSON output
    out_data = {
        "dist_overall": dist_overall,
        "dist_source": dist_source,
        "class_mae": class_mae,
        "mae_base": float(mae_base),
        "mae_ab": float(mae_ab),
        "ab_gain": float(gain),
        "ab_result": ab_result
    }
    with open(inv_dir / "phase_v2d_class_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
        
    outdir.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# Materials Data Lab — Phase V2-D: Steel Class Analysis",
        "",
        "## Sınıf Dağılımı ve Köken",
        "| Steel Class | Total Rows | Source Breakdown |",
        "| :--- | :--- | :--- |"
    ]
    
    for cls, count in dist_overall.items():
        src_counts = dist_source.get(cls, {})
        src_str = ", ".join([f"{k}: {v}" for k, v in src_counts.items() if v > 0])
        lines.append(f"| **{cls}** | {count} | {src_str} |")
        
    lines.extend([
        "",
        "## MAE Kıyası: XGB_tuned vs RF_tuned",
        "| Steel Class | Rows | XGB_tuned MAE | RF_tuned MAE | Winner |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])
    
    for cls in sorted(class_mae.keys()):
        count = dist_overall[cls]
        errs = class_mae[cls]
        xg_err = errs["err_xgb"]
        rf_err = errs["err_rf"]
        winner = "XGB" if xg_err < rf_err else "RF"
        lines.append(f"| **{cls}** | {count} | {xg_err:.3f} | {rf_err:.3f} | {winner} |")
        
    lines.extend([
        "",
        "## A/B Mini Deney (One-hot `steel_class`)",
        f"- XGBoost Base S2 MAE: {mae_base:.3f}",
        f"- XGBoost One-hot S2 MAE: {mae_ab:.3f}",
        f"- **Sonuç**: {ab_result}"
    ])
    
    with open(outdir / "phase_v2d_class_analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print("V2-D class analysis generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="reports")
    args = parser.parse_args()
    run_v2d(Path(args.input), Path(args.outdir))
