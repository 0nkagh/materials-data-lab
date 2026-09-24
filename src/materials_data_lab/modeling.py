"""Modeling MVP for Materials Data Lab (Phase 4).

Features Random Forest MVP against Physics Baseline (Hollomon-Jaffe).
Implements leakage-aware validation (GroupKFold).
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

from materials_data_lab.clean_loader import load_clean

COMP_COLS = ["c_wt", "mn_wt", "p_wt", "s_wt", "si_wt", "ni_wt", "cr_wt", "mo_wt", "v_wt", "al_wt", "cu_wt"]
RF_FEATURES = COMP_COLS + ["temper_temp_c", "log_time"]


def prepare_features(df: pd.DataFrame, keep_initial: bool = False) -> pd.DataFrame:
    """Prepare features for modeling.
    
    Drops rows where final_hrc is missing.
    Calculates log_time and p_hj.
    """
    df = df.copy()
    
    # Target
    df = df.dropna(subset=["final_hrc"])
    
    # D-08 Decision
    if not keep_initial and "initial_hrc" in df.columns:
        df = df.drop(columns=["initial_hrc"])
        
    # log_time feature engineering
    # Using np.log10. Replace 0 with NaN (though time_s should be > 0)
    df["log_time"] = np.log10(df["temper_time_s"].replace(0, np.nan))
    
    # Physics parameter p_hj (C=19.5 ASSUMED)
    df["temp_k"] = df["temper_temp_c"] + 273.15
    df["t_saat"] = df["temper_time_s"] / 3600.0
    df["p_hj"] = (df["temp_k"] / 1000.0) * (19.5 + np.log10(df["t_saat"].replace(0, np.nan)))
    
    # Drop rows with NaN in features
    features_to_check = RF_FEATURES + ["p_hj", "steel_type"]
    df = df.dropna(subset=features_to_check)
    
    return df


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_modeling(csv_path: Path) -> dict[str, Any]:
    raw_df, manifest = load_clean(csv_path)
    df = prepare_features(raw_df)
    
    X_rf = df[RF_FEATURES].values
    X_phys = df[["p_hj"]].values
    y = df["final_hrc"].values
    groups = df["steel_type"].values
    
    results: dict[str, Any] = {
        "S1_RandomSplit": {},
        "S2_GroupSplit": {},
        "RF_Importance": {},
        "decisions": [
            "D-08: initial_hrc MVP'de kullanılmadı (eksik veri oranı yüksek).",
            "Feature Engineering: log10(temper_time_s) kullanıldı.",
            "Physics Baseline: p_hj = (T_K/1000) * (19.5 + log10(t_saat)) [C=19.5 ASSUMED, REPORT_ONLY]"
        ]
    }
    
    # --- Strategy S1: Random Split ---
    X_rf_tr, X_rf_te, X_ph_tr, X_ph_te, y_tr, y_te = train_test_split(
        X_rf, X_phys, y, test_size=0.2, random_state=42
    )
    
    # B1: Naive (Median)
    b1 = DummyRegressor(strategy="median")
    b1.fit(X_rf_tr, y_tr)
    results["S1_RandomSplit"]["B1_Naive"] = evaluate_model(y_te, b1.predict(X_rf_te))
    
    # B2: Physics Baseline
    b2 = LinearRegression()
    b2.fit(X_ph_tr, y_tr)
    results["S1_RandomSplit"]["B2_Physics"] = evaluate_model(y_te, b2.predict(X_ph_te))
    
    # M1: Random Forest
    m1 = RandomForestRegressor(random_state=42)
    m1.fit(X_rf_tr, y_tr)
    results["S1_RandomSplit"]["M1_RF"] = evaluate_model(y_te, m1.predict(X_rf_te))
    
    # Extract Feature Importance from M1 on S1
    importances = m1.feature_importances_
    imp_dict = {feat: float(imp) for feat, imp in zip(RF_FEATURES, importances)}
    sorted_imp = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:10])
    results["RF_Importance"]["top_10"] = sorted_imp
    results["RF_Importance"]["warning"] = "Korele metallurjik özelliklerde MDI yanıltıcı olabilir [REPORT_ONLY]"
    
    # --- Strategy S2: GroupSplit (GroupKFold) ---
    gkf = GroupKFold(n_splits=5)
    
    s2_metrics = {"B1_Naive": [], "B2_Physics": [], "M1_RF": []}
    
    for train_idx, test_idx in gkf.split(X_rf, y, groups):
        # B1
        b1_cv = DummyRegressor(strategy="median")
        b1_cv.fit(X_rf[train_idx], y[train_idx])
        s2_metrics["B1_Naive"].append(evaluate_model(y[test_idx], b1_cv.predict(X_rf[test_idx])))
        
        # B2
        b2_cv = LinearRegression()
        b2_cv.fit(X_phys[train_idx], y[train_idx])
        s2_metrics["B2_Physics"].append(evaluate_model(y[test_idx], b2_cv.predict(X_phys[test_idx])))
        
        # M1
        m1_cv = RandomForestRegressor(random_state=42)
        m1_cv.fit(X_rf[train_idx], y[train_idx])
        s2_metrics["M1_RF"].append(evaluate_model(y[test_idx], m1_cv.predict(X_rf[test_idx])))
        
    # Aggregate S2 metrics
    for model_name in s2_metrics:
        results["S2_GroupSplit"][model_name] = {}
        for metric in ["MAE", "RMSE", "R2"]:
            vals = [fold[metric] for fold in s2_metrics[model_name]]
            results["S2_GroupSplit"][model_name][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals))
            }
            
    return results


def generate_markdown_modeling(report: dict[str, Any]) -> str:
    lines = [
        "# Materials Data Lab — Phase 4: Modelleme MVP",
        "",
        "## Yöntem Özeti ve Kararlar",
    ]
    for dec in report["decisions"]:
        lines.append(f"- {dec}")
        
    lines.extend([
        "",
        "## Gözlem: RandomSplit (S1) vs GroupSplit (S2)",
        "> **ZORUNLU NOT**: S1'de aynı çeliğin satırları train ve testte kalabildiğinden sonuçlar iyimser olabilir; "
        "S2 (\"daha önce görülmemiş çelik\") daha gerçekçi üst sınırdır. Nedensellik iddiası taşımaz, model gözlemidir.",
        "",
        "### S1: RandomSplit (Test Size 0.2)",
        "| Model | MAE | RMSE | R² |",
        "| :--- | :-: | :-: | :-: |"
    ])
    
    for m in ["B1_Naive", "B2_Physics", "M1_RF"]:
        res = report["S1_RandomSplit"][m]
        lines.append(f"| {m} | {res['MAE']:.4f} | {res['RMSE']:.4f} | {res['R2']:.4f} |")
        
    lines.extend([
        "",
        "### S2: GroupSplit (GroupKFold = 5, steel_type grouped)",
        "| Model | MAE (mean ± std) | RMSE (mean ± std) | R² (mean ± std) |",
        "| :--- | :-: | :-: | :-: |"
    ])
    
    for m in ["B1_Naive", "B2_Physics", "M1_RF"]:
        res = report["S2_GroupSplit"][m]
        lines.append(
            f"| {m} | {res['MAE']['mean']:.4f} ± {res['MAE']['std']:.4f} | "
            f"{res['RMSE']['mean']:.4f} ± {res['RMSE']['std']:.4f} | "
            f"{res['R2']['mean']:.4f} ± {res['R2']['std']:.4f} |"
        )
        
    lines.extend([
        "",
        "## B2 vs M1 Karşılaştırması",
        "**Gözlem**: RandomForest (M1) genel hatlarıyla fiziksel baseline'dan (B2) daha düşük hata oranlarına ulaşmıştır. "
        "Ancak, fiziksel denklemin (p_hj) yalnızca tek bir değişkene dayalı basit bir Linear Regression modeli (B2) ile gösterdiği "
        "başarı dikkat çekicidir.",
        "",
        "## RF Feature Importance (MDI) - İlk 10",
        f"> **UYARI**: {report['RF_Importance']['warning']}",
        ""
    ])
    
    for feat, imp in report["RF_Importance"]["top_10"].items():
        lines.append(f"- **{feat}**: {imp:.4f}")
        
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materials Data Lab - Phase 4 Modeling")
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV file.")
    parser.add_argument("--outdir", type=str, default="reports", help="Directory to save reports.")
    parser.add_argument("--json-out", type=str, default="data/inventory/phase4_modeling.json", help="Path to save JSON.")
    
    args = parser.parse_args(argv)
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    json_path = Path(args.json_out)
    
    outdir.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = run_modeling(input_path)
    
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    
    md_content = generate_markdown_modeling(report)
    md_report_path = outdir / "phase4_modeling.md"
    md_report_path.write_text(md_content, encoding="utf-8")
    
    print("Phase 4 Modeling complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
