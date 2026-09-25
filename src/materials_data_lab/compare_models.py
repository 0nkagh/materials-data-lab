"""Phase V2-C2: XGBoost vs Random Forest & SHAP Interpretation."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import sklearn

# Must be before pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from materials_data_lab.clean_loader import load_clean
from materials_data_lab.modeling import RF_FEATURES, prepare_features
from materials_data_lab.phase6a import get_n_splits

XGB_SEARCH_SPACE = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1, 2, 5]
}

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

def perform_xgb_tuning(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_iter: int = 30) -> Tuple[xgb.XGBRegressor, dict[str, Any]]:
    # Use hist tree method for speed and deterministic behavior when random_state is set
    model = xgb.XGBRegressor(random_state=42, tree_method="hist", n_jobs=1)
    
    n_splits = get_n_splits(groups, desired=5)
    gkf = GroupKFold(n_splits=n_splits)
    
    search = RandomizedSearchCV(
        model,
        param_distributions=XGB_SEARCH_SPACE,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=gkf,
        random_state=42,
        n_jobs=1
    )
    
    t0 = time.time()
    search.fit(X, y, groups=groups)
    t_elapsed = time.time() - t0
    
    search_meta = {
        "best_params": search.best_params_,
        "best_mae": float(-search.best_score_),
        "time_seconds": t_elapsed
    }
    
    return search.best_estimator_, search_meta


def evaluate_protocols(model, X, y, groups):
    # S1 RandomSplit
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m_s1 = sklearn.base.clone(model)
    m_s1.fit(X_tr, y_tr)
    s1_res = evaluate_model(y_te, m_s1.predict(X_te))
    
    # S2 GroupKFold
    n_splits = get_n_splits(groups, desired=5)
    gkf = GroupKFold(n_splits=n_splits)
    
    s2_metrics = {"MAE": [], "RMSE": [], "R2": []}
    for train_idx, test_idx in gkf.split(X, y, groups):
        m_s2 = sklearn.base.clone(model)
        m_s2.fit(X[train_idx], y[train_idx])
        res = evaluate_model(y[test_idx], m_s2.predict(X[test_idx]))
        for k in res:
            s2_metrics[k].append(res[k])
            
    s2_res = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in s2_metrics.items()}
    
    return s1_res, s2_res

def generate_shap_plots(model, X_df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    
    # TreeExplainer is fast for both RF and XGB
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_df)
    
    # Summary Bar
    plt.figure()
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_bar.png", dpi=150)
    plt.close()
    
    # Beeswarm
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(outdir / "shap_beeswarm.png", dpi=150)
    plt.close()
    
    # Dependence plot for temper_temp_c
    plt.figure()
    shap.dependence_plot("temper_temp_c", shap_values.values, X_df, show=False)
    plt.tight_layout()
    plt.savefig(outdir / "shap_dependence_temp.png", dpi=150)
    plt.close()

def run_v2c2(csv_path: Path, outdir: Path, n_iter: int = 30):
    raw_df, _ = load_clean(csv_path)
    df = prepare_features(raw_df, keep_initial=False).copy()
    
    X = df[RF_FEATURES].values
    X_df = df[RF_FEATURES]
    y = df["final_hrc"].values
    groups = df["steel_type"].values
    
    # 1. Train 4 models
    rf_def = RandomForestRegressor(random_state=42)
    # The known best params from V2-C1
    rf_tuned = RandomForestRegressor(n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features=0.5, max_depth=30, random_state=42)
    xgb_def = xgb.XGBRegressor(random_state=42, tree_method="hist", n_jobs=1)
    
    print("Tuning XGBoost...")
    xgb_tuned, xgb_search_meta = perform_xgb_tuning(X, y, groups, n_iter=n_iter)
    
    models = {
        "RF_default": rf_def,
        "RF_tuned": rf_tuned,
        "XGB_default": xgb_def,
        "XGB_tuned": xgb_tuned
    }
    
    results = {}
    for name, m in models.items():
        print(f"Evaluating {name}...")
        s1, s2 = evaluate_protocols(m, X, y, groups)
        results[name] = {"S1": s1, "S2": s2}
        
    rf_tuned_mae = results["RF_tuned"]["S2"]["MAE"]["mean"]
    xgb_tuned_mae = results["XGB_tuned"]["S2"]["MAE"]["mean"]
    
    threshold = 0.15
    gain = rf_tuned_mae - xgb_tuned_mae
    
    if gain > threshold:
        champion_name = "XGB_tuned"
        champion_model = xgb_tuned
        d12_result = f"XGBoost outperforms RF by {gain:.4f} HRC MAE, which exceeds the 0.15 threshold. XGBoost is the new champion."
    else:
        champion_name = "RF_tuned"
        champion_model = rf_tuned
        if gain > 0:
            d12_result = f"XGBoost is slightly better by {gain:.4f} HRC MAE, but does not exceed the 0.15 threshold. RF keeps crown."
        else:
            d12_result = f"XGBoost is worse by {-gain:.4f} HRC MAE. RF keeps crown."
            
    # Retrain champion on all data for SHAP
    print(f"Champion is {champion_name}. Generating SHAP...")
    champion_model.fit(X, y)
    fig_dir = outdir / "figures"
    generate_shap_plots(champion_model, X_df, fig_dir)
    
    # JSON output
    inv_dir = csv_path.parent.parent / "inventory"
    inv_dir.mkdir(parents=True, exist_ok=True)
    json_data = {
        "results": results,
        "xgb_search": xgb_search_meta,
        "d12_champion": champion_name,
        "d12_gain": float(gain),
        "d12_result": d12_result
    }
    
    with open(inv_dir / "phase_v2c2_compare.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
        
    # Markdown report
    lines = [
        "# Materials Data Lab — Phase V2-C2: XGBoost Challenger & SHAP",
        "",
        "## Yöntem (D-11 & D-12)",
        "- **D-11**: XGBoost için GroupKFold tabanlı adil yarış. Arama: RandomizedSearchCV, n_iter=30.",
        "- **D-12**: Şampiyon model değişimi için S2 MAE'de >0.15 HRC iyileşme aranır. Yoksa RF tacını korur.",
        "",
        "### XGBoost Tuning Sonucu",
        f"- Süre: {xgb_search_meta['time_seconds']:.1f} saniye",
        f"- En İyi Parametreler: `{xgb_search_meta['best_params']}`",
        "",
        "## Kıyas Matrisi",
        "| Model | S1 (RandomSplit) MAE | S1 RMSE | S1 R² | S2 (GroupSplit) MAE | S2 RMSE | S2 R² |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for name in ["RF_default", "RF_tuned", "XGB_default", "XGB_tuned"]:
        s1 = results[name]["S1"]
        s2 = results[name]["S2"]
        s1_str = f"{s1['MAE']:.4f} | {s1['RMSE']:.4f} | {s1['R2']:.4f}"
        s2_str = f"{s2['MAE']['mean']:.4f} ± {s2['MAE']['std']:.4f} | {s2['RMSE']['mean']:.4f} ± {s2['RMSE']['std']:.4f} | {s2['R2']['mean']:.4f} ± {s2['R2']['std']:.4f}"
        lines.append(f"| **{name}** | {s1_str} | {s2_str} |")
        
    lines.extend([
        "",
        "## D-12 Kararı (Şampiyon)",
        f"> **{d12_result}**",
        "",
        "*(Not: Demo V2-F aşamasına kadar default RF modeliyle çalışmaya devam edecektir.)*",
        "",
        "## SHAP Yorumlaması (Şampiyon Üzerinde)",
        "Aşağıdaki figürler, şampiyon modelin kararlarını SHAP aracılığıyla görselleştirir.",
        "",
        "### 1. Summary Bar",
        "![Summary Bar](figures/shap_summary_bar.png)",
        "Genel özellik önemini (ortalama mutlak SHAP) verir. Permutation/MDI ile korele olup olmadığına dikkat edilir.",
        "",
        "### 2. Beeswarm",
        "![Beeswarm](figures/shap_beeswarm.png)",
        "Hangi özelliğin yüksek veya düşük değerlerinin tahmini ne yönde değiştirdiğini (korelasyon) gösterir.",
        "",
        "### 3. Dependence Plot (Temperature)",
        "![Dependence Temp](figures/shap_dependence_temp.png)",
        "Sıcaklığın (temper_temp_c) tahmine olan lineer olmayan etkisini gösterir.",
        "",
        "**[REPORT_ONLY] Uyarı**: Ağaç tabanlı modellerde SHAP değerleri korele özellikler arasında dağılabilir, bu da nedensel bir yorumlamayı yanıltıcı kılabilir."
    ])
    
    with open(outdir / "phase_v2c2_compare.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"Phase V2-C2 complete. {d12_result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="reports")
    args = parser.parse_args()
    run_v2c2(Path(args.input), Path(args.outdir))
