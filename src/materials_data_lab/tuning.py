"""Hyperparameter Tuning for RF (Phase V2-C1)."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

# Must be before pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from materials_data_lab.clean_loader import load_clean
from materials_data_lab.modeling import RF_FEATURES, prepare_features
from materials_data_lab.phase6a import get_n_splits

# D-09 Search Space
SEARCH_SPACE = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 5],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", 0.5, None]
}

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

def perform_tuning(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_iter: int = 40) -> tuple[RandomForestRegressor, dict[str, Any], float]:
    """Run RandomizedSearchCV on Random Forest with GroupKFold."""
    rf = RandomForestRegressor(random_state=42)
    
    n_splits = get_n_splits(groups, desired=5)
    gkf = GroupKFold(n_splits=n_splits)
    
    search = RandomizedSearchCV(
        rf,
        param_distributions=SEARCH_SPACE,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=gkf,
        random_state=42,
        n_jobs=1
    )
    
    t0 = time.time()
    search.fit(X, y, groups=groups)
    t_elapsed = time.time() - t0
    
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values("rank_test_score").head(5)
    top_5 = []
    for _, row in results_df.iterrows():
        top_5.append({
            "params": row["params"],
            "mean_test_mae": float(-row["mean_test_score"]),
            "std_test_mae": float(row["std_test_score"])
        })
        
    search_meta = {
        "best_params": search.best_params_,
        "best_mae": float(-search.best_score_),
        "time_seconds": t_elapsed,
        "top_5": top_5
    }
    
    return search.best_estimator_, search_meta, t_elapsed

def generate_learning_curve(model, X, y, groups, out_path: Path):
    """Generate and save learning curve plot."""
    n_splits = get_n_splits(groups, desired=5)
    gkf = GroupKFold(n_splits=n_splits)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, groups=groups, cv=gkf, scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=1, random_state=42
    )
    
    train_mae = -train_scores.mean(axis=1)
    test_mae = -test_scores.mean(axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mae, 'o-', color="r", label="Training MAE")
    plt.plot(train_sizes, test_mae, 'o-', color="g", label="Validation MAE (GroupCV)")
    plt.title("Learning Curve (Tuned Random Forest)")
    plt.xlabel("Training Examples")
    plt.ylabel("Mean Absolute Error (HRC)")
    plt.legend(loc="best")
    plt.grid(True)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return {
        "train_sizes": train_sizes.tolist(),
        "train_mae": train_mae.tolist(),
        "test_mae": test_mae.tolist()
    }

def run_v2c1(csv_path: Path, outdir: Path, n_iter: int = 40):
    raw_df, _ = load_clean(csv_path)
    df = prepare_features(raw_df, keep_initial=False).copy()
    
    X = df[RF_FEATURES].values
    y = df["final_hrc"].values
    groups = df["steel_type"].values
    
    # 1. Tuning
    print(f"Running RandomizedSearchCV (n_iter={n_iter})...")
    tuned_rf, search_meta, _ = perform_tuning(X, y, groups, n_iter=n_iter)
    
    # 2. Evaluation Strategy S1: Random Split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    default_rf = RandomForestRegressor(random_state=42)
    default_rf.fit(X_tr, y_tr)
    s1_default = evaluate_model(y_te, default_rf.predict(X_te))
    
    tuned_rf_s1 = sklearn.base.clone(tuned_rf)
    tuned_rf_s1.fit(X_tr, y_tr)
    s1_tuned = evaluate_model(y_te, tuned_rf_s1.predict(X_te))
    
    # 3. Evaluation Strategy S2: GroupKFold
    n_splits = get_n_splits(groups, desired=5)
    gkf = GroupKFold(n_splits=n_splits)
    
    s2_default_metrics = {"MAE": [], "RMSE": [], "R2": []}
    s2_tuned_metrics = {"MAE": [], "RMSE": [], "R2": []}
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        # Default
        m_def = RandomForestRegressor(random_state=42)
        m_def.fit(X[train_idx], y[train_idx])
        res_def = evaluate_model(y[test_idx], m_def.predict(X[test_idx]))
        for k in res_def:
            s2_default_metrics[k].append(res_def[k])
            
        # Tuned
        m_tuned = sklearn.base.clone(tuned_rf)
        m_tuned.fit(X[train_idx], y[train_idx])
        res_tun = evaluate_model(y[test_idx], m_tuned.predict(X[test_idx]))
        for k in res_tun:
            s2_tuned_metrics[k].append(res_tun[k])
            
    def agg_metrics(metrics_dict):
        return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in metrics_dict.items()}
        
    s2_default = agg_metrics(s2_default_metrics)
    s2_tuned = agg_metrics(s2_tuned_metrics)
    
    # 4. Learning Curve
    lc_path = outdir / "figures" / "learning_curve.png"
    print("Generating learning curve...")
    lc_meta = generate_learning_curve(tuned_rf, X, y, groups, lc_path)
    
    # Calculate gain
    gain = s2_default["MAE"]["mean"] - s2_tuned["MAE"]["mean"]
    
    report = {
        "search_meta": search_meta,
        "s1": {
            "default": s1_default,
            "tuned": s1_tuned
        },
        "s2": {
            "default": s2_default,
            "tuned": s2_tuned
        },
        "gain": gain,
        "learning_curve": lc_meta
    }
    
    # Output JSON
    json_path = csv_path.parent.parent / "inventory" / "phase_v2c1_tuning.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    # Output Markdown
    md_lines = [
        "# Materials Data Lab — Phase V2-C1: Hyperparameter Tuning",
        "",
        "## Yöntem Özeti",
        "- **D-09**: RandomizedSearchCV, n_iter=40, cv=GroupKFold(5, steel_type). Arama uzayı sınırlandırılmıştır.",
        "- **D-10**: Şampiyon model, S2 GroupKFold MAE metriğine göre seçilmiştir.",
        "",
        "## Search Sonuçları",
        f"- **Süre**: {search_meta['time_seconds']:.1f} saniye",
        f"- **En İyi Parametreler**: `{search_meta['best_params']}`",
        f"- **CV MAE (best)**: {search_meta['best_mae']:.4f}",
        "",
        "### Top 5 Kombinasyon",
        "| Rank | Parametreler | Test MAE (mean ± std) |",
        "| :--- | :--- | :--- |"
    ]
    
    for i, res in enumerate(search_meta["top_5"], 1):
        md_lines.append(f"| {i} | `{res['params']}` | {res['mean_test_mae']:.4f} ± {res['std_test_mae']:.4f} |")
        
    md_lines.extend([
        "",
        "## Performans Kıyası (Default vs Tuned)",
        "| Protokol | Model | MAE | RMSE | R² |",
        "| :--- | :--- | :--- | :--- |",
        f"| S1 (RandomSplit) | Default RF | {s1_default['MAE']:.4f} | {s1_default['RMSE']:.4f} | {s1_default['R2']:.4f} |",
        f"| S1 (RandomSplit) | Tuned RF   | {s1_tuned['MAE']:.4f} | {s1_tuned['RMSE']:.4f} | {s1_tuned['R2']:.4f} |",
        "| | | | | |",
        f"| S2 (GroupSplit)  | Default RF | {s2_default['MAE']['mean']:.4f} ± {s2_default['MAE']['std']:.4f} | {s2_default['RMSE']['mean']:.4f} ± {s2_default['RMSE']['std']:.4f} | {s2_default['R2']['mean']:.4f} ± {s2_default['R2']['std']:.4f} |",
        f"| S2 (GroupSplit)  | Tuned RF   | {s2_tuned['MAE']['mean']:.4f} ± {s2_tuned['MAE']['std']:.4f} | {s2_tuned['RMSE']['mean']:.4f} ± {s2_tuned['RMSE']['std']:.4f} | {s2_tuned['R2']['mean']:.4f} ± {s2_tuned['R2']['std']:.4f} |",
        "",
        "## Öğrenme Eğrisi Gözlemi (Learning Curve)",
        "![Learning Curve](figures/learning_curve.png)",
        "Eğri, train ve validation MAE arasındaki kapanmayı ve verinin artışıyla kazanımın doygunluğa ulaşıp ulaşmadığını gösterir."
    ])
    
    md_path = outdir / "phase_v2c1_tuning.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"Phase V2-C1 complete. Gain: {gain:.4f} HRC")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="reports")
    args = parser.parse_args()
    
    run_v2c1(Path(args.input), Path(args.outdir))
