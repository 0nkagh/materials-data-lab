"""Physics-guided study (V3-A) for Materials Data Lab.

Includes:
- D-17: Monotone constraints
- D-18: Metallurgical features (CE, DI, Secondary Hardening, Si-interaction)
- D-19: Split-conformal confidence intervals
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

from materials_data_lab.clean_loader import load_clean
from materials_data_lab.modeling import RF_FEATURES, prepare_features

BEST_PARAMS = dict(
    subsample=0.7,
    reg_lambda=1,
    n_estimators=800,
    max_depth=7,
    learning_rate=0.03,
    colsample_bytree=1.0,
    random_state=42,
    tree_method="hist",
    n_jobs=-1
)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

def run_s2_cv(X, y, groups, model):
    gkf = GroupKFold(n_splits=5)
    metrics = {"MAE": [], "RMSE": [], "R2": []}
    oof_preds = np.zeros_like(y)
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        oof_preds[test_idx] = preds
        fold_metrics = evaluate_model(y[test_idx], preds)
        metrics["MAE"].append(fold_metrics["MAE"])
        metrics["RMSE"].append(fold_metrics["RMSE"])
        metrics["R2"].append(fold_metrics["R2"])
        
    return {
        "MAE": {"mean": float(np.mean(metrics["MAE"])), "std": float(np.std(metrics["MAE"]))},
        "RMSE": {"mean": float(np.mean(metrics["RMSE"])), "std": float(np.std(metrics["RMSE"]))},
        "R2": {"mean": float(np.mean(metrics["R2"])), "std": float(np.std(metrics["R2"]))},
    }, oof_preds

def run_v3a(csv_path: Path, outdir: Path):
    raw_df, _ = load_clean(csv_path)
    df = prepare_features(raw_df)
    
    X_base = df[RF_FEATURES].values
    y = df["final_hrc"].values
    groups = df["steel_type"].values
    
    report = {
        "D-17": {},
        "D-18": {},
        "D-19": {},
        "decisions": [
            "D-15: initial_hrc imputasyonu REDDİ (Verinin %64'ü eksik, sentetik doldurma overconfidence yaratır).",
            "D-16: MLflow/W&B REDDİ (Statik karar defteri / markdown raporlama karmaşıklığı yeterince düşük tutuyor).",
            "D-17: Monotone constraints deneyi.",
            "D-18: Metalurjik özellik deneyleri (CE, DI, V+Mo, Si_interaction).",
            "D-19: Split-conformal out-of-fold %90 güven aralığı."
        ]
    }
    
    # --- D-17: Monotone Constraints ---
    # RF_FEATURES order: c, mn, p, s, si, ni, cr, mo, v, al, cu, temp, log_time
    monotone_constraints = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1)
    
    # Base XGB
    xgb_base = xgb.XGBRegressor(**BEST_PARAMS)
    base_s2, _ = run_s2_cv(X_base, y, groups, xgb_base)
    
    # Monotonic XGB
    xgb_mono = xgb.XGBRegressor(**BEST_PARAMS, monotone_constraints=monotone_constraints)
    mono_s2, _ = run_s2_cv(X_base, y, groups, xgb_mono)
    
    # S1
    X_tr, X_te, y_tr, y_te = train_test_split(X_base, y, test_size=0.2, random_state=42)
    xgb_base.fit(X_tr, y_tr)
    base_s1 = evaluate_model(y_te, xgb_base.predict(X_te))
    
    xgb_mono.fit(X_tr, y_tr)
    mono_s1 = evaluate_model(y_te, xgb_mono.predict(X_te))
    
    # Extrapolation Check (Typical 4140)
    # c=0.4, mn=0.85, p=0.01, s=0.01, si=0.25, ni=0.0, cr=1.0, mo=0.22, v=0, al=0, cu=0, temp, log_time(3600)
    def pred_extrap(model, temp):
        x = np.array([[0.4, 0.85, 0.01, 0.01, 0.25, 0.0, 1.0, 0.22, 0.0, 0.0, 0.0, temp, np.log10(3600)]])
        return float(model.predict(x)[0])
    
    extrap_base_750 = pred_extrap(xgb_base, 750)
    extrap_base_800 = pred_extrap(xgb_base, 800)
    extrap_mono_750 = pred_extrap(xgb_mono, 750)
    extrap_mono_800 = pred_extrap(xgb_mono, 800)
    
    report["D-17"] = {
        "base_s1": base_s1,
        "mono_s1": mono_s1,
        "base_s2": base_s2,
        "mono_s2": mono_s2,
        "extrapolation": {
            "750C": {"base": extrap_base_750, "mono": extrap_mono_750},
            "800C": {"base": extrap_base_800, "mono": extrap_mono_800},
        }
    }
    
    # --- D-18: Metallurgical Features ---
    df_feat = df.copy()
    df_feat["ce"] = df_feat["c_wt"] + df_feat["mn_wt"]/6 + (df_feat["cr_wt"]+df_feat["mo_wt"]+df_feat["v_wt"])/5 + (df_feat["ni_wt"]+df_feat["cu_wt"])/15
    df_feat["log_di"] = np.log10(np.clip(df_feat["c_wt"], 0.001, None)) + np.log10(1 + 3.33*df_feat["mn_wt"]) + np.log10(1 + 0.7*df_feat["si_wt"]) + np.log10(1 + 0.36*df_feat["ni_wt"]) + np.log10(1 + 2.16*df_feat["cr_wt"]) + np.log10(1 + 3.0*df_feat["mo_wt"])
    df_feat["sec_hard"] = df_feat["v_wt"] + 0.5*df_feat["mo_wt"]
    df_feat["si_int"] = df_feat["si_wt"] * np.maximum(0, df_feat["temper_temp_c"] - 300)
    
    features = {
        "carbon_equivalent": "ce",
        "simplified_DI": "log_di",
        "secondary_hardening_proxy": "sec_hard",
        "si_interaction": "si_int"
    }
    
    feat_results = {}
    base_mae = base_s2["MAE"]["mean"]
    
    for name, col in features.items():
        X_new = np.column_stack([X_base, df_feat[col].values])
        res, _ = run_s2_cv(X_new, y, groups, xgb.XGBRegressor(**BEST_PARAMS))
        gain = base_mae - res["MAE"]["mean"]
        material = gain > 0.15
        feat_results[name] = {
            "MAE": res["MAE"]["mean"],
            "gain": gain,
            "material": material
        }
        
    # All combined
    X_all = np.column_stack([X_base, df_feat[[features[k] for k in features]].values])
    res_all, _ = run_s2_cv(X_all, y, groups, xgb.XGBRegressor(**BEST_PARAMS))
    feat_results["combined"] = {
        "MAE": res_all["MAE"]["mean"],
        "gain": base_mae - res_all["MAE"]["mean"],
        "material": (base_mae - res_all["MAE"]["mean"]) > 0.15
    }
    
    report["D-18"] = feat_results
    
    # --- D-19: Split-conformal ---
    # We use base xgb for conformal
    _, oof_preds = run_s2_cv(X_base, y, groups, xgb.XGBRegressor(**BEST_PARAMS))
    residuals = np.abs(y - oof_preds)
    q = np.quantile(residuals, 0.90)
    
    # Coverage analysis
    df_cov = pd.DataFrame({"y": y, "pred": oof_preds, "resid": residuals})
    df_cov["covered"] = df_cov["resid"] <= q
    
    overall_coverage = df_cov["covered"].mean()
    
    def band(val):
        if val < 25: return "<25"
        elif val < 35: return "25-35"
        elif val < 45: return "35-45"
        else: return ">45"
        
    df_cov["band"] = df_cov["y"].apply(band)
    band_coverage = df_cov.groupby("band")["covered"].mean().to_dict()
    
    report["D-19"] = {
        "q_90": float(q),
        "overall_coverage": float(overall_coverage),
        "band_coverage": band_coverage
    }
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bands = ["<25", "25-35", "35-45", ">45"]
    covs = [band_coverage.get(b, 0) * 100 for b in bands]
    
    bars = ax.bar(bands, covs, color="steelblue")
    ax.axhline(90, color="red", linestyle="--", label="Target (90%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Empirical Coverage (%)")
    ax.set_title("Conformal Coverage by Hardness Band (target: 90%)")
    ax.legend()
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height - 5,
                f"{height:.1f}%", ha='center', va='top', color='white', fontweight='bold')
        
    fig.tight_layout()
    fig_path = outdir / "figures" / "conformal_coverage.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    
    # Save JSON and MD
    out_json = outdir.parent / "data" / "inventory" / "phase_v3a_physics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    md_lines = [
        "# Materials Data Lab — V3-A: Physics-guided XGBoost & Conformal Intervals",
        "",
        "## Yöntem Özeti ve Kararlar",
    ]
    for d in report["decisions"]:
        md_lines.append(f"- {d}")
        
    md_lines.extend([
        "",
        "## D-17: Monotone Constraints",
        "Sıcaklık ve zaman özellikleri -1 (negatif) olarak kısıtlandı.",
        "",
        "| Model | S1 MAE | S2 MAE |",
        "| :--- | :-: | :-: |",
        f"| XGB_tuned (Base) | {base_s1['MAE']:.3f} | {base_s2['MAE']['mean']:.3f} |",
        f"| XGB_monotonic | {mono_s1['MAE']:.3f} | {mono_s2['MAE']['mean']:.3f} |",
        "",
        "**Ekstrapolasyon Testi (Tipik 4140 Çeliği - 3600s)**",
        "| Temp | Base Pred (HRC) | Monotonic Pred (HRC) |",
        "| :--- | :-: | :-: |",
        f"| 750°C | {extrap_base_750:.1f} | {extrap_mono_750:.1f} |",
        f"| 800°C | {extrap_base_800:.1f} | {extrap_mono_800:.1f} |",
        "",
        "## D-18: Metalurjik Özelliklerin Etkisi",
        "Tüm aday özellikler XGB_tuned modeline tek tek eklenerek (S2 MAE) test edildi.",
        "",
        "| Özellik | MAE | Gain (HRC) | Karar (>0.15 threshold) |",
        "| :--- | :-: | :-: | :-: |",
        f"| *Base Model* | {base_mae:.3f} | - | - |"
    ])
    
    for feat, info in feat_results.items():
        dec = "ACCEPTED" if info["material"] else "NOT MATERIAL"
        md_lines.append(f"| {feat} | {info['MAE']:.3f} | {info['gain']:+.3f} | {dec} |")
        
    md_lines.extend([
        "",
        "## D-19: Split-Conformal Güven Aralıkları",
        f"Out-of-fold %90 absolut residüel (q): **{q:.3f} HRC**",
        f"Genel ampirik kapsama oranı: **{overall_coverage*100:.1f}%**",
        "",
        "**Band Bazlı Kapsama (Coverage):**"
    ])
    
    for b in bands:
        md_lines.append(f"- {b} HRC: {band_coverage.get(b, 0)*100:.1f}%")
        
    md_path = outdir / "phase_v3a_physics.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    
    print("V3-A completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="reports")
    args = parser.parse_args()
    
    run_v3a(Path(args.input), Path(args.outdir))
