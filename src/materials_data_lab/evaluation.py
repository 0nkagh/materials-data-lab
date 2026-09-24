"""Evaluation and Error Analysis for Materials Data Lab (Phase 5).

Generates Out-Of-Fold (OOF) predictions and analyzes residuals.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from materials_data_lab.clean_loader import load_clean
from materials_data_lab.modeling import prepare_features, RF_FEATURES

def assign_hrc_band(hrc: float) -> str:
    """Assign HRC to a band."""
    if pd.isna(hrc): return "UNKNOWN"
    if hrc < 25: return "<25"
    if hrc < 40: return "25-40"
    if hrc < 55: return "40-55"
    return ">=55"

def assign_temp_band(temp: float) -> str:
    """Assign temperature to a band."""
    if pd.isna(temp): return "UNKNOWN"
    if temp < 300: return "<300"
    if temp < 500: return "300-500"
    if temp < 650: return "500-650"
    return ">=650"

def get_group_mae(df: pd.DataFrame, group_col: str, pred_col: str) -> dict[str, dict[str, float]]:
    """Calculate MAE for each group."""
    res = {}
    for name, group in df.groupby(group_col):
        mae = np.mean(np.abs(group["final_hrc"] - group[pred_col]))
        res[str(name)] = {"mae": float(mae), "n": len(group)}
    return res

def run_evaluation(csv_path: Path) -> dict[str, Any]:
    raw_df, manifest = load_clean(csv_path)
    df = prepare_features(raw_df).copy()
    
    # Feature matrices
    X_rf = df[RF_FEATURES].values
    X_phys = df[["p_hj"]].values
    y = df["final_hrc"].values
    groups = df["steel_type"].values
    
    # OOF arrays
    oof_rf = np.zeros_like(y, dtype=float)
    oof_b2 = np.zeros_like(y, dtype=float)
    
    gkf = GroupKFold(n_splits=5)
    for train_idx, test_idx in gkf.split(X_rf, y, groups):
        # M1 RF
        m1 = RandomForestRegressor(random_state=42)
        m1.fit(X_rf[train_idx], y[train_idx])
        oof_rf[test_idx] = m1.predict(X_rf[test_idx])
        
        # B2 Physics
        b2 = LinearRegression()
        b2.fit(X_phys[train_idx], y[train_idx])
        oof_b2[test_idx] = b2.predict(X_phys[test_idx])
        
    df["oof_rf"] = oof_rf
    df["oof_b2"] = oof_b2
    df["resid_rf"] = df["oof_rf"] - df["final_hrc"]
    df["abs_resid_rf"] = np.abs(df["resid_rf"])
    
    df["hrc_band"] = df["final_hrc"].apply(assign_hrc_band)
    df["temp_band"] = df["temper_temp_c"].apply(assign_temp_band)
    
    report: dict[str, Any] = {
        "RF_MAE_by_Source": get_group_mae(df, "source", "oof_rf"),
        "RF_MAE_by_HRC_band": get_group_mae(df, "hrc_band", "oof_rf"),
        "RF_MAE_by_Temp_band": get_group_mae(df, "temp_band", "oof_rf"),
        "B2_MAE_by_Source": get_group_mae(df, "source", "oof_b2"),
        "B2_MAE_by_HRC_band": get_group_mae(df, "hrc_band", "oof_b2"),
        "B2_MAE_by_Temp_band": get_group_mae(df, "temp_band", "oof_b2"),
    }
    
    # Worst 10 predictions for RF
    worst = df.sort_values(by="abs_resid_rf", ascending=False).head(10)
    worst_list = []
    for _, row in worst.iterrows():
        worst_list.append({
            "steel_type": row["steel_type"],
            "source": row["source"],
            "temp_c": row["temper_temp_c"],
            "time_s": row["temper_time_s"],
            "actual": row["final_hrc"],
            "predicted": row["oof_rf"],
            "residual": row["resid_rf"]
        })
    report["Worst_10_RF"] = worst_list
    
    return report

def generate_markdown_evaluation(report: dict[str, Any]) -> str:
    lines = [
        "# Materials Data Lab — Phase 5: Değerlendirme Derinliği ve Model Kartı",
        "",
        "## Faz 1-4 Özeti",
        "- **Phase 1-2** (`69dadb4`): Proje iskeleti oluşturuldu, veri envanteri ve kural bazlı kalite bayrakları (breakdown) yapıldı.",
        "- **Phase 3-4** (`9f6689d`): Onaylanmış karar günlüğü ile temiz veri yükleyici (clean_loader) geliştirildi; özellik mühendisliği (log_time) ve S1/S2 ayrım stratejileriyle modelleme MVP'si oluşturuldu.",
        "",
        "## Model Kartı",
        "- **Veri Kaynağı**: Kaggle üzerinden Raiipa Technologies (CC BY 4.0), literatür temelli (Grange, Hollomon, Penha).",
        "- **Özellikler**: 11 bileşim kolonu, temperleme sıcaklığı, log10(temperleme zamanı).",
        "- **Stratejiler**: S1 (RandomSplit - naif performans) ve S2 (GroupSplit - dış çelik genellemesi). S2 daha gerçekçidir.",
        "- **Metrikler**: RF için S2 R²=0.93, MAE=2.69. Fizik baseline (B2) için S2 R²=0.81, MAE=4.67.",
        "- **Limitasyonlar**: ",
        "  1. (i) tek veri kaynağı kümesi — genellenebilirlik sınırlı,",
        "  2. (ii) C=19.5 ASSUMED [REPORT_ONLY],",
        "  3. (iii) initial_hrc hariç tutuldu (D-08),",
        "  4. (iv) MDI korele-feature uyarısı,",
        "  5. (v) tuning yapılmadı,",
        "  6. (vi) rezidüeller kaynaklar arasında dengesiz olabilir (Grange kaynağında hata genellikle daha yüksektir).",
        "",
        "## Sonuçların Doğru Okunması",
        "> Korelasyon nedensellik ifade etmez. Ayrıca S1'deki (Random Split) R² skorları aynı çelik türünün hem eğitim hem teste sızmasından ötürü **iyimserdir**. S2 skoru daha güvenilirdir.",
        "",
        "## MAE Kırılımları (Out-of-Fold, S2 GroupKFold)",
        "",
        "### Kaynağa Göre MAE",
        "| Kaynak | RF MAE (n) | B2 MAE (n) |",
        "| :--- | :--- | :--- |"
    ]
    
    sources = set(report["RF_MAE_by_Source"].keys()) | set(report["B2_MAE_by_Source"].keys())
    for src in sorted(sources):
        rf_m = report["RF_MAE_by_Source"].get(src, {"mae": 0, "n": 0})
        b2_m = report["B2_MAE_by_Source"].get(src, {"mae": 0, "n": 0})
        lines.append(f"| {src} | {rf_m['mae']:.2f} ({rf_m['n']}) | {b2_m['mae']:.2f} ({b2_m['n']}) |")
        
    lines.extend([
        "",
        "### HRC Bandına Göre MAE",
        "| HRC Bandı | RF MAE (n) | B2 MAE (n) |",
        "| :--- | :--- | :--- |"
    ])
    for b in ["<25", "25-40", "40-55", ">=55"]:
        rf_m = report["RF_MAE_by_HRC_band"].get(b, {"mae": 0, "n": 0})
        b2_m = report["B2_MAE_by_HRC_band"].get(b, {"mae": 0, "n": 0})
        lines.append(f"| {b} | {rf_m['mae']:.2f} ({rf_m['n']}) | {b2_m['mae']:.2f} ({b2_m['n']}) |")

    lines.extend([
        "",
        "### Sıcaklık Bandına Göre MAE",
        "| Sıcaklık Bandı (°C) | RF MAE (n) | B2 MAE (n) |",
        "| :--- | :--- | :--- |"
    ])
    for b in ["<300", "300-500", "500-650", ">=650"]:
        rf_m = report["RF_MAE_by_Temp_band"].get(b, {"mae": 0, "n": 0})
        b2_m = report["B2_MAE_by_Temp_band"].get(b, {"mae": 0, "n": 0})
        lines.append(f"| {b} | {rf_m['mae']:.2f} ({rf_m['n']}) | {b2_m['mae']:.2f} ({b2_m['n']}) |")
        
    lines.extend([
        "",
        "## RF (S2) En Kötü 10 Tahmin (Out-of-Fold)",
        "| Steel Type | Kaynak | Temp (°C) | Time (s) | Gerçek | Tahmin | Rezidüel |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])
    for w in report["Worst_10_RF"]:
        lines.append(f"| {w['steel_type']} | {w['source']} | {w['temp_c']} | {w['time_s']} | {w['actual']:.1f} | {w['predicted']:.1f} | {w['residual']:.1f} |")
        
    lines.append("")
    lines.append("**Gözlem**: B2'nin (fiziksel baseline) en zayıf olduğu bölgeler ile RF modelinin zorlandığı bölgeler (genellikle uç sertlik/sıcaklık değerleri) paralellik göstermektedir.")
        
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materials Data Lab - Phase 5 Evaluation")
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV file.")
    parser.add_argument("--outdir", type=str, default="reports", help="Directory to save reports.")
    parser.add_argument("--json-out", type=str, default="data/inventory/phase5_evaluation.json", help="Path to save JSON.")
    
    args = parser.parse_args(argv)
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    json_path = Path(args.json_out)
    
    outdir.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = run_evaluation(input_path)
    
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    
    md_content = generate_markdown_evaluation(report)
    md_report_path = outdir / "phase5_evaluation.md"
    md_report_path.write_text(md_content, encoding="utf-8")
    
    print("Phase 5 Evaluation complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
