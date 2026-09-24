"""Numerical EDA for Materials Data Lab (Phase 3).

Uses load_clean to generate purely numerical EDA reports in Markdown and JSON.
No plots, no modeling.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Any

from materials_data_lab.clean_loader import load_clean


def summarize_distribution(series: pd.Series) -> dict[str, float]:
    """Calculate summary statistics for a numeric series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0, "min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0, "mean": 0, "std": 0}
    return {
        "n": len(s),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
    }

def run_eda(csv_path: Path) -> dict[str, Any]:
    df, manifest = load_clean(csv_path)
    
    report: dict[str, Any] = {
        "manifest": manifest,
    }
    
    # 1. final_hrc distribution
    hrc_dist = {
        "overall": summarize_distribution(df["final_hrc"]),
        "by_source": {
            src: summarize_distribution(group["final_hrc"]) 
            for src, group in df.groupby("source")
        }
    }
    report["final_hrc_dist"] = hrc_dist
    
    # 2. temper_temp_c and temper_time_s
    temp_dist = {
        "overall": summarize_distribution(df["temper_temp_c"]),
        "by_source": {
            src: summarize_distribution(group["temper_temp_c"]) 
            for src, group in df.groupby("source")
        }
    }
    report["temper_temp_c_dist"] = temp_dist
    
    time_dist = {
        "overall": summarize_distribution(df["temper_time_s"]),
        "by_source": {
            src: summarize_distribution(group["temper_time_s"]) 
            for src, group in df.groupby("source")
        }
    }
    report["temper_time_s_dist"] = time_dist
    
    # 3. 11 composition cols (min, median, max)
    comp_cols = ["c_wt", "mn_wt", "p_wt", "s_wt", "si_wt", "ni_wt", "cr_wt", "mo_wt", "v_wt", "al_wt", "cu_wt"]
    comp_summary = {}
    for c in comp_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) > 0:
            comp_summary[c] = {
                "min": float(s.min()),
                "median": float(s.median()),
                "max": float(s.max())
            }
    report["composition_summary"] = comp_summary
    
    # 4. Spearman correlation: final_hrc vs temp and log(time)
    df["log_time"] = np.log10(df["temper_time_s"].replace(0, np.nan))
    
    def get_spearman(d: pd.DataFrame, col1: str, col2: str) -> float:
        valid = d[[col1, col2]].dropna()
        if len(valid) < 2: return 0.0
        return float(valid[col1].corr(valid[col2], method="spearman"))
        
    spearman_temp = {
        "overall": get_spearman(df, "final_hrc", "temper_temp_c"),
        "by_source": {
            src: get_spearman(group, "final_hrc", "temper_temp_c")
            for src, group in df.groupby("source")
        }
    }
    spearman_time = {
        "overall": get_spearman(df, "final_hrc", "log_time"),
        "by_source": {
            src: get_spearman(group, "final_hrc", "log_time")
            for src, group in df.groupby("source")
        }
    }
    report["spearman_temp"] = spearman_temp
    report["spearman_time"] = spearman_time
    
    # 5. Exploratory Hollomon-Jaffe
    df["temp_k"] = df["temper_temp_c"] + 273.15
    df["t_saat"] = df["temper_time_s"] / 3600.0
    df["p_hj"] = (df["temp_k"] / 1000.0) * (19.5 + np.log10(df["t_saat"].replace(0, np.nan)))
    
    spearman_hj = {
        "overall": get_spearman(df, "final_hrc", "p_hj"),
        "by_source": {
            src: get_spearman(group, "final_hrc", "p_hj")
            for src, group in df.groupby("source")
        }
    }
    report["spearman_hj"] = spearman_hj
    
    # 6. Missingness pattern
    missing_ct = pd.crosstab(df["source"], df["initial_hrc_missing"]).to_dict()
    report["initial_hrc_missing_crosstab"] = missing_ct
    
    # 7. Flag consistency
    # Expected: SUSPECT 303 / WARNING 278 / INFO 604 / E140 123
    # Note that flags column is pipe separated
    all_flags = df["flags"].str.split("|").explode()
    all_flags = all_flags[all_flags != ""] # remove empties
    
    # We map rule to severity as defined in Phase 1
    # Actually, we can just count exact flag strings first
    flag_counts = all_flags.value_counts().to_dict()
    
    # Let's count by expected severities using simple string matching or exact knowledge
    # For Phase 1: 
    # SUSPECT rules: HARDNESS_OUT_OF_RANGE_SUSPECT, E140_EXTRAPOLATION_SUSPECT, P_UPPER_BOUND_SUSPECT, Al_UPPER_BOUND_SUSPECT, TEMP_PHYSICAL_LIMIT_SUSPECT, TIME_NON_POSITIVE_SUSPECT, ELEMENT_PERCENT_BOUNDS_SUSPECT
    suspect_flags = [k for k in flag_counts.keys() if "SUSPECT" in k]
    warning_flags = [k for k in flag_counts.keys() if "WARNING" in k]
    info_flags = [k for k in flag_counts.keys() if "INFO" in k]
    
    suspect_total = sum(flag_counts[k] for k in suspect_flags)
    warning_total = sum(flag_counts[k] for k in warning_flags)
    info_total = sum(flag_counts[k] for k in info_flags)
    e140_total = flag_counts.get("E140_EXTRAPOLATION_SUSPECT", 0)
    
    expected = {
        "SUSPECT": 303,
        "WARNING": 278,
        "INFO": 604,
        "E140": 123
    }
    actual = {
        "SUSPECT": suspect_total,
        "WARNING": warning_total,
        "INFO": info_total,
        "E140": e140_total
    }
    
    discrepancies = {}
    for k in expected:
        if expected[k] != actual[k]:
            discrepancies[k] = {"expected": expected[k], "actual": actual[k]}
            
    report["flag_consistency"] = {
        "expected": expected,
        "actual": actual,
        "discrepancies": discrepancies,
        "raw_counts": flag_counts
    }
    
    return report

def generate_markdown_eda(report: dict[str, Any]) -> str:
    lines = [
        "# Materials Data Lab — Phase 3: Sayısal Keşifsel Veri Analizi (EDA)",
        "",
        "> **NOT**: Bu rapordaki korelasyon bulguları nedensellik belirtmez. Yalnızca istatistiksel gözlemlerdir.",
        "> Analiz temiz yükleme katmanı (clean_loader) kullanılarak türetilmiş görünüm üzerinde gerçekleştirilmiştir.",
        "",
        "## 1. Final HRC Dağılımı",
        "**Genel**:"
    ]
    
    def fmt_dist(d: dict[str, float]) -> str:
        return f"n={d['n']}, Min={d['min']:.2f}, P25={d['p25']:.2f}, Median={d['median']:.2f}, P75={d['p75']:.2f}, Max={d['max']:.2f}, Mean={d['mean']:.2f}, Std={d['std']:.2f}"
    
    lines.append("- " + fmt_dist(report["final_hrc_dist"]["overall"]))
    lines.append("**Kaynak Bazında**:")
    for src, d in report["final_hrc_dist"]["by_source"].items():
        lines.append(f"- {src}: " + fmt_dist(d))
        
    lines.extend(["", "## 2. Sıcaklık ve Zaman Dağılımları", "### Tempering Temperature (°C)"])
    lines.append("**Genel**: " + fmt_dist(report["temper_temp_c_dist"]["overall"]))
    for src, d in report["temper_temp_c_dist"]["by_source"].items():
        lines.append(f"- {src}: " + fmt_dist(d))
        
    lines.append("### Tempering Time (s)")
    lines.append("**Genel**: " + fmt_dist(report["temper_time_s_dist"]["overall"]))
    for src, d in report["temper_time_s_dist"]["by_source"].items():
        lines.append(f"- {src}: " + fmt_dist(d))
        
    lines.extend(["", "## 3. Kompozisyon Özetleri (%wt)"])
    lines.append("| Element | Min | Median | Max |")
    lines.append("| :--- | :-: | :-: | :-: |")
    for c, d in report["composition_summary"].items():
        lines.append(f"| {c} | {d['min']:.4f} | {d['median']:.4f} | {d['max']:.4f} |")
        
    lines.extend([
        "", "## 4. Spearman Korelasyonları (final_hrc ile)",
        "Gözlem: Sıcaklık ve log10(zaman) ile final HRC arasındaki korelasyonlar.",
        "| Kaynak | vs Sıcaklık | vs log10(Zaman) |",
        "| :--- | :-: | :-: |"
    ])
    overall_temp = report["spearman_temp"]["overall"]
    overall_time = report["spearman_time"]["overall"]
    lines.append(f"| **Genel** | {overall_temp:.4f} | {overall_time:.4f} |")
    
    for src in report["spearman_temp"]["by_source"]:
        t_cor = report["spearman_temp"]["by_source"][src]
        time_cor = report["spearman_time"]["by_source"][src]
        lines.append(f"| {src} | {t_cor:.4f} | {time_cor:.4f} |")
        
    lines.extend([
        "", "## 5. Keşifsel Hollomon-Jaffe Korelasyonu",
        "Hesaplama: `p_hj = (T_K / 1000) * (19.5 + log10(t_saat))`",
        "> **C=19.5 ASSUMED [REPORT_ONLY]** - Modelleme değildir.",
        "| Kaynak | vs P_HJ (Spearman) |",
        "| :--- | :-: |"
    ])
    lines.append(f"| **Genel** | {report['spearman_hj']['overall']:.4f} |")
    for src, cor in report["spearman_hj"]["by_source"].items():
        lines.append(f"| {src} | {cor:.4f} |")
        
    lines.extend([
        "", "## 6. Eksiklik Deseni (Initial HRC Missing)",
        "Gözlem: Hangi kaynakta `initial_hrc` hücresi eksiktir."
    ])
    for missing_status, counts in report["initial_hrc_missing_crosstab"].items():
        lines.append(f"**Missing: {missing_status}**")
        for src, ct in counts.items():
            lines.append(f"- {src}: {ct}")
            
    lines.extend([
        "", "## 7. Bayrak Tutarlılık Kontrolü",
        "Gözlem: Phase 1 ile sayıların tutarlılığı."
    ])
    expected = report["flag_consistency"]["expected"]
    actual = report["flag_consistency"]["actual"]
    lines.append(f"- Beklenen: SUSPECT {expected['SUSPECT']} / WARNING {expected['WARNING']} / INFO {expected['INFO']} / E140 {expected['E140']}")
    lines.append(f"- Gerçekleşen: SUSPECT {actual['SUSPECT']} / WARNING {actual['WARNING']} / INFO {actual['INFO']} / E140 {actual['E140']}")
    
    if not report["flag_consistency"]["discrepancies"]:
        lines.append("- **Sonuç**: Tüm sayılar Phase 1 ile birebir eşleşiyor.")
    else:
        lines.append("- **Farklılıklar (Beklentiden Sapanlar)**:")
        for k, v in report["flag_consistency"]["discrepancies"].items():
            lines.append(f"  - {k}: Beklenen {v['expected']}, Gerçekleşen {v['actual']}")
            
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materials Data Lab - Phase 3 EDA")
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV file.")
    parser.add_argument("--outdir", type=str, default="reports", help="Directory to save reports.")
    parser.add_argument("--json-out", type=str, default="data/inventory/phase3_eda.json", help="Path to save JSON.")
    
    args = parser.parse_args(argv)
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    json_path = Path(args.json_out)
    
    outdir.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = run_eda(input_path)
    
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    
    md_content = generate_markdown_eda(report)
    md_report_path = outdir / "phase3_eda.md"
    md_report_path.write_text(md_content, encoding="utf-8")
    
    print("Phase 3 EDA complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
