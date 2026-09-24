"""Phase 2: Quality Findings Breakdown Analysis.

Reads the raw dataset and threshold findings, generating a detailed 
cross-tabulation and drill-down analysis of anomalies. Does not modify data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from materials_data_lab.thresholds import evaluate_thresholds

def run_breakdown(file_path: Path) -> dict[str, Any]:
    df = pd.read_csv(file_path, keep_default_na=True, encoding="utf-8")
    findings = evaluate_thresholds(df)
    
    source_col = [c for c in df.columns if "Source" in c][0]
    steel_col = "Steel type" if "Steel type" in df.columns else [c for c in df.columns if "type" in c.lower()][0]
    al_col = [c for c in df.columns if "Al " in c or "Al(" in c][0]
    p_col = [c for c in df.columns if "P " in c or "P(" in c][0]
    final_hrc_col = [c for c in df.columns if "Final hardness" in c][0]
    temp_col = [c for c in df.columns if "temperature" in c.lower()][0]
    time_col = [c for c in df.columns if "time" in c.lower()][0]

    flag_x_source: dict[str, dict[str, int]] = {}
    flag_x_steel: dict[str, dict[str, int]] = {}
    al_upper_bound_list = []
    p_upper_bound_list = []

    for f in findings:
        idx = f.row_index
        src = str(df.at[idx, source_col])
        stl = str(df.at[idx, steel_col])
        fname = f.flag_name
        
        flag_x_source.setdefault(fname, {})
        flag_x_source[fname][src] = flag_x_source[fname].get(src, 0) + 1
        
        flag_x_steel.setdefault(fname, {})
        flag_x_steel[fname][stl] = flag_x_steel[fname].get(stl, 0) + 1
        
        if fname == "Al_UPPER_BOUND_SUSPECT":
            al_upper_bound_list.append({
                "Source": src, 
                "Steel type": stl, 
                "Al value": float(df.at[idx, al_col])
            })
            
        if fname == "P_UPPER_BOUND_SUSPECT":
            p_upper_bound_list.append({
                "Source": src, 
                "Steel type": stl, 
                "P value": float(df.at[idx, p_col])
            })

    for fname in flag_x_steel:
        sorted_steel = sorted(flag_x_steel[fname].items(), key=lambda x: x[1], reverse=True)[:10]
        flag_x_steel[fname] = dict(sorted_steel)

    hrc_series = pd.to_numeric(df[final_hrc_col], errors='coerce')
    hrc_under_20_mask = hrc_series < 20.0
    hrc_under_20_df = df[hrc_under_20_mask]
    
    source_breakdown = hrc_under_20_df[source_col].value_counts().to_dict()
    hrc_min = float(hrc_under_20_df[final_hrc_col].min()) if not hrc_under_20_df.empty else None
    hrc_max = float(hrc_under_20_df[final_hrc_col].max()) if not hrc_under_20_df.empty else None
    temp_min = float(hrc_under_20_df[temp_col].min()) if not hrc_under_20_df.empty else None
    temp_max = float(hrc_under_20_df[temp_col].max()) if not hrc_under_20_df.empty else None
    
    under_20_steels = set(hrc_under_20_df[steel_col].unique())
    hrc_ge_20_mask = hrc_series >= 20.0
    ge_20_steels = set(df[hrc_ge_20_mask][steel_col].unique())
    
    overlap_steels = list(under_20_steels.intersection(ge_20_steels))
    overlap_examples = []
    if overlap_steels:
        for stl in overlap_steels[:3]:
            ex = df[(df[steel_col] == stl) & hrc_ge_20_mask].iloc[0]
            overlap_examples.append({
                "Steel type": stl,
                "Source": str(ex[source_col]),
                "Final HRC": float(ex[final_hrc_col])
            })
            
    deep_dive = {
        "source_breakdown": source_breakdown,
        "hrc_min": hrc_min,
        "hrc_max": hrc_max,
        "temp_min": temp_min,
        "temp_max": temp_max,
        "overlap_examples": overlap_examples
    }

    elements = ["C", "Mn", "P", "S", "Si", "Ni", "Cr", "Mo", "V", "Al", "Cu"]
    actual_comp_cols = []
    for e in elements:
        cols = [c for c in df.columns if c.startswith(e + " ") or c.startswith(e + "(")]
        if cols:
            actual_comp_cols.append(cols[0])
        
    key_cols = [source_col, steel_col] + actual_comp_cols + [temp_col, time_col]
    
    grouped = df.groupby(key_cols, dropna=False)
    
    diff_hrc_groups = []
    same_hrc_groups = []
    same_hrc_row_count = 0
    
    for name, group in grouped:
        if len(group) > 1:
            hrc_uniques = group[final_hrc_col].nunique(dropna=False)
            if hrc_uniques > 1:
                # Convert group to list of dicts. Handle NaN so it's JSON serializable
                group_dict = group.fillna("NaN").to_dict('records')
                diff_hrc_groups.append(group_dict)
            else:
                group_dict = group.fillna("NaN").to_dict('records')
                same_hrc_groups.append(group_dict)
                same_hrc_row_count += len(group)
                
    logical_duplicates = {
        "diff_hrc": {
            "group_count": len(diff_hrc_groups),
            "examples": diff_hrc_groups[:5]
        },
        "same_hrc": {
            "row_count": same_hrc_row_count,
            "group_count": len(same_hrc_groups),
            "examples": same_hrc_groups[:5]
        }
    }

    temp_grid = df.groupby(source_col)[temp_col].unique().apply(lambda x: sorted([float(i) for i in x])).to_dict()
    time_regime_df = df.groupby(source_col)[time_col].agg(['count', 'min', 'max', 'median']).reset_index()
    time_regime = time_regime_df.to_dict('records')

    report = {
        "flag_x_source": flag_x_source,
        "flag_x_steel": flag_x_steel,
        "al_upper_bound_list": al_upper_bound_list,
        "p_upper_bound_list": p_upper_bound_list,
        "hrc_under_20_deep_dive": deep_dive,
        "logical_duplicates": logical_duplicates,
        "temp_grid": temp_grid,
        "time_regime": time_regime
    }
    
    return report

def generate_markdown_breakdown(report: dict[str, Any]) -> str:
    lines = [
        "# Materials Data Lab — Phase 2: Quality Findings Breakdown",
        "",
        "> Bu rapor, Phase 1'de bulunan anomalilerin kaynak (Source) ve çelik tipi (Steel type) ",
        "bazında derinlemesine analizini içerir. Veri üzerinde herhangi bir değişiklik yapılmamıştır.",
        "",
        "## 1. Flag x Source Cross-Tab",
        "Hangi bayrakların hangi kaynaklardan geldiğinin dağılımı:",
        ""
    ]
    for fname, src_dict in report["flag_x_source"].items():
        lines.append(f"**{fname}**:")
        for src, count in src_dict.items():
            lines.append(f"- {src}: {count}")
        lines.append("")

    lines.extend([
        "## 2. Flag x Steel Type (Top 10)",
        "Hangi bayrakların en çok hangi çelik tiplerinde görüldüğü:",
        ""
    ])
    for fname, stl_dict in report["flag_x_steel"].items():
        lines.append(f"**{fname}**:")
        for stl, count in stl_dict.items():
            lines.append(f"- `{stl}`: {count}")
        lines.append("")

    lines.extend([
        "## 3. Al_UPPER_BOUND_SUSPECT Tam Liste",
        f"Toplam: {len(report['al_upper_bound_list'])}",
        ""
    ])
    for item in report["al_upper_bound_list"]:
        lines.append(f"- Source: {item['Source']}, Steel type: `{item['Steel type']}`, Al value: {item['Al value']}")

    lines.extend([
        "",
        "## 4. P_UPPER_BOUND_SUSPECT Tam Liste",
        f"Toplam: {len(report['p_upper_bound_list'])}",
        ""
    ])
    for item in report["p_upper_bound_list"]:
        lines.append(f"- Source: {item['Source']}, Steel type: `{item['Steel type']}`, P value: {item['P value']}")

    dd = report["hrc_under_20_deep_dive"]
    lines.extend([
        "",
        "## 5. Final HRC < 20 Deep-Dive",
        f"- **HRC Aralık**: {dd['hrc_min']} - {dd['hrc_max']}",
        f"- **Sıcaklık Bandı**: {dd['temp_min']} - {dd['temp_max']} °C",
        "- **Kaynak Dağılımı**:"
    ])
    for src, cnt in dd["source_breakdown"].items():
        lines.append(f"  - {src}: {cnt}")
        
    lines.append("- **Aynı Çelik Tipi İçin >= 20 HRC Örnekleri**:")
    if dd["overlap_examples"]:
        for ex in dd["overlap_examples"]:
            lines.append(f"  - Steel type: `{ex['Steel type']}` (Source: {ex['Source']}) -> Final HRC: {ex['Final HRC']}")
    else:
        lines.append("  - (Örnek bulunamadı)")

    ld = report["logical_duplicates"]
    lines.extend([
        "",
        "## 6. Logical Duplicate Analysis",
        "Anahtar: Source + Steel type + 11 kompozisyon + temperature + time",
        "",
        f"### a) Aynı Anahtar, FARKLI Final HRC (Grup Sayısı: {ld['diff_hrc']['group_count']})",
        "Örnek Gruplar:"
    ])
    for i, grp in enumerate(ld["diff_hrc"]["examples"], 1):
        lines.append(f"**Grup {i}** (Boyut: {len(grp)}):")
        for row in grp:
            hrc = row.get("Final hardness (HRC) - post tempering", row.get("Final hardness"))
            lines.append(f"  - Row {row.get('Unnamed: 0', '?')}: HRC = {hrc}")

    lines.extend([
        "",
        f"### b) Aynı Anahtar, AYNI Final HRC (Toplam Satır: {ld['same_hrc']['row_count']}, Grup Sayısı: {ld['same_hrc']['group_count']})",
        "Örnek Gruplar:"
    ])
    for i, grp in enumerate(ld["same_hrc"]["examples"], 1):
        lines.append(f"**Grup {i}** (Boyut: {len(grp)}):")
        for row in grp:
            hrc = row.get("Final hardness (HRC) - post tempering", row.get("Final hardness"))
            lines.append(f"  - Row {row.get('Unnamed: 0', '?')}: HRC = {hrc}")

    lines.extend([
        "",
        "## 7. Temperature Grid Matrix (Source x Unique Temperatures)",
    ])
    for src, temps in report["temp_grid"].items():
        lines.append(f"**{src}**:")
        lines.append(f"- Sıcaklıklar (°C): {temps}")

    lines.extend([
        "",
        "## 8. Time Regime Table (Source x Time Stats)",
        "| Source | Count | Min (s) | Max (s) | Median (s) |",
        "| :--- | :-: | :-: | :-: | :-: |"
    ])
    for tr in report["time_regime"]:
        lines.append(f"| {tr['Source']} | {tr['count']} | {tr['min']} | {tr['max']} | {tr['median']} |")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materials Data Lab - Breakdown Analysis (Phase 2)")
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV file.")
    parser.add_argument("--outdir", type=str, default="reports", help="Directory to save reports.")
    parser.add_argument("--json-out", type=str, default="data/inventory/phase2_breakdown.json", help="Path to save JSON.")
    
    args = parser.parse_args(argv)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return 1
        
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = run_breakdown(input_path)
    
    # Save JSON
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Save Markdown
    md_content = generate_markdown_breakdown(report)
    md_report_path = outdir / "phase2_breakdown.md"
    md_report_path.write_text(md_content, encoding="utf-8")
    
    print("Breakdown analysis complete.")
    print(f"  Markdown report saved to: {md_report_path}")
    print(f"  JSON output saved to: {json_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
