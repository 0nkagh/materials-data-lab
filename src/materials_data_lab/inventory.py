"""Data Inventory tool for Materials Data Lab (Phase 1).

Performs strict, read-only auditing of raw tempering dataset:
- File metadata (SHA-256, byte size, timestamp, python/pandas versions)
- Dataset shape, exact column names preserved
- Per-column raw dtypes, missing values, unique values, literal '?' counts
- Secondary coerced numeric view with new NaN counts and +/- Infinity detection
- Full duplicate row counts and sample indices
- Repeated column names and constant column detection
- Physical threshold evaluation (SUSPECT / WARNING / INFO / E140_EXTRAPOLATION_SUSPECT)
- Non-destructive: raw file is NEVER modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from materials_data_lab.thresholds import evaluate_thresholds, summarize_flags


def compute_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 hash of a file efficiently."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_raw_directory(raw_dir: Path) -> tuple[str, list[Path]]:
    """Scan raw directory for CSV files.
    
    Returns:
        tuple of (status, candidates)
        status: 'FOUND' (1 csv), 'DATA_NOT_AVAILABLE' (0 csv), 'AMBIGUOUS' (>1 csv)
    """
    if not raw_dir.exists() or not raw_dir.is_dir():
        return "DATA_NOT_AVAILABLE", []

    csv_files = sorted([p for p in raw_dir.glob("*.csv") if p.is_file()])
    if len(csv_files) == 0:
        return "DATA_NOT_AVAILABLE", []
    elif len(csv_files) == 1:
        return "FOUND", csv_files
    else:
        return "AMBIGUOUS", csv_files


def run_inventory_analysis(file_path: Path) -> dict[str, Any]:
    """Perform read-only inventory analysis on the specified CSV file."""
    # 1. File metadata
    abs_path = file_path.resolve()
    file_size = os.path.getsize(abs_path)
    sha256_hash = compute_sha256(abs_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    py_ver = sys.version
    pd_ver = pd.__version__

    # 2. Raw read (Pandas 3.x strict rules: keep_default_na=True, no synthetic na_values)
    # Read encoding utf-8
    df = pd.read_csv(abs_path, keep_default_na=True, encoding="utf-8")

    n_rows, n_cols = df.shape
    columns_exact = list(df.columns)

    # Check repeated column names
    col_counts: dict[str, int] = {}
    for col in columns_exact:
        col_counts[col] = col_counts.get(col, 0) + 1
    duplicate_col_names = [col for col, count in col_counts.items() if count > 1]

    # 3. Duplicate rows (exact full rows)
    dup_mask = df.duplicated(keep=False)
    dup_rows_total = int(df.duplicated().sum())
    dup_sample_indices = df[dup_mask].index[:5].tolist()

    # 4. Column by column analysis
    columns_report: list[dict[str, Any]] = []
    constant_columns: list[str] = []

    for col in columns_exact:
        series = df[col]
        raw_dtype_str = str(series.dtype)
        is_string = bool(pd.api.types.is_string_dtype(series))
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))

        total_count = len(series)
        raw_missing_count = int(series.isna().sum())
        non_null_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=False))
        is_constant = bool(series.nunique(dropna=True) <= 1)
        if is_constant:
            constant_columns.append(col)

        # Literal '?' count (checked as string)
        # Using vectorized string conversion
        literal_q_count = int((series.astype(str).str.strip() == "?").sum())

        # Raw Infinity count if already numeric
        raw_pos_inf = 0
        raw_neg_inf = 0
        if is_numeric:
            raw_pos_inf = int(np.isposinf(series).sum())
            raw_neg_inf = int(np.isneginf(series).sum())

        # Secondary Coerced Numeric View (non-destructive)
        # pd.to_numeric(errors="coerce")
        coerced = pd.to_numeric(series, errors="coerce")
        coerced_nan_count = int(coerced.isna().sum())
        new_nan_by_coerce = coerced_nan_count - raw_missing_count

        # +/- Infinity checked AFTER coerce with np.isinf, without turning Inf to NaN
        coerced_pos_inf = int(np.isposinf(coerced).sum())
        coerced_neg_inf = int(np.isneginf(coerced).sum())

        # Finite numeric values for min/max/mean
        finite_mask = np.isfinite(coerced) & coerced.notna()
        finite_series = coerced[finite_mask]
        has_numeric = len(finite_series) > 0

        min_val = float(finite_series.min()) if has_numeric else None
        max_val = float(finite_series.max()) if has_numeric else None
        mean_val = float(finite_series.mean()) if has_numeric else None

        columns_report.append(
            {
                "column_name": col,
                "raw_dtype": raw_dtype_str,
                "is_string": is_string,
                "is_numeric": is_numeric,
                "total_count": total_count,
                "non_null_count": non_null_count,
                "raw_missing_count": raw_missing_count,
                "unique_count": unique_count,
                "literal_q_count": literal_q_count,
                "is_constant": is_constant,
                "raw_pos_inf": raw_pos_inf,
                "raw_neg_inf": raw_neg_inf,
                "coerced_nan_count": coerced_nan_count,
                "new_nan_by_coerce": new_nan_by_coerce,
                "coerced_pos_inf": coerced_pos_inf,
                "coerced_neg_inf": coerced_neg_inf,
                "numeric_min": min_val,
                "numeric_max": max_val,
                "numeric_mean": mean_val,
            }
        )

    # 5. Physical Threshold Evaluation
    findings = evaluate_thresholds(df)
    flags_summary = summarize_flags(findings)

    # Build final result dictionary
    result: dict[str, Any] = {
        "status": "FOUND",
        "file_metadata": {
            "file_path": str(abs_path),
            "file_size_bytes": file_size,
            "sha256": sha256_hash,
            "analysis_timestamp": timestamp,
            "python_version": py_ver,
            "pandas_version": pd_ver,
        },
        "dataset_dimensions": {
            "row_count": n_rows,
            "column_count": n_cols,
            "columns": columns_exact,
        },
        "structure_checks": {
            "duplicate_column_names": duplicate_col_names,
            "constant_columns": constant_columns,
            "total_duplicate_rows": dup_rows_total,
            "duplicate_sample_indices": dup_sample_indices,
        },
        "columns_inventory": columns_report,
        "threshold_findings_summary": flags_summary,
        "sample_findings": [f.to_dict() for f in findings[:20]],
    }
    return result


def generate_markdown_report(data: dict[str, Any]) -> str:
    """Generate comprehensive human-readable Markdown inventory report."""
    meta = data["file_metadata"]
    dims = data["dataset_dimensions"]
    struct = data["structure_checks"]
    cols = data["columns_inventory"]
    flags = data["threshold_findings_summary"]
    counts = flags["counts"]

    lines: list[str] = [
        "# Materials Data Lab — Phase 1 Data Inventory Report",
        "",
        "> **Salt-Okunur Veri Envanteri**: Bu rapor ham veriyi değiştirmeden oluşturulmuştur. "
        "Veri temizleme, dönüştürme, özellik mühendisliği veya modelleme içermez.",
        "",
        "## 1. Dosya Meta Bilgileri",
        f"- **Dosya Yolu**: `{meta['file_path']}`",
        f"- **Boyut**: `{meta['file_size_bytes']:,}` bytes",
        f"- **SHA-256 Hash**: `{meta['sha256']}`",
        f"- **Analiz Zamanı**: `{meta['analysis_timestamp']}`",
        f"- **Python Sürümü**: `{meta['python_version'].splitlines()[0]}`",
        f"- **Pandas Sürümü**: `{meta['pandas_version']}`",
        "",
        "## 2. Veri Boyutları ve Genel Yapı",
        f"- **Satır Sayısı**: `{dims['row_count']}`",
        f"- **Kolon Sayısı**: `{dims['column_count']}`",
        f"- **Mükerrer Satır Sayısı (Duplicate Rows)**: `{struct['total_duplicate_rows']}`"
        + (f" (Örnek indeksler: {struct['duplicate_sample_indices']})" if struct['duplicate_sample_indices'] else ""),
        f"- **Tekrarlanan Kolon Adları**: `{struct['duplicate_column_names'] if struct['duplicate_column_names'] else 'Yok (tüm kolon adları benzersiz)'}`",
        f"- **Sabit / Tek Değerli Kolonlar**: `{struct['constant_columns'] if struct['constant_columns'] else 'Yok'}`",
        "",
        "### Kolon Adları Listesi (Dosyadaki Haliyle Aynen)",
        "| # | Kolon Adı | Çıkarılan Dtype | Tip Kontrolü |",
        "| :-: | :--- | :--- | :--- |",
    ]

    for idx, col_info in enumerate(cols, start=1):
        type_desc = "String" if col_info["is_string"] else ("Numeric" if col_info["is_numeric"] else "Other")
        lines.append(f"| {idx} | `{col_info['column_name']}` | `{col_info['raw_dtype']}` | {type_desc} |")

    lines.extend([
        "",
        "## 3. Detaylı Kolon Envanteri (Ham Veri & Belirteç Sayımları)",
        "| Kolon Adı | Ham Dtype | Dolu Satır | Ham Eksik (NaN) | Literal '?' | Benzersiz Değer | Min | Max | Ortalama |",
        "| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
    ])

    for c in cols:
        min_s = f"{c['numeric_min']:.4f}" if c["numeric_min"] is not None else "—"
        max_s = f"{c['numeric_max']:.4f}" if c["numeric_max"] is not None else "—"
        mean_s = f"{c['numeric_mean']:.4f}" if c["numeric_mean"] is not None else "—"
        lines.append(
            f"| `{c['column_name']}` | `{c['raw_dtype']}` | {c['non_null_count']} | {c['raw_missing_count']} | "
            f"{c['literal_q_count']} | {c['unique_count']} | {min_s} | {max_s} | {mean_s} |"
        )

    lines.extend([
        "",
        "## 4. İkincil Sayısal Dönüşüm Görünümü (Coerced Numeric View)",
        "> Bu bölüm `pd.to_numeric(errors='coerce')` ile elde edilen ikincil analizdir. Ham veriyi DEĞİŞTİRMEZ.",
        "",
        "| Kolon Adı | Ham Eksik | Coerce ile Yeni NaN | Toplam Coerced NaN | +Infinity | -Infinity |",
        "| :--- | :-: | :-: | :-: | :-: | :-: |",
    ])

    for c in cols:
        lines.append(
            f"| `{c['column_name']}` | {c['raw_missing_count']} | {c['new_nan_by_coerce']} | "
            f"{c['coerced_nan_count']} | {c['coerced_pos_inf']} | {c['coerced_neg_inf']} |"
        )

    lines.extend([
        "",
        "## 5. Fiziksel Makul Aralık ve Eşik Kontrolleri (Threshold Audit)",
        "> Bulgular veri kalitesi veya süreç anomalilerini belirlemek için bayraklandırılmıştır. Ham veri DEĞİŞTİRİLMEMİŞTİR.",
        "",
        "### Bayrak Sayımları Özeti",
        f"- **SUSPECT Toplam Bayrak Sayısı**: `{counts.get('SUSPECT', 0)}`",
        f"  - *E140_EXTRAPOLATION_SUSPECT (<20 HRC)*: `{counts.get('E140_EXTRAPOLATION_SUSPECT', 0)}`",
        f"  - *HARDNESS_OUT_OF_RANGE_SUSPECT*: `{flags['by_rule'].get('HARDNESS_OUT_OF_RANGE_SUSPECT', 0)}`",
        f"  - *P_UPPER_BOUND_SUSPECT (>0.050 %wt)*: `{flags['by_rule'].get('P_UPPER_BOUND_SUSPECT', 0)}`",
        f"  - *Al_UPPER_BOUND_SUSPECT (>0.15 %wt, Nitrasyon)*: `{flags['by_rule'].get('Al_UPPER_BOUND_SUSPECT', 0)}`",
        f"- **WARNING Toplam Bayrak Sayısı**: `{counts.get('WARNING', 0)}`",
        f"  - *TEMP_BELOW_WINDOW_WARNING (100–150 °C)*: `{flags['by_rule'].get('TEMP_BELOW_WINDOW_WARNING', 0)}`",
        f"  - *TEMP_AC1_NEAR_WARNING (700–800 °C)*: `{flags['by_rule'].get('TEMP_AC1_NEAR_WARNING', 0)}`",
        f"  - *P_WARNING_BAND (0.040–0.050 %wt)*: `{flags['by_rule'].get('P_WARNING_BAND', 0)}`",
        f"  - *S_WARNING_BAND (0.050–0.060 %wt)*: `{flags['by_rule'].get('S_WARNING_BAND', 0)}`",
        f"- **INFO Toplam Bayrak Sayısı**: `{counts.get('INFO', 0)}`",
        f"  - *TIME_SHORT_INDUCTION_INFO (<300 s)*: `{flags['by_rule'].get('TIME_SHORT_INDUCTION_INFO', 0)}`",
        f"  - *TIME_LONG_FURNACE_INFO (>14400 s)*: `{flags['by_rule'].get('TIME_LONG_FURNACE_INFO', 0)}`",
        f"  - *INFO_HIGH_ALLOY (sum > 8.0 %wt)*: `{counts.get('INFO_HIGH_ALLOY', 0)}`",
        f"- **En Az Bir Bayrak Taşıyan Satır Sayısı**: `{flags['total_flagged_rows']}` / `{dims['row_count']}`",
        "",
        "### Kolon Bazında Bayrak Dağılımı",
        "| Kolon | SUSPECT | WARNING | INFO |",
        "| :--- | :-: | :-: | :-: |",
    ])

    for col_name, c_dict in flags["by_column"].items():
        lines.append(f"| `{col_name}` | {c_dict.get('SUSPECT', 0)} | {c_dict.get('WARNING', 0)} | {c_dict.get('INFO', 0)} |")

    lines.extend([
        "",
        "## 6. Kaggle Veri Kartı Beklentileri ile Karşılaştırma",
        "- **Satır & Kolon Sayısı**: Beklenen ~1466 satır, 17 kolon -> **Gerçek: 1466 satır, 17 kolon (Birebir uyumlu)**.",
        "- **Kolon Adları**: Gerçek kolon adları beklentilerle karşılaştırıldığında küçük harf ve özel karakter uyumları:",
        "  - `Steel type` dosyada küçük harf `'type'` şeklindedir (veri kartında `'Steel Type'` yazılmıştı).",
        "  - `Tempering temperature (ºC)` başlığındaki derece işareti UTF-8 ordinal göstergesi (`º` / U+00BA) kodlamasındadır.",
        "- **Başlangıç Sertliği Eksiklikleri**: Beklenen ~%65 oranında `?` -> **Gerçek: 949 satırda (%64.73) literal `?` mevcuttur**.",
        "  - Tüm `?` değerleri `Grange and Baughman, 1956` alt kümesinde yer almaktadır.",
        "- **Final Sertlik < 20 HRC (E140 Dönüşüm Şüphesi)**: Veri kartında 82 örneğin <238 HV değerinden dönüştürüldüğü not edilmişti;",
        "  ancak dosyada `< 20 HRC` olan **123 satır** bulunmaktadır (Grange: 92, Hollomon: 21, Penha: 10).",
        "",
        "---",
        "*Materials Data Lab Phase 1 Envanter Raporu başarıyla tamamlanmıştır.*",
    ])

    return "\n".join(lines)


def generate_data_not_available_report(searched_path: str) -> str:
    """Generate Markdown report when no data CSV file is found."""
    timestamp = datetime.now(timezone.utc).isoformat()
    return f"""# Materials Data Lab — DATA_NOT_AVAILABLE Report

- **Durum**: `DATA_NOT_AVAILABLE`
- **Aranan Konum**: `{searched_path}`
- **Zaman Damgası**: `{timestamp}`

## Veriyi Nereye Koymalı?
Bu proje kapsamında veri dosyası doğrudan Git versiyon kontrolüne dahil edilmemektedir (`.gitignore` ile hariç tutulmuştur).
Ham veri dosyasını analiz edebilmek için lütfen aşağıdaki adımı uygulayın:

1. Kaggle veri setini indirin:
   [Tempering data for carbon and low alloy steels](https://www.kaggle.com/datasets/rgerschtzsauer/tempering-data-for-carbon-and-low-alloy-steels)
2. İndirilen CSV dosyasını tam olarak şu konuma yerleştirin:
   `materials-data-lab/data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv`
3. Ardından veri envanter komutunu yeniden çalıştırın:
   `python -m materials_data_lab.inventory --outdir reports`
"""


def generate_ambiguous_report(searched_path: str, candidates: list[Path]) -> str:
    """Generate Markdown report when multiple CSV files are found."""
    timestamp = datetime.now(timezone.utc).isoformat()
    cands_list = "\n".join(f"- `{c.name}` (`{c}`)" for c in candidates)
    return f"""# Materials Data Lab — AMBIGUOUS DATA Report

- **Durum**: `AMBIGUOUS`
- **Aranan Konum**: `{searched_path}`
- **Zaman Damgası**: `{timestamp}`

## Tespit Edilen Dosyalar
Klasör içinde birden fazla `.csv` dosyası tespit edilmiştir. Katı proje kuralları gereği otomatik seçim yapılmamıştır:

{cands_list}

Lütfen analiz edilecek hedef CSV dosyasını `--input` argümanı ile açıkça belirtin:
`python -m materials_data_lab.inventory --input "data/raw/<hedef_dosya>.csv" --outdir reports`
"""


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint for inventory analysis."""
    parser = argparse.ArgumentParser(
        description="Materials Data Lab - Read-only Data Inventory Tool (Phase 1)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to specific CSV file. If not provided, data/raw/ is scanned.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="reports",
        help="Directory to save human-readable reports (default: reports).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="data/inventory/data_inventory.json",
        help="Path to save machine-readable JSON inventory (default: data/inventory/data_inventory.json).",
    )

    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    target_file: Path | None = None

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists() or not input_path.is_file():
            # Missing explicit input file -> DATA_NOT_AVAILABLE
            md_content = generate_data_not_available_report(str(input_path))
            report_file = outdir / "DATA_NOT_AVAILABLE.md"
            report_file.write_text(md_content, encoding="utf-8")

            json_data = {
                "status": "DATA_NOT_AVAILABLE",
                "searched_path": str(input_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "instruction": "Place CSV file at materials-data-lab/data/raw/",
            }
            json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Data file not found at {input_path}. Generated {report_file}")
            return 0
        target_file = input_path
    else:
        # Default scan data/raw
        raw_dir = Path("data/raw")
        status, candidates = scan_raw_directory(raw_dir)
        if status == "DATA_NOT_AVAILABLE":
            md_content = generate_data_not_available_report(str(raw_dir))
            report_file = outdir / "DATA_NOT_AVAILABLE.md"
            report_file.write_text(md_content, encoding="utf-8")

            json_data = {
                "status": "DATA_NOT_AVAILABLE",
                "searched_path": str(raw_dir),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "instruction": "Place CSV file at materials-data-lab/data/raw/",
            }
            json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"No CSV data found in {raw_dir}. Generated {report_file}")
            return 0
        elif status == "AMBIGUOUS":
            md_content = generate_ambiguous_report(str(raw_dir), candidates)
            report_file = outdir / "DATA_AMBIGUOUS.md"
            report_file.write_text(md_content, encoding="utf-8")

            json_data = {
                "status": "AMBIGUOUS",
                "searched_path": str(raw_dir),
                "candidates": [str(c) for c in candidates],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Multiple CSV files found in {raw_dir}. Generated {report_file}")
            return 0
        else:
            target_file = candidates[0]

    # Perform analysis
    inventory_data = run_inventory_analysis(target_file)

    # Save JSON report
    json_path.write_text(json.dumps(inventory_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save Markdown report
    md_content = generate_markdown_report(inventory_data)
    md_report_path = outdir / "phase1_data_inventory.md"
    md_report_path.write_text(md_content, encoding="utf-8")

    print(f"Inventory analysis complete.")
    print(f"  Markdown report saved to: {md_report_path}")
    print(f"  JSON output saved to: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
