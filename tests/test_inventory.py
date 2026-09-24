"""Unit tests for Materials Data Lab Phase 1.

All tests are completely independent of network access, pip install,
and the presence of the actual raw dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from materials_data_lab.inventory import (
    compute_sha256,
    main,
    run_inventory_analysis,
)
from materials_data_lab.thresholds import (
    FlagSeverity,
    evaluate_thresholds,
    summarize_flags,
)


def test_1_non_existent_file_produces_data_not_available(tmp_path: Path):
    """1. Var olmayan dosya -> DATA_NOT_AVAILABLE raporu üretiliyor, process başarılı bitiyor."""
    fake_csv = tmp_path / "non_existent_raw_data.csv"
    out_dir = tmp_path / "reports"
    json_out = tmp_path / "data" / "inventory" / "inventory.json"

    # Call CLI main with non-existent path
    exit_code = main([
        "--input", str(fake_csv),
        "--outdir", str(out_dir),
        "--json-out", str(json_out),
    ])

    assert exit_code == 0, "CLI should exit cleanly with code 0 on DATA_NOT_AVAILABLE"

    report_md = out_dir / "DATA_NOT_AVAILABLE.md"
    assert report_md.exists(), "DATA_NOT_AVAILABLE.md must be generated"
    content = report_md.read_text(encoding="utf-8")
    assert "DATA_NOT_AVAILABLE" in content
    assert str(fake_csv) in content
    assert "Veriyi Nereye Koymalı?" in content

    assert json_out.exists(), "Machine-readable JSON must be generated"
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["status"] == "DATA_NOT_AVAILABLE"
    assert data["searched_path"] == str(fake_csv)
    assert "instruction" in data


def test_2_synthetic_flawed_fixture_counts(tmp_path: Path):
    """2. Bilinen kusurlu küçük SYNTHETİK fixture (1 eksik değer, 1 Inf, 1 mükerrer satır, 1 aralık-dışı değer).
    
    Tüm sayımlar doğru raporlanıyor.
    """
    synth_file = tmp_path / "synthetic_audit.csv"

    # Row 0: valid normal baseline
    # Row 1: duplicate of Row 0 (1 mükerrer satır)
    # Row 2: 1 missing value (NaN in C (%wt))
    # Row 3: 1 Inf value (inf in Tempering time (s))
    # Row 4: 1 out-of-range value (Final hardness = 85.0 -> SUSPECT)
    csv_text = (
        "Source,Steel type,Initial hardness (HRC) - post quenching,Tempering time (s),Tempering temperature (ºC),C (%wt),Mn (%wt),P (%wt),S (%wt),Si (%wt),Ni (%wt),Cr (%wt),Mo (%wt),V (%wt),Al (%wt),Cu (%wt),Final hardness (HRC) - post tempering\n"
        "TestSource,AISI-1045,55.0,3600,400.0,0.45,0.70,0.020,0.025,0.25,0.10,0.15,0.05,0.01,0.02,0.01,45.0\n"
        "TestSource,AISI-1045,55.0,3600,400.0,0.45,0.70,0.020,0.025,0.25,0.10,0.15,0.05,0.01,0.02,0.01,45.0\n"
        "TestSource,AISI-1045,55.0,3600,400.0,,0.70,0.020,0.025,0.25,0.10,0.15,0.05,0.01,0.02,0.01,45.0\n"
        "TestSource,AISI-1045,55.0,inf,400.0,0.45,0.70,0.020,0.025,0.25,0.10,0.15,0.05,0.01,0.02,0.01,45.0\n"
        "TestSource,AISI-1045,55.0,3600,400.0,0.45,0.70,0.020,0.025,0.25,0.10,0.15,0.05,0.01,0.02,0.01,85.0\n"
    )
    synth_file.write_text(csv_text, encoding="utf-8")

    result = run_inventory_analysis(synth_file)

    # 1. Total duplicate rows check (Row 1 is exact duplicate of Row 0)
    assert result["structure_checks"]["total_duplicate_rows"] == 1
    assert len(result["structure_checks"]["duplicate_sample_indices"]) == 2  # indices 0 and 1

    # 2. Missing value check in 'C (%wt)'
    c_info = next(c for c in result["columns_inventory"] if "C (%wt)" in c["column_name"])
    assert c_info["raw_missing_count"] == 1
    assert c_info["non_null_count"] == 4

    # 3. Infinity check in 'Tempering time (s)'
    time_info = next(c for c in result["columns_inventory"] if "Tempering time" in c["column_name"])
    assert time_info["coerced_pos_inf"] == 1

    # 4. Out-of-range value check (Final hardness = 85.0)
    summary = result["threshold_findings_summary"]
    assert summary["counts"]["SUSPECT"] >= 1
    suspect_findings = [f for f in result["sample_findings"] if f["severity"] == FlagSeverity.SUSPECT]
    assert any(f["value"] == 85.0 for f in suspect_findings)


def test_3_sha256_computation_stability(tmp_path: Path):
    """3. SHA-256 hesabı bilinen içerikte kararlı ve doğru."""
    test_file = tmp_path / "known_content.txt"
    sample_bytes = b"Materials Data Lab Phase 1 Deterministic Content"
    test_file.write_bytes(sample_bytes)

    expected_hash = hashlib.sha256(sample_bytes).hexdigest()
    computed_1 = compute_sha256(test_file)
    computed_2 = compute_sha256(test_file)

    assert computed_1 == expected_hash
    assert computed_1 == computed_2


def test_4_gitignore_excludes_raw_data():
    """4. .gitignore içinde data/raw/ kuralı mevcut ve git check-ignore ile doğrulanıyor."""
    gitignore_path = Path(".gitignore")
    assert gitignore_path.exists(), ".gitignore file must exist"
    content = gitignore_path.read_text(encoding="utf-8")
    assert "data/raw/" in content, "data/raw/ rule must be in .gitignore"

    # Verify using git check-ignore command
    proc = subprocess.run(
        ["git", "check-ignore", "data/raw/sample_tempering_dataset.csv"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, "git check-ignore should confirm data/raw/ files are ignored"
    assert "data/raw/" in proc.stdout


def test_5_threshold_evaluation_rules():
    """5. Eşik tablosu birim testi: sentetik satırlarda bayraklar doğru seviyede üretiliyor.
    
    (örn. HRC 75 -> SUSPECT; final HRC 15 -> SUSPECT + E140_EXTRAPOLATION_SUSPECT;
    sıcaklık 120 -> WARNING; C 1.5 -> SUSPECT; sum(Ni+Cr+Mo+V+Si) > 8 -> INFO_HIGH_ALLOY).
    """
    df = pd.DataFrame([
        {
            # Row 0: HRC 75 -> SUSPECT
            "Final hardness (HRC) - post tempering": 75.0,
            "Tempering temperature (ºC)": 400.0,
            "Tempering time (s)": 3600,
            "C (%wt)": 0.40,
            "Ni (%wt)": 0.5,
            "Cr (%wt)": 0.5,
            "Mo (%wt)": 0.2,
            "V (%wt)": 0.05,
            "Si (%wt)": 0.25,
        },
        {
            # Row 1: final HRC 15 -> SUSPECT + E140_EXTRAPOLATION_SUSPECT
            "Final hardness (HRC) - post tempering": 15.0,
            "Tempering temperature (ºC)": 500.0,
            "Tempering time (s)": 3600,
            "C (%wt)": 0.35,
            "Ni (%wt)": 0.5,
            "Cr (%wt)": 0.5,
            "Mo (%wt)": 0.2,
            "V (%wt)": 0.05,
            "Si (%wt)": 0.25,
        },
        {
            # Row 2: sıcaklık 120 -> WARNING
            "Final hardness (HRC) - post tempering": 50.0,
            "Tempering temperature (ºC)": 120.0,
            "Tempering time (s)": 3600,
            "C (%wt)": 0.40,
            "Ni (%wt)": 0.5,
            "Cr (%wt)": 0.5,
            "Mo (%wt)": 0.2,
            "V (%wt)": 0.05,
            "Si (%wt)": 0.25,
        },
        {
            # Row 3: C 1.5 -> SUSPECT
            "Final hardness (HRC) - post tempering": 50.0,
            "Tempering temperature (ºC)": 400.0,
            "Tempering time (s)": 3600,
            "C (%wt)": 1.5,
            "Ni (%wt)": 0.5,
            "Cr (%wt)": 0.5,
            "Mo (%wt)": 0.2,
            "V (%wt)": 0.05,
            "Si (%wt)": 0.25,
        },
        {
            # Row 4: sum(Ni+Cr+Mo+V+Si) > 8 -> INFO_HIGH_ALLOY
            # Ni(3.0) + Cr(2.5) + Mo(1.0) + V(0.4) + Si(1.5) = 8.4 > 8.0
            "Final hardness (HRC) - post tempering": 50.0,
            "Tempering temperature (ºC)": 400.0,
            "Tempering time (s)": 3600,
            "C (%wt)": 0.40,
            "Ni (%wt)": 3.0,
            "Cr (%wt)": 2.5,
            "Mo (%wt)": 1.0,
            "V (%wt)": 0.4,
            "Si (%wt)": 1.5,
        },
    ])

    findings = evaluate_thresholds(df)
    summary = summarize_flags(findings)

    # 1. HRC 75 on Row 0 -> SUSPECT
    row0_findings = [f for f in findings if f.row_index == 0]
    assert any(
        f.severity == FlagSeverity.SUSPECT and f.value == 75.0
        for f in row0_findings
    ), "HRC 75 must trigger SUSPECT"

    # 2. Final HRC 15 on Row 1 -> SUSPECT + E140_EXTRAPOLATION_SUSPECT
    row1_findings = [f for f in findings if f.row_index == 1]
    assert any(
        f.severity == FlagSeverity.SUSPECT and f.value == 15.0
        for f in row1_findings
    ), "Final HRC 15 must trigger SUSPECT"
    assert any(
        f.flag_name == "E140_EXTRAPOLATION_SUSPECT" and f.value == 15.0
        for f in row1_findings
    ), "Final HRC 15 must trigger E140_EXTRAPOLATION_SUSPECT"

    # 3. Sıcaklık 120 on Row 2 -> WARNING
    row2_findings = [f for f in findings if f.row_index == 2]
    assert any(
        f.severity == FlagSeverity.WARNING and f.flag_name == "TEMP_BELOW_WINDOW_WARNING"
        for f in row2_findings
    ), "Temperature 120 must trigger WARNING"

    # 4. C 1.5 on Row 3 -> SUSPECT
    row3_findings = [f for f in findings if f.row_index == 3]
    assert any(
        f.severity == FlagSeverity.SUSPECT and f.flag_name == "C_UPPER_BOUND_SUSPECT"
        for f in row3_findings
    ), "C 1.5 must trigger SUSPECT"

    # 5. sum(Ni+Cr+Mo+V+Si) > 8 on Row 4 -> INFO_HIGH_ALLOY
    row4_findings = [f for f in findings if f.row_index == 4]
    assert any(
        f.flag_name == "INFO_HIGH_ALLOY" and f.severity == FlagSeverity.INFO
        for f in row4_findings
    ), "Alloy sum > 8 must trigger INFO_HIGH_ALLOY"

    # Verify summary counts
    assert summary["counts"]["E140_EXTRAPOLATION_SUSPECT"] == 1
    assert summary["counts"]["INFO_HIGH_ALLOY"] == 1
    assert summary["counts"]["WARNING"] >= 1
    assert summary["counts"]["SUSPECT"] >= 3
