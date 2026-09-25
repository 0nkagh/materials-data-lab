"""Tests for Phase 3 Clean Loader and EDA."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from materials_data_lab.clean_loader import load_clean
from materials_data_lab.eda import run_eda, generate_markdown_eda

@pytest.fixture
def synthetic_csv_phase3(tmp_path: Path) -> Path:
    """Create a synthetic dataset for testing clean loader."""
    data = {
        "Source": [" SrcA ", "SrcB", "Grange and Baughman, 1956"],
        "Steel type": ["Type1 ", "", "Type3"],
        "Initial hardness (HRC) - post quenching": ["?", "50.0", "?"],
        "Tempering time (s)": [3600, 3600, 3600],
        "Tempering temperature (ºC)": [600.0, 600.0, 600.0],
        "C (%wt)": [0.4, 0.5, 0.4],
        "Mn (%wt)": [0.7, 0.8, 0.7],
        "P (%wt)": [0.03, 0.03, 0.03],
        "S (%wt)": [0.02, 0.02, 0.02],
        "Si (%wt)": [0.2, 0.2, 0.2],
        "Ni (%wt)": [0.0, 0.0, 0.0],
        "Cr (%wt)": [0.0, 0.0, 0.0],
        "Mo (%wt)": [0.0, 0.0, 0.0],
        "V (%wt)": [0.0, 0.0, 0.0],
        "Al (%wt)": [0.0, 0.0, 0.0],
        "Cu (%wt)": [0.0, 0.0, 0.0],
        "Final hardness (HRC) - post tempering": [45.0, 15.0, 45.0], # 15.0 is < 20 HRC -> E140_EXTRAPOLATION_SUSPECT
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "synthetic_phase3.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_clean_loader_features(synthetic_csv_phase3: Path):
    """Test the clean loader rules and manifest."""
    df, manifest = load_clean(synthetic_csv_phase3)
    
    # Check 17 slug columns + flags + initial_hrc_missing = 19 cols
    expected_slugs = [
        "source", "steel_type", "initial_hrc", "temper_time_s", "temper_temp_c",
        "c_wt", "mn_wt", "p_wt", "s_wt", "si_wt", "ni_wt", "cr_wt", "mo_wt",
        "v_wt", "al_wt", "cu_wt", "final_hrc"
    ]
    for s in expected_slugs:
        assert s in df.columns, f"Missing slug column: {s}"
        
    # Check strip
    assert df.at[0, "source"] == "SrcA"
    assert df.at[0, "steel_type"] == "Type1"
    assert manifest["counts"]["nk1_strip_affected_cells"] > 0
    
    # Check UNKNOWN_GRADE
    assert df.at[1, "steel_type"] == "UNKNOWN_GRADE"
    assert manifest["counts"]["d05_unknown_grade_rows"] == 1
    
    # Check "?" to NaN (missing HRC)
    assert pd.isna(df.at[0, "initial_hrc"])
    assert pd.isna(df.at[2, "initial_hrc"])
    assert df.at[0, "initial_hrc_missing"]
    assert not df.at[1, "initial_hrc_missing"]
    assert manifest["counts"]["initial_hrc_missing_count"] == 2
    
    # Check flags
    # Row 1 has final_hrc = 15.0 < 20 -> E140 and HARDNESS_OUT_OF_RANGE
    flags_row1 = df.at[1, "flags"]
    assert "E140_EXTRAPOLATION_SUSPECT" in flags_row1
    assert "HARDNESS_OUT_OF_RANGE_SUSPECT" in flags_row1


def test_hollomon_jaffe_calculation(synthetic_csv_phase3: Path):
    """Test HJ unit test: T=600C, t=1h (3600s)."""
    df, _ = load_clean(synthetic_csv_phase3)
    # The first row has 600C and 3600s
    temp_k = df.at[0, "temper_temp_c"] + 273.15
    t_saat = df.at[0, "temper_time_s"] / 3600.0
    p_hj = (temp_k / 1000.0) * (19.5 + np.log10(t_saat))
    
    # 600C = 873.15K. t_saat=1 -> log10(1)=0. p_hj = 0.87315 * 19.5
    expected_p_hj = (873.15 / 1000.0) * 19.5
    assert np.isclose(p_hj, expected_p_hj)


def test_eda_generation(synthetic_csv_phase3: Path):
    """Test that EDA generates valid outputs deterministically."""
    report = run_eda(synthetic_csv_phase3)
    
    # Check some keys
    assert "final_hrc_dist" in report
    assert "spearman_hj" in report
    
    # Missingness pattern
    ct = report["initial_hrc_missing_crosstab"]
    # Row 0 and Row 2 have missing=True. Source is 'SrcA' and 'Grange and Baughman, 1956'
    assert ct[True]["Grange and Baughman, 1956"] == 1
    
    # Markdown gen
    md = generate_markdown_eda(report)
    assert "C=19.5 ASSUMED [REPORT_ONLY]" in md
    assert "Phase 3: Sayısal Keşifsel Veri Analizi" in md
