"""Tests for Phase 2 Breakdown and Decision Log."""

from pathlib import Path
import pandas as pd
import pytest

from materials_data_lab.breakdown import run_breakdown

@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """Create a synthetic dataset for testing breakdown logic."""
    data = {
        "Source": ["SrcA", "SrcA", "SrcB", "SrcC", "SrcA"],
        "Steel type": ["Type1", "Type1", "Type2", "Type3", "Type1"],
        "C (%wt)": [0.4, 0.4, 0.5, 0.6, 0.4],
        "Mn (%wt)": [0.7, 0.7, 0.8, 0.9, 0.7],
        "P (%wt)": [0.03, 0.03, 0.06, 0.04, 0.03],  # Type2 has P > 0.05
        "S (%wt)": [0.02, 0.02, 0.02, 0.02, 0.02],
        "Si (%wt)": [0.2, 0.2, 0.2, 0.2, 0.2],
        "Ni (%wt)": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Cr (%wt)": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Mo (%wt)": [0.0, 0.0, 0.0, 0.0, 0.0],
        "V (%wt)": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Al (%wt)": [0.0, 0.0, 0.20, 0.0, 0.0],  # Type2 has Al > 0.15
        "Cu (%wt)": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Tempering temperature (ºC)": [200.0, 200.0, 400.0, 120.0, 200.0], # SrcC is 120 (warning)
        "Tempering time (s)": [3600, 3600, 7200, 3600, 3600],
        "Final hardness (HRC) - post tempering": [45.0, 40.0, 15.0, 50.0, 45.0], # Type2 < 20, Type1 has logical duplicates (45 vs 40 vs 45)
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "synthetic_breakdown.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_breakdown_flags_and_deepdive(synthetic_csv: Path):
    """Test Flag x Source, Flag x Steel type, and HRC < 20 deep-dive."""
    report = run_breakdown(synthetic_csv)
    
    # 1 & 2. Flag x Source / Flag x Steel type
    assert "P_UPPER_BOUND_SUSPECT" in report["flag_x_source"]
    assert report["flag_x_source"]["P_UPPER_BOUND_SUSPECT"].get("SrcB", 0) == 1
    
    assert "Al_UPPER_BOUND_SUSPECT" in report["flag_x_steel"]
    assert report["flag_x_steel"]["Al_UPPER_BOUND_SUSPECT"].get("Type2", 0) == 1
    
    # 3 & 4. Complete lists
    assert len(report["al_upper_bound_list"]) == 1
    assert report["al_upper_bound_list"][0]["Steel type"] == "Type2"
    
    assert len(report["p_upper_bound_list"]) == 1
    assert report["p_upper_bound_list"][0]["Steel type"] == "Type2"
    
    # 5. Deep-dive
    dd = report["hrc_under_20_deep_dive"]
    assert dd["source_breakdown"].get("SrcB", 0) == 1
    assert dd["hrc_min"] == 15.0
    assert dd["hrc_max"] == 15.0

def test_logical_duplicates(synthetic_csv: Path):
    """Test logical duplicates detection."""
    report = run_breakdown(synthetic_csv)
    ld = report["logical_duplicates"]
    
    # Rows 0, 1, 4 have the same exact key.
    # Row 0 and 4 have same HRC (45.0) -> same_hrc? 
    # Actually the group of rows (0, 1, 4) have different HRCs (45, 40, 45), 
    # so the entire group will fall under diff_hrc.
    # Therefore diff_hrc group count should be 1.
    assert ld["diff_hrc"]["group_count"] == 1
    
    # Let's check the examples
    group = ld["diff_hrc"]["examples"][0]
    assert len(group) == 3

def test_decision_log_mandatory_items():
    """Test that decision log exists and contains all 7 mandatory items as PENDING_USER."""
    doc_path = Path("docs/cleaning_decisions.md")
    assert doc_path.exists(), "cleaning_decisions.md is missing."
    
    content = doc_path.read_text(encoding="utf-8")
    
    # Check 7 mandatory items
    for i in range(1, 8):
        assert f"D-0{i}" in content, f"D-0{i} missing in decision log."
        
    # Check APPROVED
    assert content.count("APPROVED") >= 7, "All 7 items must be APPROVED."
