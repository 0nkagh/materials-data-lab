"""Tests for Phase 5 Evaluation."""

import pandas as pd
from pathlib import Path
from materials_data_lab.evaluation import assign_hrc_band, assign_temp_band, get_group_mae

def test_assign_hrc_band():
    """Verify HRC band assignment boundaries."""
    assert assign_hrc_band(24.9) == "<25"
    assert assign_hrc_band(25.0) == "25-40"
    assert assign_hrc_band(39.9) == "25-40"
    assert assign_hrc_band(40.0) == "40-55"
    assert assign_hrc_band(54.9) == "40-55"
    assert assign_hrc_band(55.0) == ">=55"
    assert assign_hrc_band(pd.NA) == "UNKNOWN"

def test_assign_temp_band():
    """Verify Temperature band assignment boundaries."""
    assert assign_temp_band(299.9) == "<300"
    assert assign_temp_band(300.0) == "300-500"
    assert assign_temp_band(499.9) == "300-500"
    assert assign_temp_band(500.0) == "500-650"
    assert assign_temp_band(649.9) == "500-650"
    assert assign_temp_band(650.0) == ">=650"
    assert assign_temp_band(pd.NA) == "UNKNOWN"

def test_get_group_mae():
    """Verify out-of-fold residual grouping calculation."""
    df = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "final_hrc": [10.0, 20.0, 10.0, 20.0],
        "pred": [12.0, 19.0, 15.0, 15.0]
    })
    # Group A: |10-12|=2, |20-19|=1 -> MAE = 1.5, n=2
    # Group B: |10-15|=5, |20-15|=5 -> MAE = 5.0, n=2
    res = get_group_mae(df, "group", "pred")
    
    assert res["A"]["mae"] == 1.5
    assert res["A"]["n"] == 2
    assert res["B"]["mae"] == 5.0
    assert res["B"]["n"] == 2

def test_readme_sections():
    """Verify README contains required sections."""
    readme_path = Path("README.md")
    assert readme_path.exists(), "README.md not found"
    
    content = readme_path.read_text(encoding="utf-8")
    assert "## Results" in content, "Missing Results section in README"
    assert "## Usage" in content, "Missing Usage section in README"
    assert "## Project Status" in content, "Missing Project Status section in README"
    assert "## Data Attribution" in content, "Missing Data Attribution section in README"
