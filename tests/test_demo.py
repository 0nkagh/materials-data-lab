import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from materials_data_lab.model_artifact import build_or_load_artifact, _hash_file
from materials_data_lab.demo import check_extrapolation, make_prediction

@pytest.fixture
def synthetic_csv(tmp_path):
    df = pd.DataFrame({
        "c_wt": [0.4, 0.5, 0.6, 0.4],
        "mn_wt": [0.8, 0.9, 0.8, 0.8],
        "p_wt": [0.01, 0.02, 0.01, 0.01],
        "s_wt": [0.01, 0.01, 0.02, 0.01],
        "si_wt": [0.2, 0.3, 0.2, 0.2],
        "ni_wt": [0.0, 0.1, 0.0, 0.0],
        "cr_wt": [0.0, 0.0, 0.1, 0.0],
        "mo_wt": [0.0, 0.0, 0.0, 0.1],
        "v_wt": [0.0, 0.0, 0.0, 0.0],
        "al_wt": [0.0, 0.0, 0.0, 0.0],
        "cu_wt": [0.0, 0.0, 0.0, 0.0],
        "temper_temp_c": [200.0, 300.0, 400.0, 500.0],
        "temper_time_s": [3600.0, 7200.0, 3600.0, 3600.0],
        "final_hrc": [50.0, 45.0, 40.0, 35.0],
        "steel_type": ["A", "B", "C", "D"],
        "initial_hrc": [60.0, 55.0, 50.0, 45.0],
        "data_source": ["Test", "Test", "Test", "Test"]
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_build_or_load_artifact(synthetic_csv, tmp_path):
    models_dir = tmp_path / "models"
    
    # First call: should build
    models1, meta1 = build_or_load_artifact(synthetic_csv, models_dir=models_dir)
    assert meta1["retrained"] is True
    assert (models_dir / "artifact_meta.json").exists()
    assert (models_dir / "rf_final.joblib").exists()
    
    # Second call: should load
    models2, meta2 = build_or_load_artifact(synthetic_csv, models_dir=models_dir)
    assert meta2["retrained"] is False
    assert meta1["csv_sha256"] == meta2["csv_sha256"]
    
    # Change CSV: should rebuild
    with open(synthetic_csv, "a") as f:
        f.write("\n0.4,0.8,0.01,0.01,0.2,0.0,0.0,0.0,0.0,0.0,0.0,600.0,3600.0,30.0,E,40.0,Test")
        
    models3, meta3 = build_or_load_artifact(synthetic_csv, models_dir=models_dir)
    assert meta3["retrained"] is True
    assert meta3["csv_sha256"] != meta1["csv_sha256"]


def test_check_extrapolation():
    good = {
        "c_wt": 0.4, "mn_wt": 0.8, "p_wt": 0.015, "s_wt": 0.015, "si_wt": 0.25,
        "ni_wt": 0.0, "cr_wt": 0.0, "mo_wt": 0.0, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0,
        "temper_temp_c": 400.0, "temper_time_s": 3600.0
    }
    assert "✅" in check_extrapolation(good)
    
    bad = good.copy()
    bad["c_wt"] = 1.0 # Out of bounds (max 0.77)
    assert "⚠" in check_extrapolation(bad)

def test_make_prediction():
    # Create a mock model
    class MockModel:
        def predict(self, X):
            return np.array([42.5])
            
    models = {"rf": MockModel(), "b2": MockModel()}
    
    rf_pred, b2_pred, warn = make_prediction(
        models, 0.4, 0.8, 0.015, 0.015, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400.0, 3600.0
    )
    
    assert rf_pred == "42.5 HRC"
    assert b2_pred == "42.5 HRC"
    assert "✅" in warn
