from fastapi.testclient import TestClient
import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from materials_data_lab.api import app
from pathlib import Path
from unittest.mock import patch

client = TestClient(app)

# Dummy models and meta for testing
class MockModel:
    def predict(self, X):
        import numpy as np
        return np.full(X.shape[0], 42.5)

dummy_models = {"xgb": MockModel(), "b2": MockModel()}
dummy_meta = {"champion": "XGB_tuned", "features": ["c_wt", "mn_wt", "p_wt", "s_wt", "si_wt", "ni_wt", "cr_wt", "mo_wt", "v_wt", "al_wt", "cu_wt", "temper_temp_c", "log_time"], "conformal_q90": 5.08}

@patch("materials_data_lab.api.get_models", return_value=(dummy_models, dummy_meta))
@patch("materials_data_lab.api.make_prediction", return_value=("42.5 HRC", "90% interval: 37.4 \u2013 47.6 HRC (q90 = \u00b15.08 HRC)", "42.5 HRC", "\u2705", Path("dummy.png"), "dummy"))
def test_health(mock_make_pred, mock_get_models):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["champion"] == "XGB_tuned"
    assert "model_version" in data

@patch("materials_data_lab.api.get_models", return_value=(dummy_models, dummy_meta))
@patch("materials_data_lab.api.make_prediction", return_value=("42.5 HRC", "90% interval: 37.4 \u2013 47.6 HRC (q90 = \u00b15.08 HRC)", "42.5 HRC", "\u2705", Path("dummy.png"), "dummy"))
def test_predict_valid_4140(mock_make_pred, mock_get_models):
    # Typical 4140 composition
    payload = {
        "c_wt": 0.40,
        "mn_wt": 0.85,
        "p_wt": 0.01,
        "s_wt": 0.01,
        "si_wt": 0.25,
        "ni_wt": 0.0,
        "cr_wt": 1.00,
        "mo_wt": 0.22,
        "v_wt": 0.0,
        "al_wt": 0.0,
        "cu_wt": 0.0,
        "temper_temp_c": 500.0,
        "temper_time_s": 3600.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "champion_hrc" in data
    assert "physics_hrc" in data
    assert "interval_low" in data
    assert "interval_high" in data
    assert "q90" in data
    assert "within_bounds" in data
    assert "warnings" in data
    assert "is_extrapolated" in data
    
    assert data["interval_low"] < data["interval_high"]
    assert data["is_extrapolated"] is False

@patch("materials_data_lab.api.get_models", return_value=(dummy_models, dummy_meta))
@patch("materials_data_lab.api.make_prediction", return_value=("42.5 HRC", "90% interval: 37.4 – 47.6 HRC (q90 = ±5.08 HRC)", "42.5 HRC", "⚠ Input outside observed data range — extrapolation", Path("dummy.png"), "dummy"))
def test_predict_extrapolated_true(mock_make_pred, mock_get_models):
    payload = {
        "c_wt": 1.9, # Extrapolated
        "mn_wt": 0.85,
        "p_wt": 0.01,
        "s_wt": 0.01,
        "si_wt": 0.25,
        "ni_wt": 0.0,
        "cr_wt": 1.00,
        "mo_wt": 0.22,
        "v_wt": 0.0,
        "al_wt": 0.0,
        "cu_wt": 0.0,
        "temper_temp_c": 500.0,
        "temper_time_s": 3600.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["is_extrapolated"] is True
    assert data["within_bounds"] is False
    assert len(data["warnings"]) > 0

def test_predict_missing_field():
    payload = {
        "c_wt": 0.40,
        "mn_wt": 0.85,
        # missing p_wt, s_wt, etc
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_extra_invalid_field():
    payload = {
        "c_wt": 0.40,
        "mn_wt": 0.85,
        "p_wt": 0.01,
        "s_wt": 0.01,
        "si_wt": 0.25,
        "ni_wt": 0.0,
        "cr_wt": 1.00,
        "mo_wt": 0.22,
        "v_wt": 0.0,
        "al_wt": 0.0,
        "cu_wt": 0.0,
        "temper_temp_c": 500.0,
        "temper_time_s": 3600.0,
        "not_a_real_field": 1.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

@patch("materials_data_lab.api.get_models", return_value=(dummy_models, dummy_meta))
def test_recommend_valid_4140(mock_get_models):
    payload = {
        "c_wt": 0.40,
        "mn_wt": 0.85,
        "p_wt": 0.01,
        "s_wt": 0.01,
        "si_wt": 0.25,
        "ni_wt": 0.0,
        "cr_wt": 1.00,
        "mo_wt": 0.22,
        "v_wt": 0.0,
        "al_wt": 0.0,
        "cu_wt": 0.0,
        "target_hrc": 42.5
    }
    
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommendable"] is True
    assert data["T_c"] == 100.0
    assert data["t_s"] == 1800.0
    assert data["n_solutions"] > 0
    assert data["predicted_hrc"] == 42.5
    
    assert "candidates" in data
    assert len(data["candidates"]) > 0
    assert len(data["candidates"]) <= 3
    
    # Check consistency between best_row and candidates[0]
    assert data["candidates"][0]["T_c"] == data["T_c"]
    assert data["candidates"][0]["t_s"] == data["t_s"]
    assert data["candidates"][0]["predicted_hrc"] == data["predicted_hrc"]
    
    assert "is_extrapolated" in data
    assert "high_uncertainty" in data
    assert data["is_extrapolated"] is False
    assert data["high_uncertainty"] is False # n_solutions > 5 for dummy model

@patch("materials_data_lab.api.get_models", return_value=(dummy_models, dummy_meta))
def test_recommend_extrapolated_and_high_uncertainty(mock_get_models):
    payload = {
        "c_wt": 1.90, # Extrapolated
        "mn_wt": 0.85,
        "p_wt": 0.01,
        "s_wt": 0.01,
        "si_wt": 0.25,
        "ni_wt": 0.0,
        "cr_wt": 1.00,
        "mo_wt": 0.22,
        "v_wt": 0.0,
        "al_wt": 0.0,
        "cu_wt": 0.0,
        "target_hrc": 42.5
    }
    
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_extrapolated"] is True
    assert data["high_uncertainty"] is True
    # Still provides solutions since target is achievable by dummy model
    assert data["recommendable"] is True
    assert len(data["candidates"]) > 0

@patch("materials_data_lab.api.get_models", return_value=(dummy_models, dummy_meta))
def test_recommend_impossible(mock_get_models):
    payload = {
        "c_wt": 0.40,
        "mn_wt": 0.85,
        "p_wt": 0.01,
        "s_wt": 0.01,
        "si_wt": 0.25,
        "ni_wt": 0.0,
        "cr_wt": 1.00,
        "mo_wt": 0.22,
        "v_wt": 0.0,
        "al_wt": 0.0,
        "cu_wt": 0.0,
        "target_hrc": 70.0
    }
    
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommendable"] is False
    assert data["T_c"] is None

def test_ruff_ci_config():
    base = Path(__file__).parent.parent
    pyproject_path = base / "pyproject.toml"
    assert pyproject_path.exists()
    
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)
        
    assert "tool" in config
    assert "ruff" in config["tool"]
    assert "lint" in config["tool"]["ruff"]
    assert "E" in config["tool"]["ruff"]["lint"]["select"]
    assert "F" in config["tool"]["ruff"]["lint"]["select"]
    
    # Check CI workflow
    ci_path = base / ".github" / "workflows" / "ci.yml"
    with open(ci_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "ruff check ." in content
