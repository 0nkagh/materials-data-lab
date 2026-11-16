"""Tests for FastAPI endpoints, artifact readiness, bounds, and input validation."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import materials_data_lab.api as api_mod
from materials_data_lab.api import app
from materials_data_lab.model_artifact import ArtifactLoadError, ArtifactNotReadyError

client = TestClient(app)


class MockModel:
    def predict(self, X):
        return np.full(X.shape[0], 42.5)


dummy_models = {"xgb": MockModel(), "b2": MockModel()}
dummy_meta = {
    "champion": "XGB_tuned",
    "features": [
        "c_wt", "mn_wt", "p_wt", "s_wt", "si_wt",
        "ni_wt", "cr_wt", "mo_wt", "v_wt", "al_wt", "cu_wt",
        "temper_temp_c", "log_time",
    ],
    "conformal_q90": 5.08,
}


@patch("materials_data_lab.api.get_models", return_value=(dummy_models, dummy_meta))
@patch(
    "materials_data_lab.api.make_prediction",
    return_value=(
        "42.5 HRC",
        "OOF empirical band (nominal 90%, not guaranteed): 37.4 – 47.6 HRC (q90 = ±5.08 HRC)",
        "42.5 HRC",
        "✅ Within 1D observed bounds",
        Path("dummy.png"),
        "dummy",
    ),
)
def test_health(mock_make_pred, mock_get_models):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["champion"] == "XGB_tuned"
    assert "model_version" in data


def test_api_returns_503_model_not_ready_without_training(monkeypatch):
    monkeypatch.setattr(
        api_mod,
        "get_models",
        lambda: (_ for _ in ()).throw(ArtifactNotReadyError("Artifacts missing in models/")),
    )
    h = client.get("/health")
    assert h.status_code == 503
    assert h.json()["status"] == "model_not_ready"

    payload_pred = {
        "c_wt": 0.40, "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25,
        "ni_wt": 0.0, "cr_wt": 1.00, "mo_wt": 0.22, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0,
        "temper_temp_c": 500.0, "temper_time_s": 3600.0,
    }
    p = client.post("/predict", json=payload_pred)
    assert p.status_code == 503
    assert p.json()["detail"]["status"] == "model_not_ready"

    monkeypatch.setattr(
        api_mod,
        "get_models",
        lambda: (_ for _ in ()).throw(ArtifactLoadError("Corrupt joblib")),
    )
    payload_rec = {**{k: v for k, v in payload_pred.items() if k not in ("temper_temp_c", "temper_time_s")}, "target_hrc": 42.5}
    r = client.post("/recommend", json=payload_rec)
    assert r.status_code == 503
    assert r.json()["detail"]["status"] == "model_not_ready"


@patch("materials_data_lab.api.get_models", return_value=(dummy_models, dummy_meta))
def test_predict_valid_4140_end_to_end_formatting(mock_get_models, monkeypatch):
    class MockExplainer:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            class MockShapValues:
                values = np.zeros(13)

            return [MockShapValues()]

    import shap
    import matplotlib.pyplot as plt

    monkeypatch.setattr(shap, "TreeExplainer", MockExplainer)
    monkeypatch.setattr(shap.plots, "waterfall", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)

    payload = {
        "c_wt": 0.40, "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25,
        "ni_wt": 0.0, "cr_wt": 1.00, "mo_wt": 0.22, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0,
        "temper_temp_c": 500.0, "temper_time_s": 3600.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["interval_low"] < data["interval_high"]
    assert data["is_extrapolated"] is False
    assert data["within_bounds"] is True
    assert data["warnings"] == []
    assert data["multivariate_support_verified"] is False
    assert "scope_note" in data
    assert "application-supported input scope" in data["scope_note"]
    assert "metallurgical suitability" in data["scope_note"]


@patch("materials_data_lab.api.get_models", return_value=(dummy_models, dummy_meta))
@patch(
    "materials_data_lab.api.make_prediction",
    return_value=(
        "42.5 HRC",
        "OOF empirical band (nominal 90%, not guaranteed): 37.4 – 47.6 HRC (q90 = ±5.08 HRC)",
        "42.5 HRC",
        "⚠ Input outside 1D observed data bounds (c_wt=1.9 %wt outside observed [0.25, 1.15])",
        Path("dummy.png"),
        "dummy",
    ),
)
def test_predict_extrapolated_true(mock_make_pred, mock_get_models):
    payload = {
        "c_wt": 1.9,
        "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25,
        "ni_wt": 0.0, "cr_wt": 1.00, "mo_wt": 0.22, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0,
        "temper_temp_c": 500.0, "temper_time_s": 3600.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_extrapolated"] is True
    assert data["within_bounds"] is False
    assert len(data["warnings"]) > 0


def test_predict_missing_and_extra_invalid_fields():
    assert client.post("/predict", json={"c_wt": 0.40, "mn_wt": 0.85}).status_code == 422

    payload = {
        "c_wt": 0.40, "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25,
        "ni_wt": 0.0, "cr_wt": 1.00, "mo_wt": 0.22, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0,
        "temper_temp_c": 500.0, "temper_time_s": 3600.0,
        "not_a_real_field": 1.0,
    }
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "bad_raw_json",
    [
        '{"c_wt": NaN, "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25, "ni_wt": 0.0, "cr_wt": 1.0, "mo_wt": 0.2, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0, "temper_temp_c": 500.0, "temper_time_s": 3600.0}',
        '{"c_wt": "nan", "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25, "ni_wt": 0.0, "cr_wt": 1.0, "mo_wt": 0.2, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0, "temper_temp_c": 500.0, "temper_time_s": 3600.0}',
        '{"c_wt": true, "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25, "ni_wt": 0.0, "cr_wt": 1.0, "mo_wt": 0.2, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0, "temper_temp_c": 500.0, "temper_time_s": 3600.0}',
        '{"c_wt": 0.4, "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25, "ni_wt": 0.0, "cr_wt": 1.0, "mo_wt": 0.2, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0, "temper_temp_c": 0.0, "temper_time_s": 3600.0}',
        '{"c_wt": 0.4, "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25, "ni_wt": 0.0, "cr_wt": 1.0, "mo_wt": 0.2, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0, "temper_temp_c": 850.0, "temper_time_s": 3600.0}',
        '{"c_wt": 0.4, "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25, "ni_wt": 0.0, "cr_wt": 1.0, "mo_wt": 0.2, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0, "temper_temp_c": 800.1, "temper_time_s": 3600.0}',
        '{"c_wt": 60.0, "mn_wt": 50.0, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25, "ni_wt": 0.0, "cr_wt": 1.0, "mo_wt": 0.2, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0, "temper_temp_c": 500.0, "temper_time_s": 3600.0}',
    ],
)
def test_predict_rejects_nan_bool_str_and_out_of_scope_inputs(bad_raw_json):
    resp = client.post("/predict", content=bad_raw_json, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


@patch("materials_data_lab.api.get_models", return_value=(dummy_models, dummy_meta))
def test_recommend_valid_4140(mock_get_models):
    payload = {
        "c_wt": 0.40, "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25,
        "ni_wt": 0.0, "cr_wt": 1.00, "mo_wt": 0.22, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0,
        "target_hrc": 42.5,
    }

    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommendable"] is True
    assert data["status"] == "CANDIDATE_FOUND"
    assert data["T_c"] == 100.0
    assert data["t_s"] == 1800.0
    assert data["n_solutions"] > 0
    assert data["predicted_hrc"] == 42.5
    assert data["multivariate_support_verified"] is False
    assert 0 < len(data["candidates"]) <= 3
    assert data["candidates"][0]["T_c"] == data["T_c"]
    assert data["is_extrapolated"] is False
    assert data["high_uncertainty"] is False


@patch("materials_data_lab.api.get_models", return_value=(dummy_models, dummy_meta))
def test_recommend_extrapolated_returns_out_of_scope_without_candidates(mock_get_models):
    payload = {
        "c_wt": 1.90,  # Outside 1D observed bounds [0.25, 1.15]
        "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25,
        "ni_wt": 0.0, "cr_wt": 1.00, "mo_wt": 0.22, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0,
        "target_hrc": 42.5,
    }

    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommendable"] is False
    assert data["status"] == "OUT_OF_OBSERVED_SCOPE"
    assert data["is_extrapolated"] is True
    assert data["high_uncertainty"] is True
    assert data["T_c"] is None
    assert data["t_s"] is None
    assert data["candidates"] == []
    assert data["n_solutions"] == 0


@patch("materials_data_lab.api.get_models", return_value=(dummy_models, dummy_meta))
def test_recommend_no_candidate_found_in_range(mock_get_models):
    payload = {
        "c_wt": 0.40, "mn_wt": 0.85, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25,
        "ni_wt": 0.0, "cr_wt": 1.00, "mo_wt": 0.22, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0,
        "target_hrc": 70.0,
    }

    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommendable"] is False
    assert data["status"] == "NO_CANDIDATE_FOUND"
    assert data["T_c"] is None
    assert data["candidates"] == []


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

    ci_path = base / ".github" / "workflows" / "ci.yml"
    with open(ci_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "ruff check ." in content
