"""Tests for safe artifact loading, explicit training, and prediction helpers."""

import json

import numpy as np
import pandas as pd
import pytest

from materials_data_lab.demo import check_extrapolation, make_prediction
from materials_data_lab.model_artifact import (
    ArtifactLoadError,
    ArtifactNotReadyError,
    ArtifactOverwriteError,
    _hash_file,
    build_or_load_artifact,
    load_artifact,
    train_and_save_artifact,
)
from materials_data_lab.modeling import RF_FEATURES


@pytest.fixture
def synthetic_csv(tmp_path):
    df = pd.DataFrame({
        "c_wt": [0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4],
        "mn_wt": [0.8, 0.9, 0.8, 0.8, 0.9, 0.8, 0.8],
        "p_wt": [0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01],
        "s_wt": [0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01],
        "si_wt": [0.2, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2],
        "ni_wt": [0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
        "cr_wt": [0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0],
        "mo_wt": [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1],
        "v_wt": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "al_wt": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "cu_wt": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "temper_temp_c": [200.0, 300.0, 400.0, 500.0, 300.0, 400.0, 500.0],
        "temper_time_s": [3600.0, 7200.0, 3600.0, 3600.0, 7200.0, 3600.0, 3600.0],
        "final_hrc": [50.0, 45.0, 40.0, 35.0, 45.0, 40.0, 35.0],
        "steel_type": ["A", "B", "C", "D", "E", "F", "G"],
        "initial_hrc": [60.0, 55.0, 50.0, 45.0, 55.0, 50.0, 45.0],
        "data_source": ["Test", "Test", "Test", "Test", "Test", "Test", "Test"],
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_missing_artifact_does_not_trigger_training(tmp_path, monkeypatch):
    models_dir = tmp_path / "missing_models"
    called = False

    def fail_if_trained(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Automatic training must not run when artifact is missing")

    monkeypatch.setattr("materials_data_lab.model_artifact.train_and_save_artifact", fail_if_trained)

    with pytest.raises(ArtifactNotReadyError, match="Missing|does not exist"):
        load_artifact(models_dir=models_dir)
    with pytest.raises(ArtifactNotReadyError):
        build_or_load_artifact(models_dir=models_dir)
    assert called is False


def test_load_failure_does_not_retrain_or_modify_existing_files(tmp_path, monkeypatch):
    models_dir = tmp_path / "fake_models"
    models_dir.mkdir(parents=True)

    meta_path = models_dir / "artifact_meta.json"
    xgb_path = models_dir / "xgb_champion.joblib"
    b2_path = models_dir / "b2_linear.joblib"

    xgb_bytes = b"dummy-xgb-pickle-content"
    b2_bytes = b"dummy-b2-pickle-content"
    xgb_path.write_bytes(xgb_bytes)
    b2_path.write_bytes(b2_bytes)
    meta_path.write_text(
        json.dumps({
            "schema_version": 2,
            "conformal_q90": 4.97,
            "features": RF_FEATURES,
            "sklearn_version": "0.0.1-test",
        }),
        encoding="utf-8",
    )
    meta_hash_before = _hash_file(meta_path)

    train_called = False

    def fail_if_trained(*args, **kwargs):
        nonlocal train_called
        train_called = True
        raise AssertionError("Training must not run on load failure")

    def failing_joblib_load(path):
        raise ValueError("Simulated pickle incompatibility")

    monkeypatch.setattr("materials_data_lab.model_artifact.train_and_save_artifact", fail_if_trained)
    monkeypatch.setattr("materials_data_lab.model_artifact.joblib.load", failing_joblib_load)

    with pytest.raises(ArtifactLoadError) as exc_info:
        load_artifact(models_dir=models_dir)

    assert "Simulated pickle incompatibility" in str(exc_info.value)
    assert "version differences" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert train_called is False
    # Existing files remain byte-for-byte untouched
    assert xgb_path.read_bytes() == xgb_bytes
    assert b2_path.read_bytes() == b2_bytes
    assert _hash_file(meta_path) == meta_hash_before


def test_version_mismatch_logs_warning_but_does_not_reject_valid_load(tmp_path, monkeypatch):
    models_dir = tmp_path / "version_diff_models"
    models_dir.mkdir()
    (models_dir / "xgb_champion.joblib").write_bytes(b"fake1")
    (models_dir / "b2_linear.joblib").write_bytes(b"fake2")
    (models_dir / "artifact_meta.json").write_text(
        json.dumps({
            "schema_version": 2,
            "conformal_q90": 5.08,
            "features": RF_FEATURES,
            "sklearn_version": "1.0.0-old",
            "pandas_version": "2.0.0-old",
        }),
        encoding="utf-8",
    )

    dummy_obj = object()
    monkeypatch.setattr("materials_data_lab.model_artifact.joblib.load", lambda p: dummy_obj)

    # Succeeds even with NO CSV on disk and mismatched metadata library versions
    models, meta = load_artifact(models_dir=models_dir)
    assert models["xgb"] is dummy_obj
    assert meta["retrained"] is False
    assert "sklearn_version" in meta["version_mismatches"]
    assert "pandas_version" in meta["version_mismatches"]


def test_explicit_train_respects_overwrite_guard_and_csv_verification(synthetic_csv, tmp_path):
    models_dir = tmp_path / "trained_models"

    # Explicit training works on clean directory
    _, meta1 = train_and_save_artifact(synthetic_csv, models_dir=models_dir, overwrite=False)
    assert meta1["retrained"] is True
    assert meta1["schema_version"] == 2
    assert "artifact_hashes" in meta1
    assert (models_dir / "xgb_champion.joblib").is_file()

    # Refuses overwrite without explicit opt-in even if called directly
    with pytest.raises(ArtifactOverwriteError):
        train_and_save_artifact(synthetic_csv, models_dir=models_dir, overwrite=False)

    # Load without CSV succeeds
    _, loaded_meta = load_artifact(models_dir=models_dir)
    assert loaded_meta["retrained"] is False

    # Appending to CSV fails explicit CSV hash check WITHOUT auto-retraining
    with open(synthetic_csv, "a", encoding="utf-8") as f:
        f.write("\n0.4,0.8,0.01,0.01,0.2,0.0,0.0,0.0,0.0,0.0,0.0,600.0,3600.0,30.0,E,40.0,Test")
    with pytest.raises(ArtifactNotReadyError, match="CSV SHA-256 mismatch"):
        load_artifact(models_dir=models_dir, csv_path=synthetic_csv, verify_csv_hash=True)

    # Explicit overwrite succeeds when requested
    _, meta2 = train_and_save_artifact(synthetic_csv, models_dir=models_dir, overwrite=True)
    assert meta2["csv_sha256"] != meta1["csv_sha256"]


def test_checksum_tamper_blocks_joblib_load(tmp_path, monkeypatch):
    models_dir = tmp_path / "tampered_models"
    models_dir.mkdir()
    (models_dir / "xgb_champion.joblib").write_bytes(b"corrupted-bytes")
    (models_dir / "b2_linear.joblib").write_bytes(b"ok-bytes")
    (models_dir / "artifact_meta.json").write_text(
        json.dumps({
            "schema_version": 2,
            "conformal_q90": 5.0,
            "features": RF_FEATURES,
            "artifact_hashes": {
                "xgb_champion.joblib": "0" * 64,
                "b2_linear.joblib": _hash_file(models_dir / "b2_linear.joblib"),
            },
        }),
        encoding="utf-8",
    )

    def fail_if_unpickled(_):
        raise AssertionError("joblib.load must not execute when SHA-256 verification fails")

    monkeypatch.setattr("materials_data_lab.model_artifact.joblib.load", fail_if_unpickled)

    with pytest.raises(ArtifactLoadError, match="Integrity check failed"):
        load_artifact(models_dir=models_dir)


def test_check_extrapolation():
    good = {
        "c_wt": 0.4, "mn_wt": 0.8, "p_wt": 0.015, "s_wt": 0.015, "si_wt": 0.25,
        "ni_wt": 0.0, "cr_wt": 0.0, "mo_wt": 0.0, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0,
        "temper_temp_c": 400.0, "temper_time_s": 3600.0,
    }
    msg = check_extrapolation(good)
    assert "✅" in msg
    assert "application input scope" in msg.lower()
    assert "metallurgical suitability" in msg.lower()
    assert "multivariate" in msg.lower()

    bad_c = good.copy()
    bad_c["c_wt"] = 1.25  # Beyond observed max 1.15
    c_warn = check_extrapolation(bad_c)
    assert "⚠" in c_warn
    assert "application input scope" in c_warn.lower()

    bad_mo = good.copy()
    bad_mo["mo_wt"] = 0.50  # Beyond observed max 0.36 (previously missed)
    assert "⚠" in check_extrapolation(bad_mo)

    high_temp_in_app_scope = good.copy()
    high_temp_in_app_scope["temper_temp_c"] = 750.0  # > observed 704.4, <= app cap 800.0
    temp_warn = check_extrapolation(high_temp_in_app_scope)
    assert "⚠" in temp_warn
    assert "outside observed dataset range" in temp_warn.lower()

    invalid_time = good.copy()
    invalid_time["temper_time_s"] = 0.0
    with pytest.raises(ValueError, match="application supported scope"):
        check_extrapolation(invalid_time)


def test_make_prediction(monkeypatch):
    class MockExplainer:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            class MockShapValues:
                @property
                def values(self):
                    arr = np.array([0.1] * 13)
                    arr[0] = 1.0
                    return arr

            return [MockShapValues()]

    import shap

    monkeypatch.setattr(shap, "TreeExplainer", MockExplainer)
    monkeypatch.setattr(shap.plots, "waterfall", lambda *args, **kwargs: None)

    class MockModel:
        def predict(self, X):
            return np.array([42.5])

    models = {"xgb": MockModel(), "b2": MockModel()}
    meta = {"features": RF_FEATURES, "conformal_q90": 5.08}

    xgb_pred, interval_text, b2_pred, warn, img_path, top_text = make_prediction(
        models, meta, 0.4, 0.8, 0.015, 0.015, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400.0, 3600.0
    )

    assert xgb_pred == "42.5 HRC"
    assert b2_pred == "42.5 HRC"
    assert "OOF empirical band" in interval_text
    assert "37.4 – 47.6 HRC" in interval_text
    assert "q90 = ±5.08 HRC" in interval_text
    assert "✅" in warn
    assert str(img_path).endswith(".png")
    assert "c_wt" in top_text
    assert "not causal" in top_text


def test_feature_order_mismatch():
    class MockModel:
        pass

    models = {"xgb": MockModel(), "b2": MockModel()}
    meta = {"features": ["wrong", "features"]}

    with pytest.raises(ValueError, match="Artifact features mismatch with code"):
        make_prediction(models, meta, 0.4, 0.8, 0.015, 0.015, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400.0, 3600.0)


def test_app_cli_missing_artifact_returns_1_without_training(tmp_path, monkeypatch, capsys):
    import importlib.util
    import sys
    from pathlib import Path

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    spec = importlib.util.spec_from_file_location("root_app", app_path)
    assert spec is not None and spec.loader is not None
    root_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root_app)

    trained = False

    def fail_if_trained(*args, **kwargs):
        nonlocal trained
        trained = True
        raise AssertionError("app.py must never trigger training")

    monkeypatch.setattr("materials_data_lab.model_artifact.train_and_save_artifact", fail_if_trained)
    monkeypatch.setattr(sys, "argv", ["app.py", "--models-dir", str(tmp_path / "no_models")])

    code = root_app.main()
    captured = capsys.readouterr()
    assert code == 1
    assert trained is False
    assert "MODEL NOT READY" in captured.err

