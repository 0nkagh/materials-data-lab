"""Tests for Phase 4 Modeling MVP."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from materials_data_lab.modeling import prepare_features, evaluate_model, RF_FEATURES

def test_evaluate_model_metrics():
    """Verify MAE, RMSE, R2 against synthetic truth."""
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    # MAE = (|0.5| + |-0.5| + |0| + |-1|) / 4 = 2/4 = 0.5
    # MSE = (0.25 + 0.25 + 0 + 1) / 4 = 1.5 / 4 = 0.375 -> RMSE = sqrt(0.375) ≈ 0.61237
    # Mean y_true = (3 - 0.5 + 2 + 7) / 4 = 11.5 / 4 = 2.875
    # SS_tot = (3-2.875)^2 + (-0.5-2.875)^2 + (2-2.875)^2 + (7-2.875)^2 = 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875
    # SS_res = 1.5
    # R2 = 1 - (1.5 / 29.1875) = 1 - 0.05139... = 0.948608...
    
    metrics = evaluate_model(y_true, y_pred)
    assert np.isclose(metrics["MAE"], 0.5)
    assert np.isclose(metrics["RMSE"], np.sqrt(0.375))
    assert np.isclose(metrics["R2"], 1 - (1.5 / 29.1875))


def test_prepare_features_phj():
    """Verify p_hj calculation and log_time in prepare_features."""
    # Create single row DF
    df = pd.DataFrame({
        "final_hrc": [50.0],
        "initial_hrc": [60.0],  # D-08: will be dropped
        "temper_time_s": [3600.0],
        "temper_temp_c": [600.0],
        "steel_type": ["TypeA"]
    })
    # Add required comp columns to avoid dropping
    for c in RF_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
            
    res = prepare_features(df)
    
    # log_time = log10(3600)
    assert np.isclose(res.at[0, "log_time"], np.log10(3600.0))
    
    # p_hj for 600C, 1h
    # T_K = 873.15. t_saat = 1. p_hj = 0.87315 * (19.5 + 0)
    expected_phj = 0.87315 * 19.5
    assert np.isclose(res.at[0, "p_hj"], expected_phj)
    
    assert "initial_hrc" not in res.columns


def test_split_determinism():
    """Verify split determinism given a random state."""
    X = np.arange(100).reshape((50, 2))
    y = np.arange(50)
    
    X1_tr, X1_te, y1_tr, y1_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    np.testing.assert_array_equal(X1_tr, X2_tr)
    np.testing.assert_array_equal(y1_te, y2_te)


def test_groupkfold_grade_disjoint_synthetic():
    """Verify that GroupKFold effectively separates grades."""
    from sklearn.model_selection import GroupKFold
    
    # Sentetik veri: 4 grade, her birinde 25 satır
    grades = ["GradeA", "GradeB", "GradeC", "GradeD"]
    rows = []
    for g in grades:
        for i in range(25):
            rows.append({"steel_type": g, "val": i})
            
    df_syn = pd.DataFrame(rows)
    X_syn = np.zeros((100, 13))
    
    gkf = GroupKFold(n_splits=2)
    folds_checked = 0
    for train_idx, test_idx in gkf.split(X_syn, groups=df_syn["steel_type"]):
        train_grades = set(df_syn.iloc[train_idx]["steel_type"])
        test_grades = set(df_syn.iloc[test_idx]["steel_type"])
        assert train_grades.intersection(test_grades) == set()
        folds_checked += 1
        
    assert folds_checked == 2


def test_empty_dataframe_fails_clean(tmp_path):
    """Empty dataframe should yield a meaningful error like DATA_NOT_AVAILABLE."""
    from materials_data_lab.clean_loader import load_clean
    import pytest
    
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(empty_csv, index=False)
    
    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        load_clean(empty_csv)
