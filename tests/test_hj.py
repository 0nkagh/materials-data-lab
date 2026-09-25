import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from materials_data_lab.hj_variants import get_b2_classic_feature, get_composite_c_design_matrix

def test_phj_derivation():
    # Known T/t values
    # T_C = 226.85 => T_K = 500
    # t_s = 36000 => t_h = 10
    # p_hj = (500/1000) * (19.5 + log10(10)) = 0.5 * (19.5 + 1) = 10.25
    df = pd.DataFrame({
        "temper_temp_c": [226.85],
        "temper_time_s": [36000]
    })
    val = get_b2_classic_feature(df)[0]
    np.testing.assert_allclose(val, 10.25)

def test_design_matrix_size():
    df = pd.DataFrame({
        "temper_temp_c": [226.85, 300],
        "temper_time_s": [36000, 7200],
        "c_wt": [0.4, 0.5],
        "mn_wt": [0.8, 0.7],
        "p_wt": [0.01, 0.02],
        "s_wt": [0.01, 0.02],
        "si_wt": [0.2, 0.3],
        "ni_wt": [0.0, 1.0],
        "cr_wt": [1.0, 0.5],
        "mo_wt": [0.2, 0.1],
        "v_wt": [0.0, 0.0],
        "al_wt": [0.0, 0.0],
        "cu_wt": [0.0, 0.0],
    })
    
    X, elements = get_composite_c_design_matrix(df)
    assert X.shape == (2, 13)
    assert len(elements) == 11

def test_composite_fit_determinism():
    df = pd.DataFrame({
        "temper_temp_c": [226.85, 300, 400],
        "temper_time_s": [36000, 7200, 3600],
        "c_wt": [0.4, 0.5, 0.3],
        "mn_wt": [0.8, 0.7, 0.6],
        "p_wt": [0.01, 0.02, 0.01],
        "s_wt": [0.01, 0.02, 0.01],
        "si_wt": [0.2, 0.3, 0.2],
        "ni_wt": [0.0, 1.0, 0.5],
        "cr_wt": [1.0, 0.5, 1.0],
        "mo_wt": [0.2, 0.1, 0.0],
        "v_wt": [0.0, 0.0, 0.1],
        "al_wt": [0.0, 0.0, 0.0],
        "cu_wt": [0.0, 0.0, 0.1],
        "final_hrc": [50, 45, 40]
    })
    X, _ = get_composite_c_design_matrix(df)
    y = df["final_hrc"].values
    
    model1 = LinearRegression().fit(X, y)
    model2 = LinearRegression().fit(X, y)
    
    np.testing.assert_array_equal(model1.coef_, model2.coef_)

def test_leakage_protection():
    # Create 10 samples with 3 groups
    df = pd.DataFrame({
        "temper_temp_c": np.random.rand(10) * 500,
        "temper_time_s": np.random.rand(10) * 36000,
        "c_wt": np.random.rand(10),
        "mn_wt": np.random.rand(10),
        "p_wt": np.random.rand(10),
        "s_wt": np.random.rand(10),
        "si_wt": np.random.rand(10),
        "ni_wt": np.random.rand(10),
        "cr_wt": np.random.rand(10),
        "mo_wt": np.random.rand(10),
        "v_wt": np.random.rand(10),
        "al_wt": np.random.rand(10),
        "cu_wt": np.random.rand(10),
        "final_hrc": np.random.rand(10) * 60,
        "steel_type": [0,0,0,1,1,1,2,2,2,2]
    })
    
    X, _ = get_composite_c_design_matrix(df)
    y = df["final_hrc"].values
    groups = df["steel_type"].values
    
    gkf = GroupKFold(n_splits=3)
    coefs = []
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        m = LinearRegression().fit(X[train_idx], y[train_idx])
        coefs.append(m.coef_)
        
    # Check that coefficients are different across folds
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(coefs[0], coefs[1])
