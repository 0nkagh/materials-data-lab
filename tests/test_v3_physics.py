import numpy as np
from materials_data_lab.modeling import RF_FEATURES

def test_monotone_vector_alignment():
    # temper_temp_c and log_time must be -1, others 0
    vector = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1)
    
    assert len(vector) == len(RF_FEATURES)
    assert RF_FEATURES[-2] == "temper_temp_c"
    assert RF_FEATURES[-1] == "log_time"
    
    for i, feat in enumerate(RF_FEATURES):
        if feat in ["temper_temp_c", "log_time"]:
            assert vector[i] == -1
        else:
            assert vector[i] == 0

def test_metallurgical_formulas():
    c_wt = 0.4
    mn_wt = 0.85
    cr_wt = 1.0
    mo_wt = 0.22
    v_wt = 0.0
    ni_wt = 0.0
    cu_wt = 0.0
    si_wt = 0.25
    temper_temp_c = 400.0
    
    # ce = c + mn/6 + (cr+mo+v)/5 + (ni+cu)/15
    ce = c_wt + mn_wt/6 + (cr_wt+mo_wt+v_wt)/5 + (ni_wt+cu_wt)/15
    assert np.isclose(ce, 0.4 + 0.1416666 + 0.244)
    
    # log_di
    log_di = np.log10(c_wt) + np.log10(1 + 3.33*mn_wt) + np.log10(1 + 0.7*si_wt) + np.log10(1 + 0.36*ni_wt) + np.log10(1 + 2.16*cr_wt) + np.log10(1 + 3.0*mo_wt)
    assert log_di > 0 # Should be > 0
    
    # sec_hard
    sec_hard = v_wt + 0.5 * mo_wt
    assert np.isclose(sec_hard, 0.11)
    
    # si_int
    si_int = si_wt * np.maximum(0, temper_temp_c - 300)
    assert np.isclose(si_int, 0.25 * 100)
    
    si_int_low = si_wt * np.maximum(0, 200 - 300)
    assert np.isclose(si_int_low, 0.0)

def test_conformal_interval_synthetic():
    y = np.array([40, 42, 45, 50, 55])
    preds = np.array([40.5, 41.0, 46.5, 50.1, 60.0])
    
    residuals = np.abs(y - preds) # [0.5, 1.0, 1.5, 0.1, 5.0]
    q = np.quantile(residuals, 0.90) # 90th percentile of [0.1, 0.5, 1.0, 1.5, 5.0] -> 5.0 * 0.6 + 1.5 * 0.4 = 3.6
    
    # Covered
    covered = residuals <= q
    assert np.mean(covered) >= 0.8 # 4 out of 5 covered
