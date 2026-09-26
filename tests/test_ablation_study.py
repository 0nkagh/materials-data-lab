import numpy as np
import pandas as pd

def test_ablation_smoke():
    # Smoke test for ablation steps to ensure models run
    # Create synthetic tiny dataset with all required features
    from xgboost import XGBRegressor
    
    np.random.seed(42)
    n_samples = 20
    
    # 11 chem features + temper_temp_c, temper_time_s
    cols = ["c_wt", "mn_wt", "p_wt", "s_wt", "si_wt", "ni_wt", "cr_wt", "mo_wt", "v_wt", "al_wt", "cu_wt", "temper_temp_c", "temper_time_s"]
    data = {}
    for c in cols:
        data[c] = np.random.uniform(0.01, 1.0, size=n_samples)
    
    # Add target and group
    data["final_hrc"] = np.random.uniform(20, 60, size=n_samples)
    data["steel_type"] = ["TypeA"]*10 + ["TypeB"]*10
    
    df = pd.DataFrame(data)
    df["log_time"] = np.log10(df["temper_time_s"])
    
    C = 19.5
    df["p_hj"] = (df["temper_temp_c"] + 273.15) * (C + df["log_time"]) / 1000.0
    
    # XGB
    xgb = XGBRegressor(n_estimators=5, max_depth=3, random_state=42)
    
    # 1. Yalnız 11 kimya
    feats_1 = ["c_wt", "mn_wt", "p_wt", "s_wt", "si_wt", "ni_wt", "cr_wt", "mo_wt", "v_wt", "al_wt", "cu_wt"]
    xgb.fit(df[feats_1], df["final_hrc"])
    p1 = xgb.predict(df[feats_1])
    assert len(p1) == n_samples
    
    # 2. +T, t
    feats_2 = feats_1 + ["temper_temp_c", "temper_time_s"]
    xgb.fit(df[feats_2], df["final_hrc"])
    p2 = xgb.predict(df[feats_2])
    assert len(p2) == n_samples
    
    # 3. +log_time
    feats_3 = feats_1 + ["temper_temp_c", "log_time"]
    xgb.fit(df[feats_3], df["final_hrc"])
    p3 = xgb.predict(df[feats_3])
    assert len(p3) == n_samples
    
    # 4. +Hp sabit-C
    feats_4 = feats_3 + ["p_hj"]
    xgb.fit(df[feats_4], df["final_hrc"])
    p4 = xgb.predict(df[feats_4])
    assert len(p4) == n_samples
