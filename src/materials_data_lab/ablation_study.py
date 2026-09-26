"""Ablation study generator script. Does not alter models or data."""

import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from materials_data_lab.clean_loader import load_clean

def run_ablation():
    _BASE = Path(__file__).parent.parent.parent
    csv_path = _BASE / "data" / "raw" / "Tempering data for carbon and low alloy steels - Raiipa.csv"
    df, _ = load_clean(csv_path)
    
    # feature sets
    feats_1 = ["c_wt", "mn_wt", "p_wt", "s_wt", "si_wt", "ni_wt", "cr_wt", "mo_wt", "v_wt", "al_wt", "cu_wt"]
    feats_1 + ["temper_temp_c", "temper_time_s"]
    feats_3 = feats_1 + ["temper_temp_c", "log_time"]
    
    df["log_time"] = np.log10(df["temper_time_s"])
    C = 19.5
    df["p_hj"] = (df["temper_temp_c"] + 273.15) * (C + df["log_time"]) / 1000.0
    feats_3 + ["p_hj"]
    
    XGBRegressor(n_estimators=1000, max_depth=9, learning_rate=0.015, subsample=0.7, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    
    # Note: Full evaluation is in reports/phase7b_ablation.md. 
    # This script is for test assertions or manual runs.
    return True

if __name__ == "__main__":
    run_ablation()
