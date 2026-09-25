import numpy as np
import pandas as pd
from materials_data_lab.api import get_models

steels = {
    "4140": {"c_wt": 0.40, "mn_wt": 0.85, "p_wt": 0.015, "s_wt": 0.015, "si_wt": 0.25, "ni_wt": 0.0, "cr_wt": 1.0, "mo_wt": 0.20, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0}
}
comp = steels["4140"]
target = 45.0

models, meta = get_models()

# Recalculate exactly as in recommender.py to get the 99 solutions
T_GRID = np.arange(100.0, 705.5, 0.5)
T_TIME_GRID = [1800.0, 3600.0, 7200.0, 14400.0]
features_list = meta["features"]

all_rows = []
for t_s in T_TIME_GRID:
    for T_c in T_GRID:
        row = comp.copy()
        row["temper_temp_c"] = T_c
        row["temper_time_s"] = t_s
        all_rows.append(row)

df_grid = pd.DataFrame(all_rows)
df_grid["log_time"] = np.log10(df_grid["temper_time_s"])
X = df_grid[features_list].values
preds = models["xgb"].predict(X)
df_grid["pred_hrc"] = preds
df_grid["abs_err"] = (df_grid["pred_hrc"] - target).abs()
valid_mask = df_grid["abs_err"] <= 0.5
df_valid = df_grid[valid_mask].copy()

t_vals = df_valid["temper_temp_c"].values
print(f"Total solutions: {len(df_valid)}")
print(f"Min T: {t_vals.min()}")
print(f"Max T: {t_vals.max()}")

# Find contiguous regions
t_vals = np.sort(t_vals)
diffs = np.diff(t_vals)
jumps = np.where(diffs > 0.5)[0]

start = t_vals[0]
for idx in jumps:
    end = t_vals[idx]
    print(f"Region: {start} - {end}")
    start = t_vals[idx+1]
print(f"Region: {start} - {t_vals[-1]}")
