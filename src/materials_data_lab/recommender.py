import numpy as np
import pandas as pd

def recommend_recipe(models, meta, comp_dict, target_hrc):
    """
    Given an XGBoost champion model and a composition dictionary,
    search for the optimal tempering recipe (T, t) to reach target_hrc.
    Uses grid scan over T: [100.0, 705.0] at 0.5C resolution and t: [1800, 3600, 7200, 14400].
    Prioritizes shortest t, then lowest T.
    """
    T_GRID = np.arange(100.0, 705.5, 0.5)
    T_TIME_GRID = [1800.0, 3600.0, 7200.0, 14400.0]
    TOLERANCE = 0.5
    
    model = models["xgb"]
    features_list = meta["features"]
    q90 = meta.get("conformal_q90", 5.08)
    
    all_rows = []
    for t_s in T_TIME_GRID:
        for T_c in T_GRID:
            row = comp_dict.copy()
            row["temper_temp_c"] = T_c
            row["temper_time_s"] = t_s
            all_rows.append(row)
            
    df_grid = pd.DataFrame(all_rows)
    
    # Feature engineering for log_time
    df_grid["log_time"] = np.log10(df_grid["temper_time_s"])
    
    X = df_grid[features_list].values
    preds = model.predict(X)
    
    df_grid["pred_hrc"] = preds
    diffs = np.abs(df_grid["pred_hrc"] - target_hrc)
    
    valid_mask = diffs <= TOLERANCE
    df_valid = df_grid[valid_mask].copy()
    
    from materials_data_lab.demo import check_extrapolation
    
    # Check chemistry extrapolation with dummy T/t
    dummy_input = comp_dict.copy()
    dummy_input["temper_temp_c"] = 400.0
    dummy_input["temper_time_s"] = 3600.0
    extrap_msg = check_extrapolation(dummy_input)
    is_extrapolated = "⚠" in extrap_msg
    
    notes = ["Uyarı: Reçeteler model başlangıç noktasıdır, laboratuvar doğrulaması şarttır."]
    if is_extrapolated:
        notes.append("⚠ " + extrap_msg.replace("⚠ ", ""))
    
    if df_valid.empty:
        # Check if the target is completely out of support
        min_pred = df_grid["pred_hrc"].min()
        max_pred = df_grid["pred_hrc"].max()
        if target_hrc < min_pred or target_hrc > max_pred:
            reason = f"Hedef ({target_hrc} HRC) modelin bu bileşim için tahmin edebileceği sınırların dışında (min: {min_pred:.1f}, max: {max_pred:.1f})."
        else:
            reason = f"Hedef ({target_hrc} HRC) model sınırları içinde ancak ağaç adımlarında {TOLERANCE} HRC toleransına girecek kadar yakın bir kesişim bulunamadı."
            
        return {
            "recommendable": False,
            "T_c": None,
            "t_s": None,
            "predicted_hrc": None,
            "interval_low": None,
            "interval_high": None,
            "n_solutions": 0,
            "notes": notes + ["NOT_ACHIEVABLE: " + reason],
            "is_extrapolated": is_extrapolated,
            "high_uncertainty": is_extrapolated or 0 < 5, # Will just be true since n_solutions = 0
            "candidates": []
        }
        
    # Sort by t_s (shortest), then T_c (lowest)
    df_valid = df_valid.sort_values(["temper_time_s", "temper_temp_c"])
    
    # Cluster contiguous T solutions (0.5C step means consecutive unique T values)
    unique_t = np.sort(df_valid["temper_temp_c"].unique())
    
    windows = []
    if len(unique_t) > 0:
        cur_start = unique_t[0]
        cur_prev = unique_t[0]
        for val in unique_t[1:]:
            if val - cur_prev <= 0.51: # Allow a tiny bit of float tolerance
                cur_prev = val
            else:
                windows.append((cur_start, cur_prev))
                cur_start = val
                cur_prev = val
        windows.append((cur_start, cur_prev))
        
    candidates = []
    for w_min, w_max in windows:
        # Find the best row in this window
        mask = (df_valid["temper_temp_c"] >= w_min - 0.01) & (df_valid["temper_temp_c"] <= w_max + 0.01)
        w_df = df_valid[mask]
        
        # Already sorted by t_s then T_c, so the first is the best
        w_best = w_df.iloc[0]
        candidates.append({
            "T_c": float(w_best["temper_temp_c"]),
            "t_s": float(w_best["temper_time_s"]),
            "predicted_hrc": float(w_best["pred_hrc"]),
            "window_T_min": float(w_min),
            "window_T_max": float(w_max)
        })
        
    # Sort candidates by t_s (shortest), then T_c (lowest)
    candidates.sort(key=lambda x: (x["t_s"], x["T_c"]))
    
    # Keep top 3
    candidates = candidates[:3]
    
    best_row = df_valid.iloc[0]
    n_solutions = len(df_valid)
    
    pred_hrc = float(best_row["pred_hrc"])
    t_c = float(best_row["temper_temp_c"])
    t_s = float(best_row["temper_time_s"])
    
    high_uncertainty = is_extrapolated or n_solutions < 5
    
    return {
        "recommendable": True,
        "T_c": t_c,
        "t_s": t_s,
        "predicted_hrc": pred_hrc,
        "interval_low": pred_hrc - q90,
        "interval_high": pred_hrc + q90,
        "n_solutions": n_solutions,
        "notes": notes,
        "is_extrapolated": is_extrapolated,
        "high_uncertainty": high_uncertainty,
        "candidates": candidates
    }
