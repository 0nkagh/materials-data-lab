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
    
    notes = ["Uyarı: Reçeteler model başlangıç noktasıdır, laboratuvar doğrulaması şarttır."]
    
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
            "notes": notes + ["NOT_ACHIEVABLE: " + reason]
        }
        
    # Sort by t_s (shortest), then T_c (lowest)
    df_valid = df_valid.sort_values(["temper_time_s", "temper_temp_c"])
    
    best_row = df_valid.iloc[0]
    n_solutions = len(df_valid)
    
    pred_hrc = float(best_row["pred_hrc"])
    t_c = float(best_row["temper_temp_c"])
    t_s = float(best_row["temper_time_s"])
    
    return {
        "recommendable": True,
        "T_c": t_c,
        "t_s": t_s,
        "predicted_hrc": pred_hrc,
        "interval_low": pred_hrc - q90,
        "interval_high": pred_hrc + q90,
        "n_solutions": n_solutions,
        "notes": notes
    }
