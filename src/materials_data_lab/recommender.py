"""Candidate tempering condition recommender using grid scan over tree predictions."""

import numpy as np
import pandas as pd

from materials_data_lab.data_bounds import (
    COMP_COLS,
    check_observed_bounds,
    validate_supported_inputs,
    validate_target_hrc,
)


def recommend_recipe(models, meta, comp_dict, target_hrc):
    """Search for candidate tempering conditions (T, t) near target_hrc.

    Searches over T in [100.0, 704.0] at 0.5°C numerical scan step and discrete
    t in [1800, 3600, 7200, 14400] s, prioritizing shortest t then lowest T.

    Safety and scope policy:
    - Validates application supported input scope and finite numeric inputs upfront.
    - Refuses to generate candidate conditions (recommendable=False) when input
      chemistry falls outside 1D observed univariate bounds.
    - Observed dataset range, application supported input scope, and true metallurgical
      suitability (e.g., composition-dependent Ac1, quench conditions, initial HRC,
      section size) are distinct concepts; 1D bounds do NOT verify multivariate support.
    """
    validated_comp = validate_supported_inputs(comp_dict, require_treatment=False)
    target = validate_target_hrc(target_hrc)

    q90 = float(meta.get("conformal_q90", 5.08))
    within_chem, oob_details = check_observed_bounds(validated_comp, keys=COMP_COLS)
    is_extrapolated = not within_chem

    notes = [
        "Uyarı: Çıktılar yalnızca laboratuvar doğrulaması gerektiren aday işlem koşullarıdır; üretime hazır reçete değildir.",
        "Kapsam ayrımı: Gözlenen 1B veri aralığı, bu uygulamanın desteklediği girdi kapsamı ve gerçek metalurjik uygunluk farklı kavramlardır; 1B min–max içi bileşimlerde dahi çok değişkenli veri desteği, bileşime ve koşullara bağlı faz dönüşümü/Ac1, su verme hızı/ortamı, parça kesiti ve başlangıç sertliği doğrulanmamıştır.",
    ]

    if is_extrapolated:
        notes.append(
            "OUT_OF_OBSERVED_SCOPE: Girdi bileşimi gözlenen tek değişkenli veri sınırlarının dışında "
            f"({'; '.join(oob_details)}). Güvenlik politikası gereği aday işlem koşulu üretilmedi."
        )
        return {
            "recommendable": False,
            "status": "OUT_OF_OBSERVED_SCOPE",
            "T_c": None,
            "t_s": None,
            "predicted_hrc": None,
            "interval_low": None,
            "interval_high": None,
            "q90": q90,
            "n_solutions": 0,
            "notes": notes,
            "is_extrapolated": True,
            "high_uncertainty": True,
            "multivariate_support_verified": False,
            "candidates": [],
        }

    # Note: 0.5°C is a numerical scan step over discrete tree splits (not physical sub-degree sensitivity)
    T_GRID = np.arange(100.0, 704.5, 0.5)
    T_TIME_GRID = [1800.0, 3600.0, 7200.0, 14400.0]
    TOLERANCE = 0.5

    model = models["xgb"]
    features_list = meta["features"]

    all_rows = []
    for t_s in T_TIME_GRID:
        for T_c in T_GRID:
            row = validated_comp.copy()
            row["temper_temp_c"] = float(T_c)
            row["temper_time_s"] = float(t_s)
            all_rows.append(row)

    df_grid = pd.DataFrame(all_rows)
    df_grid["log_time"] = np.log10(df_grid["temper_time_s"])

    X = df_grid[features_list].values
    preds = model.predict(X)

    df_grid["pred_hrc"] = preds
    diffs = np.abs(df_grid["pred_hrc"] - target)

    valid_mask = diffs <= TOLERANCE
    df_valid = df_grid[valid_mask].copy()

    if df_valid.empty:
        min_pred = float(df_grid["pred_hrc"].min())
        max_pred = float(df_grid["pred_hrc"].max())
        if target < min_pred or target > max_pred:
            reason = (
                f"Hedef ({target} HRC) bu bileşim için model tahmin aralığının "
                f"([{min_pred:.1f}, {max_pred:.1f}] HRC) dışında kaldı; fiziksel olanaksızlık anlamına gelmez."
            )
        else:
            reason = (
                f"Hedef ({target} HRC) model tahmin aralığında ancak taranan 4 süre ve 0.5°C basamaklı ağaç "
                f"adımlarında nokta tahmini ±{TOLERANCE} HRC içine giren kesikli aday bulunamadı."
            )

        return {
            "recommendable": False,
            "status": "NO_CANDIDATE_FOUND",
            "T_c": None,
            "t_s": None,
            "predicted_hrc": None,
            "interval_low": None,
            "interval_high": None,
            "q90": q90,
            "n_solutions": 0,
            "notes": notes + [f"NO_CANDIDATE_FOUND (NOT_ACHIEVABLE): {reason}"],
            "is_extrapolated": False,
            "high_uncertainty": True,
            "multivariate_support_verified": False,
            "candidates": [],
        }

    # Heuristic priority: shortest t_s in grid, then lowest T_c (NOT energy/cost/embrittlement optimization)
    df_valid = df_valid.sort_values(["temper_time_s", "temper_temp_c"])

    unique_t = np.sort(df_valid["temper_temp_c"].unique())
    windows = []
    if len(unique_t) > 0:
        cur_start = unique_t[0]
        cur_prev = unique_t[0]
        for val in unique_t[1:]:
            if val - cur_prev <= 0.51:
                cur_prev = val
            else:
                windows.append((cur_start, cur_prev))
                cur_start = val
                cur_prev = val
        windows.append((cur_start, cur_prev))

    candidates = []
    for w_min, w_max in windows:
        mask = (df_valid["temper_temp_c"] >= w_min - 0.01) & (df_valid["temper_temp_c"] <= w_max + 0.01)
        w_df = df_valid[mask]
        w_best = w_df.iloc[0]
        candidates.append({
            "T_c": float(w_best["temper_temp_c"]),
            "t_s": float(w_best["temper_time_s"]),
            "predicted_hrc": float(w_best["pred_hrc"]),
            "window_T_min": float(w_min),
            "window_T_max": float(w_max),
        })

    candidates.sort(key=lambda x: (x["t_s"], x["T_c"]))
    candidates = candidates[:3]

    best_row = df_valid.iloc[0]
    n_solutions = int(len(df_valid))

    pred_hrc = float(best_row["pred_hrc"])
    t_c = float(best_row["temper_temp_c"])
    t_s = float(best_row["temper_time_s"])

    high_uncertainty = is_extrapolated or n_solutions < 5
    notes.extend([
        f"Eşleştirme notu: ±{TOLERANCE} HRC toleransı yalnız model nokta tahminine uygulanır "
        f"(0.5°C tarama adımı ağaç eşiklerinin sayısal taramasıdır); ampirik OOF tahmin bandı ±{q90:.2f} HRC'dir.",
        "Seçim politikası: [1800, 3600, 7200, 14400] s üzerinde en kısa süre sonra en düşük sıcaklık seçilir; "
        "bu kural fırın enerji/maliyet optimizasyonu veya gevrekleşme (250–575 °C) denetimi değildir.",
    ])

    return {
        "recommendable": True,
        "status": "CANDIDATE_FOUND",
        "T_c": t_c,
        "t_s": t_s,
        "predicted_hrc": pred_hrc,
        "interval_low": pred_hrc - q90,
        "interval_high": pred_hrc + q90,
        "q90": q90,
        "n_solutions": n_solutions,
        "notes": notes,
        "is_extrapolated": False,
        "high_uncertainty": high_uncertainty,
        "multivariate_support_verified": False,
        "candidates": candidates,
    }
