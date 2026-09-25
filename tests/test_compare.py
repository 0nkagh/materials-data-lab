import pytest
import numpy as np
import pandas as pd
from materials_data_lab.compare_models import XGB_SEARCH_SPACE, perform_xgb_tuning, generate_shap_plots

def test_xgb_search_space_size():
    total = 1
    for k, v in XGB_SEARCH_SPACE.items():
        total *= len(v)
    # 3 * 4 * 3 * 3 * 3 * 3 = 972
    assert total == 972

def test_xgb_tuning_deterministic():
    # Use hist method with limited data for fast test
    X = np.random.RandomState(42).rand(50, 5)
    y = np.random.RandomState(42).rand(50)
    groups = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2] * 5)
    
    m1, meta1 = perform_xgb_tuning(X, y, groups, n_iter=2)
    m2, meta2 = perform_xgb_tuning(X, y, groups, n_iter=2)
    
    assert meta1["best_params"] == meta2["best_params"]
    assert np.isclose(meta1["best_mae"], meta2["best_mae"])
    
def test_generate_shap_plots(tmp_path):
    pytest.importorskip("shap")
    import xgboost as xgb
    
    # Small synthetic dataset
    X = np.random.RandomState(42).rand(20, 5)
    y = np.random.RandomState(42).rand(20)
    cols = ["c_wt", "mn_wt", "si_wt", "temper_temp_c", "log_time"]
    X_df = pd.DataFrame(X, columns=cols)
    
    model = xgb.XGBRegressor(random_state=42, n_estimators=5, max_depth=3)
    model.fit(X, y)
    
    outdir = tmp_path / "figures"
    generate_shap_plots(model, X_df, outdir)
    
    assert (outdir / "shap_summary_bar.png").exists()
    assert (outdir / "shap_beeswarm.png").exists()
    assert (outdir / "shap_dependence_temp.png").exists()
