import numpy as np
from materials_data_lab.tuning import SEARCH_SPACE, perform_tuning, generate_learning_curve
from sklearn.ensemble import RandomForestRegressor

def test_search_space_size():
    # Calculate total combinations
    total = 1
    for k, v in SEARCH_SPACE.items():
        total *= len(v)
    assert total == 3 * 4 * 3 * 3 * 3 # 324

def test_tuning_pipeline_deterministic():
    X = np.random.RandomState(42).rand(50, 5)
    y = np.random.RandomState(42).rand(50)
    groups = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2] * 5)
    
    m1, meta1, _ = perform_tuning(X, y, groups, n_iter=2)
    m2, meta2, _ = perform_tuning(X, y, groups, n_iter=2)
    
    assert meta1["best_params"] == meta2["best_params"]
    assert np.isclose(meta1["best_mae"], meta2["best_mae"])

def test_generate_learning_curve(tmp_path):
    X = np.random.RandomState(42).rand(50, 5)
    y = np.random.RandomState(42).rand(50)
    groups = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2] * 5)
    
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    out_path = tmp_path / "learning_curve.png"
    
    meta = generate_learning_curve(rf, X, y, groups, out_path)
    
    assert out_path.exists()
    assert len(meta["train_sizes"]) == 8
    assert len(meta["train_mae"]) == 8
    assert len(meta["test_mae"]) == 8
