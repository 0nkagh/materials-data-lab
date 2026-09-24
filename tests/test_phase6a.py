"""Tests for Phase 6a."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from materials_data_lab.phase6a import get_n_splits

def test_subset_filter_count():
    """Verify subset filter counts non-nulls correctly."""
    df = pd.DataFrame({
        "initial_hrc": [10.0, np.nan, 20.0, None, 30.0],
        "other": [1, 2, 3, 4, 5]
    })
    sub_df = df.dropna(subset=["initial_hrc"])
    assert len(sub_df) == 3
    assert sub_df["initial_hrc"].tolist() == [10.0, 20.0, 30.0]

def test_n_splits_protection():
    """Verify n_splits protection logic works for n_groups < 5."""
    groups_many = np.array([1, 2, 3, 4, 5, 6, 7])
    assert get_n_splits(groups_many, desired=5) == 5
    
    groups_few = np.array([1, 1, 2, 2, 3])
    assert get_n_splits(groups_few, desired=5) == 3
    
    groups_one = np.array([1, 1, 1])
    assert get_n_splits(groups_one, desired=5) == 1

def test_permutation_importance_deterministic():
    """Verify permutation importance runs deterministically."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])
    
    m1 = RandomForestRegressor(random_state=42)
    m1.fit(X, y)
    pi1 = permutation_importance(m1, X, y, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error")
    
    m2 = RandomForestRegressor(random_state=42)
    m2.fit(X, y)
    pi2 = permutation_importance(m2, X, y, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error")
    
    np.testing.assert_array_equal(pi1.importances_mean, pi2.importances_mean)
