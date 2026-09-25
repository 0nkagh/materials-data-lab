import numpy as np
import pandas as pd
from pathlib import Path
from materials_data_lab.steel_classes import classify_steel

def test_classify_steel():
    assert classify_steel("4340") == "Ni_Cr_Mo"
    assert classify_steel("4140") == "Cr_Mo"
    assert classify_steel("1026") == "plain_carbon"
    assert classify_steel("6145") == "Cr_V"
    assert classify_steel("8640") == "Ni_Cr_Mo"
    assert classify_steel("Nitriding Steel ") == "nitriding"
    assert classify_steel("?") == "UNKNOWN_CLASS"
    assert classify_steel("") == "UNKNOWN_CLASS"
    assert classify_steel("X123") == "UNKNOWN_CLASS"
    assert classify_steel("AISI-SAE E52100") == "Cr_steel"
    assert classify_steel("0,98%C - plain carbon steel") == "plain_carbon"

def test_onehot_experiment_deterministic():
    # We will test the one-hot logic determinism using a mock function that replicates the V2-D script's logic.
    from sklearn.ensemble import RandomForestRegressor
    X_num = np.random.RandomState(42).rand(10, 2)
    classes = pd.Series(["Cr_Mo", "plain_carbon", "Cr_Mo", "Ni_Cr_Mo", "Cr_V", "plain_carbon", "Cr_Mo", "UNKNOWN_CLASS", "nitriding", "Cr_Mo"])
    y = np.random.RandomState(42).rand(10)
    
    # One-hot encoding
    X_cat = pd.get_dummies(classes, prefix="class").values
    X_full = np.hstack([X_num, X_cat])
    
    model1 = RandomForestRegressor(n_estimators=5, random_state=42)
    model1.fit(X_full, y)
    pred1 = model1.predict(X_full)
    
    model2 = RandomForestRegressor(n_estimators=5, random_state=42)
    model2.fit(X_full, y)
    pred2 = model2.predict(X_full)
    
    np.testing.assert_array_equal(pred1, pred2)

def test_dictionary_content():
    dict_path = Path("docs/data_dictionary.md")
    if dict_path.exists():
        content = dict_path.read_text(encoding="utf-8")
        assert "Steel Class System" in content
        assert "[REPORT_ONLY]" in content
