"""Clean data loader for Materials Data Lab (Phase 3).

Applies approved cleaning decisions non-destructively to produce a derived DataFrame.
"""

from pathlib import Path
from typing import Any
import pandas as pd

from materials_data_lab.thresholds import evaluate_thresholds

SLUG_MAPPING = {
    "Source": "source",
    "Steel type": "steel_type",
    "Initial hardness (HRC) - post quenching": "initial_hrc",
    "Tempering time (s)": "temper_time_s",
    "Tempering temperature (ºC)": "temper_temp_c",
    "C (%wt)": "c_wt",
    "Mn (%wt)": "mn_wt",
    "P (%wt)": "p_wt",
    "S (%wt)": "s_wt",
    "Si (%wt)": "si_wt",
    "Ni (%wt)": "ni_wt",
    "Cr (%wt)": "cr_wt",
    "Mo (%wt)": "mo_wt",
    "V (%wt)": "v_wt",
    "Al (%wt)": "al_wt",
    "Cu (%wt)": "cu_wt",
    "Final hardness (HRC) - post tempering": "final_hrc"
}

def load_clean(csv_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load CSV and apply approved Phase 2 cleaning decisions."""
    manifest: dict[str, Any] = {
        "applied_decisions": ["D-01", "NK-1", "D-05", "D-07", "flag_column", "initial_hrc_missing"],
        "counts": {}
    }
    
    # 1. D-01: read_csv with na_values=["?"]
    try:
        df = pd.read_csv(csv_path, keep_default_na=True, na_values=["?"], encoding="utf-8")
        if df.empty or len(df.columns) == 0:
            raise ValueError("DATA_NOT_AVAILABLE: The provided dataset is empty.")
    except pd.errors.EmptyDataError:
        raise ValueError("DATA_NOT_AVAILABLE: The provided dataset is empty.")
    
    # Find column names that might differ slightly in different environments/files, but fallback to dictionary keys
    # To be safe, we will just rely on the SLUG_MAPPING matching the exact file we have.
    # But wait, 'Steel type' is sometimes 'type' in Kaggle but in our CSV it is 'Steel type' as per Phase 1.
    # The dictionary matches Phase 1 EXACT column names.
    
    # 2. NK-1: str.strip() on Source and Steel type
    strip_affected = 0
    for col in ["Source", "Steel type"]:
        if col in df.columns:
            original = df[col].astype(str).copy()
            stripped = original.str.strip()
            affected = (original != stripped).sum()
            strip_affected += affected
            df[col] = stripped
    manifest["counts"]["nk1_strip_affected_cells"] = int(strip_affected)
    
    # 3. D-05: Steel type NaN fill with "UNKNOWN_GRADE"
    unknown_grade_count = 0
    if "Steel type" in df.columns:
        missing_mask = df["Steel type"].isna() | (df["Steel type"] == "")
        unknown_grade_count = missing_mask.sum()
        df.loc[missing_mask, "Steel type"] = "UNKNOWN_GRADE"
    manifest["counts"]["d05_unknown_grade_rows"] = int(unknown_grade_count)
    
    # 4. initial_hrc_missing bool column
    missing_hrc_count = 0
    if "Initial hardness (HRC) - post quenching" in df.columns:
        df["initial_hrc_missing"] = df["Initial hardness (HRC) - post quenching"].isna()
        missing_hrc_count = df["initial_hrc_missing"].sum()
    manifest["counts"]["initial_hrc_missing_count"] = int(missing_hrc_count)
    
    # 5. Flag column: evaluate_thresholds
    findings = evaluate_thresholds(df)
    flag_dict: dict[int, list[str]] = {}
    for f in findings:
        idx = f.row_index
        flag_dict.setdefault(idx, [])
        flag_dict[idx].append(f.flag_name)
        
    df["flags"] = df.index.map(lambda i: "|".join(flag_dict.get(i, [])))
    
    # 6. D-07: rename columns according to SLUG_MAPPING
    # Also keep the new columns "initial_hrc_missing" and "flags"
    df = df.rename(columns=SLUG_MAPPING)
    
    return df, manifest
