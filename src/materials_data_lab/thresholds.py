"""Threshold definitions, physical sanity checks, and provenance taxonomy.

All checks are strictly non-destructive. Findings are reported as flags
(SUSPECT / WARNING / INFO / E140_EXTRAPOLATION_SUSPECT), never modifying data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import pandas as pd


class Provenance:
    STANDARD_CROSSCHECKED = "STANDARD_CROSSCHECKED"
    REPORT_ONLY = "REPORT_ONLY"
    UNKNOWN = "UNKNOWN"


class FlagSeverity:
    SUSPECT = "SUSPECT"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class FlagFinding:
    row_index: int
    column: str
    value: Any
    severity: str  # SUSPECT, WARNING, INFO
    flag_name: str  # e.g., E140_EXTRAPOLATION_SUSPECT, INFO_HIGH_ALLOY, etc.
    description: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Element-specific upper bounds (wt%) [REPORT_ONLY, SAE J403/J404 style heuristics]
ELEMENT_THRESHOLDS: dict[str, dict[str, Any]] = {
    "C": {"suspect_max": 1.20, "note": "Carbon content upper limit for standard hypereutectoid carbon/alloy steels"},
    "Mn": {"suspect_max": 2.10, "note": "Manganese upper limit for high-Mn carbon steels"},
    "P": {
        "suspect_max": 0.050,
        "warning_range": (0.040, 0.050),
        "note": "Phosphorus residual/impurity limit; warning band 0.040-0.050",
    },
    "S": {
        "suspect_max": 0.060,
        "warning_range": (0.050, 0.060),
        "note": "Sulfur residual limit; warning band 0.050-0.060",
    },
    "Si": {"suspect_max": 2.50, "note": "Silicon upper limit for spring/silicon-alloyed steels"},
    "Ni": {"suspect_max": 5.00, "note": "Nickel upper limit for low-alloy steels"},
    "Cr": {"suspect_max": 2.50, "note": "Chromium upper limit for low-alloy steels"},
    "Mo": {"suspect_max": 1.00, "note": "Molybdenum upper limit for low-alloy steels"},
    "V": {"suspect_max": 0.50, "note": "Vanadium upper limit for low-alloy microalloyed steels"},
    "Al": {
        "suspect_max": 0.15,
        "note": "Aluminium > 0.15%wt indicates intentional nitriding steel (nitralloy) rather than standard deoxidation",
    },
    "Cu": {"suspect_max": 0.40, "note": "Copper upper limit for weathering/low-alloy steels"},
}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column in df matching candidates (case-insensitive substring or exact match)."""
    # 1. Exact match
    for cand in candidates:
        if cand in df.columns:
            return cand
    # 2. Case-insensitive exact match
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for cand in candidates:
        if cand.strip().lower() in lower_map:
            return lower_map[cand.strip().lower()]
    # 3. Substring match
    for cand in candidates:
        cand_lower = cand.strip().lower()
        for col in df.columns:
            if cand_lower in str(col).strip().lower():
                return col
    return None


def evaluate_thresholds(df: pd.DataFrame) -> list[FlagFinding]:
    """Evaluate all physical sanity thresholds on the DataFrame.
    
    Returns a list of FlagFinding objects.
    Does not modify df in any way.
    """
    findings: list[FlagFinding] = []

    # Identify key columns
    final_hrc_col = _find_column(df, ["Final hardness (HRC) - post tempering", "Final hardness", "final hardness"])
    init_hrc_col = _find_column(df, ["Initial hardness (HRC) - post quenching", "Initial hardness", "initial hardness"])
    temp_col = _find_column(df, ["Tempering temperature", "temperature", "sıcaklık"])
    time_col = _find_column(df, ["Tempering time", "time", "süre"])

    # Map elements
    element_col_map: dict[str, str] = {}
    for elem_sym in ELEMENT_THRESHOLDS:
        col = _find_column(df, [f"{elem_sym} (%wt)", f"{elem_sym} (%)", elem_sym])
        if col is not None:
            element_col_map[elem_sym] = col

    # Check row by row or vectorized
    for idx in df.index:
        # 1. Hardness checks (ASTM E18 / ISO 6508: 20-70 HRC)
        if final_hrc_col is not None:
            val_raw = df.at[idx, final_hrc_col]
            try:
                val = float(val_raw)
                if not pd.isna(val):
                    if val < 20.0 or val > 70.0:
                        findings.append(
                            FlagFinding(
                                row_index=int(idx),
                                column=str(final_hrc_col),
                                value=val,
                                severity=FlagSeverity.SUSPECT,
                                flag_name="HARDNESS_OUT_OF_RANGE_SUSPECT",
                                description=f"Final hardness ({val} HRC) is outside the 20-70 HRC reliable measurement range [ASTM E18 / ISO 6508]",
                                provenance=Provenance.STANDARD_CROSSCHECKED,
                            )
                        )
                    if val < 20.0:
                        # Extra specific tag for E140 extrapolation suspect
                        findings.append(
                            FlagFinding(
                                row_index=int(idx),
                                column=str(final_hrc_col),
                                value=val,
                                severity=FlagSeverity.SUSPECT,
                                flag_name="E140_EXTRAPOLATION_SUSPECT",
                                description=f"Final hardness ({val} HRC) < 20 HRC indicates likely mathematical extrapolation from <238 HV Vickers values [ASTM E140 Table 1]",
                                provenance=Provenance.STANDARD_CROSSCHECKED,
                            )
                        )
            except (ValueError, TypeError):
                pass

        if init_hrc_col is not None:
            val_raw = df.at[idx, init_hrc_col]
            try:
                val = float(val_raw)
                if not pd.isna(val):
                    if val < 20.0 or val > 70.0:
                        findings.append(
                            FlagFinding(
                                row_index=int(idx),
                                column=str(init_hrc_col),
                                value=val,
                                severity=FlagSeverity.SUSPECT,
                                flag_name="HARDNESS_OUT_OF_RANGE_SUSPECT",
                                description=f"Initial hardness ({val} HRC) is outside the 20-70 HRC reliable measurement range [ASTM E18 / ISO 6508]",
                                provenance=Provenance.STANDARD_CROSSCHECKED,
                            )
                        )
            except (ValueError, TypeError):
                pass

        # 2. Temperature checks (°C)
        if temp_col is not None:
            val_raw = df.at[idx, temp_col]
            try:
                val = float(val_raw)
                if not pd.isna(val):
                    if val <= 0.0 or val > 800.0:
                        findings.append(
                            FlagFinding(
                                row_index=int(idx),
                                column=str(temp_col),
                                value=val,
                                severity=FlagSeverity.SUSPECT,
                                flag_name="TEMP_PHYSICAL_LIMIT_SUSPECT",
                                description=f"Tempering temperature ({val} °C) is outside physical range (0 < T <= 800 °C)",
                                provenance=Provenance.STANDARD_CROSSCHECKED,
                            )
                        )
                    elif (0.0 < val < 150.0) or (100.0 <= val <= 150.0):
                        findings.append(
                            FlagFinding(
                                row_index=int(idx),
                                column=str(temp_col),
                                value=val,
                                severity=FlagSeverity.WARNING,
                                flag_name="TEMP_BELOW_WINDOW_WARNING",
                                description=f"Tempering temperature ({val} °C) is in 100-150 °C warning range (below typical practical window 150-700 °C)",
                                provenance=Provenance.STANDARD_CROSSCHECKED,
                            )
                        )
                    elif 700.0 <= val <= 800.0:
                        findings.append(
                            FlagFinding(
                                row_index=int(idx),
                                column=str(temp_col),
                                value=val,
                                severity=FlagSeverity.WARNING,
                                flag_name="TEMP_AC1_NEAR_WARNING",
                                description=f"Tempering temperature ({val} °C) is in 700-800 °C warning range (near Ac1, reaustenitization risk)",
                                provenance=Provenance.STANDARD_CROSSCHECKED,
                            )
                        )
            except (ValueError, TypeError):
                pass

        # 3. Tempering time checks (s)
        if time_col is not None:
            val_raw = df.at[idx, time_col]
            try:
                val = float(val_raw)
                if not pd.isna(val):
                    if val <= 0.0:
                        findings.append(
                            FlagFinding(
                                row_index=int(idx),
                                column=str(time_col),
                                value=val,
                                severity=FlagSeverity.SUSPECT,
                                flag_name="TIME_NON_POSITIVE_SUSPECT",
                                description=f"Tempering time ({val} s) must be strictly positive",
                                provenance=Provenance.STANDARD_CROSSCHECKED,
                            )
                        )
                    elif 0.0 < val < 300.0:
                        findings.append(
                            FlagFinding(
                                row_index=int(idx),
                                column=str(time_col),
                                value=val,
                                severity=FlagSeverity.INFO,
                                flag_name="TIME_SHORT_INDUCTION_INFO",
                                description=f"Tempering time ({val} s) < 300 s indicates short/induction tempering regime",
                                provenance=Provenance.REPORT_ONLY,
                            )
                        )
                    elif val > 14400.0:
                        findings.append(
                            FlagFinding(
                                row_index=int(idx),
                                column=str(time_col),
                                value=val,
                                severity=FlagSeverity.INFO,
                                flag_name="TIME_LONG_FURNACE_INFO",
                                description=f"Tempering time ({val} s) > 14400 s (4h) indicates long furnace tempering regime",
                                provenance=Provenance.REPORT_ONLY,
                            )
                        )
            except (ValueError, TypeError):
                pass

        # 4. Elements (%wt)
        alloy_sum = 0.0
        alloy_elements_present = 0

        for elem_sym, col_name in element_col_map.items():
            val_raw = df.at[idx, col_name]
            try:
                val = float(val_raw)
                if not pd.isna(val):
                    # Physical percentage bounds
                    if val < 0.0 or val > 100.0:
                        findings.append(
                            FlagFinding(
                                row_index=int(idx),
                                column=str(col_name),
                                value=val,
                                severity=FlagSeverity.SUSPECT,
                                flag_name="ELEMENT_PERCENT_BOUNDS_SUSPECT",
                                description=f"{elem_sym} content ({val} %wt) is outside valid percentage 0-100%wt",
                                provenance=Provenance.STANDARD_CROSSCHECKED,
                            )
                        )
                    else:
                        th_info = ELEMENT_THRESHOLDS[elem_sym]
                        sus_max = th_info.get("suspect_max")
                        if sus_max is not None and val > sus_max:
                            findings.append(
                                FlagFinding(
                                row_index=int(idx),
                                column=str(col_name),
                                value=val,
                                severity=FlagSeverity.SUSPECT,
                                flag_name=f"{elem_sym}_UPPER_BOUND_SUSPECT",
                                description=f"{elem_sym} ({val} %wt) exceeds upper threshold {sus_max} %wt. {th_info.get('note', '')}",
                                provenance=Provenance.REPORT_ONLY,
                            )
                        )
                        elif "warning_range" in th_info:
                            w_min, w_max = th_info["warning_range"]
                            if w_min <= val <= w_max:
                                findings.append(
                                    FlagFinding(
                                        row_index=int(idx),
                                        column=str(col_name),
                                        value=val,
                                        severity=FlagSeverity.WARNING,
                                        flag_name=f"{elem_sym}_WARNING_BAND",
                                        description=f"{elem_sym} ({val} %wt) falls in warning band {w_min}-{w_max} %wt. {th_info.get('note', '')}",
                                        provenance=Provenance.REPORT_ONLY,
                                    )
                                )

                    # Sum for Ni, Cr, Mo, V, Si
                    if elem_sym in ("Ni", "Cr", "Mo", "V", "Si"):
                        alloy_sum += val
                        alloy_elements_present += 1
            except (ValueError, TypeError):
                pass

        # 5. Row-based sum(Ni+Cr+Mo+V+Si) > 8.0 -> INFO_HIGH_ALLOY
        if alloy_elements_present > 0 and alloy_sum > 8.0:
            findings.append(
                FlagFinding(
                    row_index=int(idx),
                    column="sum(Ni+Cr+Mo+V+Si)",
                    value=round(alloy_sum, 4),
                    severity=FlagSeverity.INFO,
                    flag_name="INFO_HIGH_ALLOY",
                    description=f"Combined alloy sum Ni+Cr+Mo+V+Si ({round(alloy_sum, 4)} %wt) > 8.0 %wt indicates high-alloy kinetics regime",
                    provenance=Provenance.REPORT_ONLY,
                )
            )

    return findings


def summarize_flags(findings: list[FlagFinding]) -> dict[str, Any]:
    """Summarize list of findings into separate counts."""
    counts = {
        "SUSPECT": 0,
        "WARNING": 0,
        "INFO": 0,
        "E140_EXTRAPOLATION_SUSPECT": 0,
        "INFO_HIGH_ALLOY": 0,
    }
    by_column: dict[str, dict[str, int]] = {}
    by_rule: dict[str, int] = {}
    flagged_rows: set[int] = set()

    for f in findings:
        flagged_rows.add(f.row_index)
        by_rule[f.flag_name] = by_rule.get(f.flag_name, 0) + 1

        if f.severity in counts:
            counts[f.severity] += 1
        if f.flag_name == "E140_EXTRAPOLATION_SUSPECT":
            counts["E140_EXTRAPOLATION_SUSPECT"] += 1
        if f.flag_name == "INFO_HIGH_ALLOY":
            counts["INFO_HIGH_ALLOY"] += 1

        col_dict = by_column.setdefault(f.column, {"SUSPECT": 0, "WARNING": 0, "INFO": 0})
        if f.severity in col_dict:
            col_dict[f.severity] += 1

    return {
        "counts": counts,
        "total_findings": len(findings),
        "total_flagged_rows": len(flagged_rows),
        "by_column": by_column,
        "by_rule": by_rule,
    }
