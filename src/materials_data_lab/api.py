"""FastAPI REST service for Materials Data Lab (load-only artifacts, strict input bounds)."""

import logging
import math
import re
import warnings
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from materials_data_lab import __version__
from materials_data_lab.demo import make_prediction
from materials_data_lab.model_artifact import ArtifactError, load_artifact

logger = logging.getLogger(__name__)

_INTERVAL_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*[–-]\s*([-+]?\d+(?:\.\d+)?)\s*HRC\s*\(q90\s*=\s*±([-+]?\d+(?:\.\d+)?)\s*HRC\)"
)

app = FastAPI(
    title="Materials Data Lab API",
    description=(
        "Programmatic access to the Tempering Hardness research prototype. "
        "Does NOT trigger automatic training at runtime."
    ),
    version=__version__,
)


def _sanitize_for_json(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, BaseException):
        return str(obj)
    return obj


@app.exception_handler(RequestValidationError)
async def _request_validation_exception_handler(_request: Request, exc: RequestValidationError):
    safe_errors = jsonable_encoder(_sanitize_for_json(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": safe_errors})


def _check_finite(v: float) -> float:
    if isinstance(v, bool) or not math.isfinite(v):
        raise ValueError("Value must be a finite real number (NaN/Infinity/bool disallowed).")
    return float(v)


class _CompositionBase(BaseModel):
    c_wt: float = Field(..., ge=0.0, le=100.0)
    mn_wt: float = Field(..., ge=0.0, le=100.0)
    p_wt: float = Field(..., ge=0.0, le=100.0)
    s_wt: float = Field(..., ge=0.0, le=100.0)
    si_wt: float = Field(..., ge=0.0, le=100.0)
    ni_wt: float = Field(..., ge=0.0, le=100.0)
    cr_wt: float = Field(..., ge=0.0, le=100.0)
    mo_wt: float = Field(..., ge=0.0, le=100.0)
    v_wt: float = Field(..., ge=0.0, le=100.0)
    al_wt: float = Field(..., ge=0.0, le=100.0)
    cu_wt: float = Field(..., ge=0.0, le=100.0)

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def reject_bool_and_non_finite(cls, v):
        if isinstance(v, bool):
            raise ValueError("Boolean is not a valid numeric measurement.")
        if isinstance(v, str):
            raise ValueError("String is not accepted; provide a finite numeric value.")
        if isinstance(v, (int, float)) and not math.isfinite(float(v)):
            raise ValueError("NaN and Infinity are disallowed.")
        return v

    @field_validator("*", mode="after")
    @classmethod
    def ensure_finite_float(cls, v):
        if not math.isfinite(float(v)):
            raise ValueError("NaN and Infinity are disallowed.")
        return float(v)

    @model_validator(mode="after")
    def validate_composition_sum(self):
        total = (
            self.c_wt + self.mn_wt + self.p_wt + self.s_wt + self.si_wt
            + self.ni_wt + self.cr_wt + self.mo_wt + self.v_wt + self.al_wt + self.cu_wt
        )
        _check_finite(total)
        if total > 100.0 + 1e-9:
            raise ValueError(f"Total alloying element percentage ({total:.4f} %wt) cannot exceed 100 %wt.")
        return self


class CompositionInput(_CompositionBase):
    """Input schema enforcing this application's supported input scope (not universal metallurgical bounds)."""

    temper_temp_c: float = Field(
        ...,
        gt=0.0,
        le=800.0,
        description=(
            "Application-supported temperature input scope (0 < T <= 800 °C). "
            "Distinct from observed dataset bounds (100.0–704.4 °C) and composition-dependent "
            "metallurgical suitability (Ac1 and transformation depend on chemistry and conditions)."
        ),
    )
    temper_time_s: float = Field(..., gt=0.0)


class RecommendInput(_CompositionBase):
    target_hrc: float = Field(..., ge=0.0, le=100.0)


# Global variables to hold loaded models and metadata (never auto-trained)
_MODELS = None
_META = None


def get_models():
    """Load ready model artifacts from disk without touching raw CSV or training."""
    global _MODELS, _META
    if _MODELS is None or _META is None:
        _base = Path(__file__).parent.parent.parent
        models_dir = _base / "models"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _MODELS, _META = load_artifact(models_dir=models_dir)
            for w in caught:
                logger.warning("Artifact load warning (%s): %s", type(w.message).__name__, w.message)

    return _MODELS, _META


@app.get("/health")
def health_check():
    try:
        _, meta = get_models()
    except ArtifactError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "model_not_ready",
                "model_version": __version__,
                "error": type(exc).__name__,
                "detail": str(exc),
            },
        )
    resp = {
        "status": "ok",
        "champion": meta.get("champion", "unknown"),
        "model_version": __version__,
    }
    if meta.get("version_mismatches"):
        resp["version_warnings"] = meta["version_mismatches"]
    return resp


@app.post("/predict")
def predict(comp: CompositionInput):
    try:
        models, meta = get_models()
    except ArtifactError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "model_not_ready",
                "error": type(exc).__name__,
                "message": str(exc),
            },
        ) from exc

    try:
        xgb_str, interval_str, b2_str, warning_text, _, _ = make_prediction(
            models=models,
            meta=meta,
            c_wt=comp.c_wt,
            mn_wt=comp.mn_wt,
            p_wt=comp.p_wt,
            s_wt=comp.s_wt,
            si_wt=comp.si_wt,
            ni_wt=comp.ni_wt,
            cr_wt=comp.cr_wt,
            mo_wt=comp.mo_wt,
            v_wt=comp.v_wt,
            al_wt=comp.al_wt,
            cu_wt=comp.cu_wt,
            temper_temp_c=comp.temper_temp_c,
            temper_time_s=comp.temper_time_s,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    champion_hrc = float(xgb_str.replace(" HRC", ""))
    physics_hrc = float(b2_str.replace(" HRC", ""))

    match = _INTERVAL_RE.search(interval_str)
    if not match:
        raise HTTPException(status_code=500, detail=f"Unexpected prediction interval format: {interval_str!r}")
    interval_low = float(match.group(1))
    interval_high = float(match.group(2))
    q90 = float(match.group(3))

    warnings_list = []
    if "⚠" in warning_text:
        warnings_list.append(warning_text.replace("⚠ ", ""))

    within_bounds = len(warnings_list) == 0
    is_extrapolated = not within_bounds

    return {
        "champion_hrc": champion_hrc,
        "physics_hrc": physics_hrc,
        "interval_low": interval_low,
        "interval_high": interval_high,
        "q90": q90,
        "within_bounds": within_bounds,
        "is_extrapolated": is_extrapolated,
        "warnings": warnings_list,
        "multivariate_support_verified": False,
        "scope_note": (
            "Observed 1D dataset bounds (e.g., T 100.0–704.4 °C), application-supported input scope "
            "(0 < T <= 800 °C; Ac1 depends on composition and conditions), and true metallurgical "
            "suitability are distinct; joint multivariate support and external/per-family coverage "
            "guarantees are not established."
        ),
    }


@app.post("/recommend")
def recommend(comp: RecommendInput):
    try:
        models, meta = get_models()
    except ArtifactError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "model_not_ready",
                "error": type(exc).__name__,
                "message": str(exc),
            },
        ) from exc

    comp_dict = comp.model_dump(exclude={"target_hrc"})
    target_hrc = comp.target_hrc

    from materials_data_lab.recommender import recommend_recipe

    try:
        result = recommend_recipe(models, meta, comp_dict, target_hrc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result
