from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
import warnings

from materials_data_lab.model_artifact import build_or_load_artifact
from materials_data_lab.demo import make_prediction
from materials_data_lab import __version__

# Initialize FastAPI app
app = FastAPI(
    title="Materials Data Lab API",
    description="Programmatic access to the Tempering Hardness Predictor",
    version=__version__,
)

class CompositionInput(BaseModel):
    c_wt: float = Field(..., ge=0.0)
    mn_wt: float = Field(..., ge=0.0)
    p_wt: float = Field(..., ge=0.0)
    s_wt: float = Field(..., ge=0.0)
    si_wt: float = Field(..., ge=0.0)
    ni_wt: float = Field(..., ge=0.0)
    cr_wt: float = Field(..., ge=0.0)
    mo_wt: float = Field(..., ge=0.0)
    v_wt: float = Field(..., ge=0.0)
    al_wt: float = Field(..., ge=0.0)
    cu_wt: float = Field(..., ge=0.0)
    temper_temp_c: float = Field(..., ge=0.0)
    temper_time_s: float = Field(..., gt=0.0)

    model_config = ConfigDict(extra='forbid')

# Global variables to hold models and meta
_MODELS = None
_META = None

def get_models():
    global _MODELS, _META
    if _MODELS is None or _META is None:
        # Load the artifact
        # Path resolution similar to app.py
        _SRC = Path(__file__).parent
        _BASE = _SRC.parent.parent
        csv_path = _BASE / "data" / "raw" / "Tempering data for carbon and low alloy steels - Raiipa.csv"
        models_dir = _BASE / "models"
        
        # Suppress warnings that might occur during load
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _MODELS, _META = build_or_load_artifact(csv_path, models_dir=models_dir)
            
    return _MODELS, _META

@app.get("/health")
def health_check():
    _, meta = get_models()
    return {
        "status": "ok",
        "champion": meta.get("champion", "unknown"),
        "model_version": __version__
    }

@app.post("/predict")
def predict(comp: CompositionInput):
    models, meta = get_models()
    
    # Call make_prediction from demo.py
    # make_prediction returns:
    # f"{xgb_pred:.1f} HRC", interval_text, f"{b2_pred:.1f} HRC", warning_text, tmp_path, top_text
    # We parse the strings back to floats to return structured JSON
    
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
        temper_time_s=comp.temper_time_s
    )
    
    champion_hrc = float(xgb_str.replace(" HRC", ""))
    physics_hrc = float(b2_str.replace(" HRC", ""))
    
    # Parse interval string: "90% interval: 37.4 – 47.6 HRC (q90 = ±5.08 HRC)"
    interval_parts = interval_str.split(" interval: ")[1].split(" HRC (q90 = ±")
    bounds = interval_parts[0].split(" – ")
    interval_low = float(bounds[0])
    interval_high = float(bounds[1])
    q90 = float(interval_parts[1].replace(" HRC)", ""))
    
    warnings_list = []
    if "⚠" in warning_text:
        warnings_list.append(warning_text.replace("⚠ ", ""))
        
    within_bounds = len(warnings_list) == 0
    
    return {
        "champion_hrc": champion_hrc,
        "physics_hrc": physics_hrc,
        "interval_low": interval_low,
        "interval_high": interval_high,
        "q90": q90,
        "within_bounds": within_bounds,
        "warnings": warnings_list
    }

class RecommendInput(BaseModel):
    c_wt: float = Field(..., ge=0.0)
    mn_wt: float = Field(..., ge=0.0)
    p_wt: float = Field(..., ge=0.0)
    s_wt: float = Field(..., ge=0.0)
    si_wt: float = Field(..., ge=0.0)
    ni_wt: float = Field(..., ge=0.0)
    cr_wt: float = Field(..., ge=0.0)
    mo_wt: float = Field(..., ge=0.0)
    v_wt: float = Field(..., ge=0.0)
    al_wt: float = Field(..., ge=0.0)
    cu_wt: float = Field(..., ge=0.0)
    target_hrc: float = Field(..., ge=0.0)

    model_config = ConfigDict(extra='forbid')

@app.post("/recommend")
def recommend(comp: RecommendInput):
    models, meta = get_models()
    comp_dict = comp.model_dump(exclude={"target_hrc"})
    target_hrc = comp.target_hrc
    
    from materials_data_lab.recommender import recommend_recipe
    
    result = recommend_recipe(models, meta, comp_dict, target_hrc)
    return result

