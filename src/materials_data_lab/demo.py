import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import tempfile
import time
import os
from materials_data_lab.modeling import RF_FEATURES

# Observed bounds in the dataset
OBSERVED_BOUNDS = {
    "c_wt": (0.19, 0.77),
    "mn_wt": (0.28, 1.70),
    "p_wt": (0.007, 0.051),
    "s_wt": (0.003, 0.046),
    "si_wt": (0.12, 1.51),
    "ni_wt": (0.0, 1.95),
    "cr_wt": (0.0, 1.10),
    "mo_wt": (0.0, 0.52),
    "v_wt": (0.0, 0.03),
    "al_wt": (0.0, 0.081),
    "cu_wt": (0.0, 0.23),
    "temper_temp_c": (100.0, 705.0),
    "temper_time_s": (10.0, 115200.0)
}

def check_extrapolation(inputs: dict[str, float]) -> str:
    """Check if inputs are outside the observed data range."""
    for key, val in inputs.items():
        min_val, max_val = OBSERVED_BOUNDS[key]
        if val < min_val or val > max_val:
            return "⚠ Input outside observed data range — extrapolation"
    return "✅ Within observed bounds"

def make_prediction(models, meta, c_wt, mn_wt, p_wt, s_wt, si_wt, ni_wt, cr_wt, mo_wt, v_wt, al_wt, cu_wt, temper_temp_c, temper_time_s) -> tuple[str, str, str]:
    """Pure function to handle the prediction logic, disconnected from the UI definition."""
    if meta is not None and meta.get("features") != RF_FEATURES:
        raise ValueError("Artifact features mismatch with code")

    inputs = {
        "c_wt": c_wt, "mn_wt": mn_wt, "p_wt": p_wt, "s_wt": s_wt, "si_wt": si_wt,
        "ni_wt": ni_wt, "cr_wt": cr_wt, "mo_wt": mo_wt, "v_wt": v_wt, "al_wt": al_wt, "cu_wt": cu_wt,
        "temper_temp_c": temper_temp_c, "temper_time_s": temper_time_s
    }
    
    warning_text = check_extrapolation(inputs)
    
    log_time = np.log10(temper_time_s) if temper_time_s > 0 else 0
    
    # Feature ordering must match RF_FEATURES from modeling.py exactly
    # ['c_wt', 'mn_wt', 'p_wt', 's_wt', 'si_wt', 'ni_wt', 'cr_wt', 'mo_wt', 'v_wt', 'al_wt', 'cu_wt', 'temper_temp_c', 'log_time']
    x_rf = np.array([[c_wt, mn_wt, p_wt, s_wt, si_wt, ni_wt, cr_wt, mo_wt, v_wt, al_wt, cu_wt, temper_temp_c, log_time]])
    
    temp_k = temper_temp_c + 273.15
    t_saat = temper_time_s / 3600.0
    p_hj = (temp_k / 1000.0) * (19.5 + np.log10(t_saat) if t_saat > 0 else 0)
    
    x_phys = np.array([[p_hj]])
    
    xgb_model = models["xgb"]
    b2_model = models["b2"]
    
    xgb_pred = xgb_model.predict(x_rf)[0]
    b2_pred = b2_model.predict(x_phys)[0]
    
    q90 = meta.get("conformal_q90", 5.08) if meta else 5.08
    interval_text = f"90% interval: {xgb_pred - q90:.1f} – {xgb_pred + q90:.1f} HRC (q90 = ±{q90:.2f} HRC)"
    
    time.time()
    explainer = shap.TreeExplainer(xgb_model)
    df_rf = pd.DataFrame(x_rf, columns=RF_FEATURES)
    shap_values = explainer(df_rf)
    
    fig = plt.figure(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    tmp_path = os.path.join(tempfile.gettempdir(), f"shap_{time.time()}.png")
    plt.tight_layout()
    plt.savefig(tmp_path, bbox_inches='tight')
    plt.close(fig)
    
    feature_names = RF_FEATURES
    vals = shap_values[0].values
    top_indices = np.argsort(np.abs(vals))[-3:][::-1]
    top_text = ", ".join([f"{feature_names[i]}: {vals[i]:+.1f} HRC" for i in top_indices])
    
    return f"{xgb_pred:.1f} HRC", interval_text, f"{b2_pred:.1f} HRC", warning_text, tmp_path, top_text

def create_demo(models, meta=None):
    """Creates the Gradio interface block."""
    with gr.Blocks(title="Materials Data Lab - Local Demo") as demo:
        gr.Markdown("# Materials Data Lab — Tempering Hardness Predictor")
        gr.Markdown("model v1.1 · champion: XGB_tuned · data: Raiipa CC BY 4.0")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Composition (wt%)")
                c_wt = gr.Number(label="Carbon (c_wt)", value=0.40)
                mn_wt = gr.Number(label="Manganese (mn_wt)", value=0.80)
                p_wt = gr.Number(label="Phosphorus (p_wt)", value=0.015)
                s_wt = gr.Number(label="Sulfur (s_wt)", value=0.015)
                si_wt = gr.Number(label="Silicon (si_wt)", value=0.25)
                ni_wt = gr.Number(label="Nickel (ni_wt)", value=0.0)
                cr_wt = gr.Number(label="Chromium (cr_wt)", value=0.0)
                mo_wt = gr.Number(label="Molybdenum (mo_wt)", value=0.0)
                v_wt = gr.Number(label="Vanadium (v_wt)", value=0.0)
                al_wt = gr.Number(label="Aluminum (al_wt)", value=0.0)
                cu_wt = gr.Number(label="Copper (cu_wt)", value=0.0)
                
            with gr.Column():
                gr.Markdown("### Tempering Conditions")
                temper_temp_c = gr.Slider(minimum=100.0, maximum=705.0, value=400.0, label="Temperature (°C)")
                temper_time_s = gr.Slider(minimum=10.0, maximum=115200.0, step=10, value=3600.0, label="Time (s)")
                
                predict_btn = gr.Button("Predict Hardness", variant="primary")
                
                gr.Markdown("### Predictions")
                with gr.Row():
                    with gr.Column():
                        xgb_out = gr.Textbox(label="Champion: XGBoost (tuned)", text_align="center")
                        interval_out = gr.Textbox(label="Conformal Interval", text_align="center")
                    b2_out = gr.Textbox(label="Physics Baseline (B2)", text_align="center")
                
                warning_out = gr.Textbox(label="Data Quality Status", interactive=False)
                
                with gr.Accordion("Why this prediction?", open=False):
                    top_text_out = gr.Textbox(label="Top 3 Features", interactive=False)
                    shap_img_out = gr.Image(label="SHAP Waterfall", type="filepath")
                
        inputs = [c_wt, mn_wt, p_wt, s_wt, si_wt, ni_wt, cr_wt, mo_wt, v_wt, al_wt, cu_wt, temper_temp_c, temper_time_s]
        outputs = [xgb_out, interval_out, b2_out, warning_out, shap_img_out, top_text_out]
        
        def predict_wrapper(*args):
            return make_prediction(models, meta, *args)
            
        predict_btn.click(fn=predict_wrapper, inputs=inputs, outputs=outputs)
        
        gr.Markdown("---")
        gr.Markdown(
            "Data source: Tempering data for carbon and low alloy steels (Raiipa Technologies) | "
            "Educational demo, not for production QC | "
            "[GitHub Repository](https://github.com/0nkagh/materials-data-lab)"
        )
        
    return demo
