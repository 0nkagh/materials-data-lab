"""Gradio demo interface and prediction helper for Materials Data Lab."""

import os
import tempfile
import time

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from materials_data_lab import __version__
from materials_data_lab.data_bounds import (
    BOUNDS_PROVENANCE,
    OBSERVED_UNIVARIATE_BOUNDS,
    check_observed_bounds,
    validate_supported_inputs,
)
from materials_data_lab.modeling import RF_FEATURES

OBSERVED_BOUNDS = OBSERVED_UNIVARIATE_BOUNDS


def check_extrapolation(inputs: dict[str, float]) -> str:
    """Validate application supported input scope and check 1D observed dataset bounds.

    Raises ValueError on non-numeric, NaN, Infinity, or out-of-supported-scope inputs.
    Observed dataset range, application supported input scope, and true metallurgical
    suitability are distinct concepts; passing 1D bounds does NOT verify joint
    multivariate support or metallurgical suitability.
    """
    validated = validate_supported_inputs(inputs, require_treatment=True)
    within, oob_details = check_observed_bounds(validated)
    if not within:
        return (
            "⚠ Input outside 1D observed data bounds "
            f"({'; '.join(oob_details)}) — within application input scope, but outside observed dataset range; "
            "high extrapolation uncertainty and metallurgical suitability unverified"
        )
    return (
        "✅ Within application input scope and 1D observed bounds "
        "(observed range, app scope, and metallurgical suitability are distinct; "
        "joint multivariate support and metallurgical suitability remain unverified)"
    )


def make_prediction(
    models,
    meta,
    c_wt,
    mn_wt,
    p_wt,
    s_wt,
    si_wt,
    ni_wt,
    cr_wt,
    mo_wt,
    v_wt,
    al_wt,
    cu_wt,
    temper_temp_c,
    temper_time_s,
) -> tuple[str, str, str, str, str, str]:
    """Handle single-point hardness prediction and SHAP attribution."""
    if meta is not None and meta.get("features") != RF_FEATURES:
        raise ValueError("Artifact features mismatch with code")

    raw_inputs = {
        "c_wt": c_wt,
        "mn_wt": mn_wt,
        "p_wt": p_wt,
        "s_wt": s_wt,
        "si_wt": si_wt,
        "ni_wt": ni_wt,
        "cr_wt": cr_wt,
        "mo_wt": mo_wt,
        "v_wt": v_wt,
        "al_wt": al_wt,
        "cu_wt": cu_wt,
        "temper_temp_c": temper_temp_c,
        "temper_time_s": temper_time_s,
    }
    validated = validate_supported_inputs(raw_inputs, require_treatment=True)
    warning_text = check_extrapolation(validated)

    t_s = validated["temper_time_s"]
    temp_c = validated["temper_temp_c"]
    log_time = np.log10(t_s)

    # Feature ordering must match RF_FEATURES from modeling.py exactly
    x_rf = np.array([[validated[feat] if feat != "log_time" else log_time for feat in RF_FEATURES]])

    temp_k = temp_c + 273.15
    t_saat = t_s / 3600.0
    p_hj = (temp_k / 1000.0) * (19.5 + np.log10(t_saat))
    x_phys = np.array([[p_hj]])

    xgb_model = models["xgb"]
    b2_model = models["b2"]

    xgb_pred = float(xgb_model.predict(x_rf)[0])
    b2_pred = float(b2_model.predict(x_phys)[0])

    q90 = float(meta.get("conformal_q90", 5.08)) if meta else 5.08
    interval_text = (
        f"OOF empirical band (nominal 90%, not guaranteed): "
        f"{xgb_pred - q90:.1f} – {xgb_pred + q90:.1f} HRC (q90 = ±{q90:.2f} HRC)"
    )

    explainer = shap.TreeExplainer(xgb_model)
    df_rf = pd.DataFrame(x_rf, columns=RF_FEATURES)
    shap_values = explainer(df_rf)

    fig = plt.figure(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    tmp_path = os.path.join(tempfile.gettempdir(), f"shap_{time.time()}.png")
    plt.tight_layout()
    plt.savefig(tmp_path, bbox_inches="tight")
    plt.close(fig)

    vals = shap_values[0].values
    top_indices = np.argsort(np.abs(vals))[-3:][::-1]
    top_text = (
        ", ".join([f"{RF_FEATURES[i]}: {vals[i]:+.1f} HRC" for i in top_indices])
        + " (model attribution over correlated features; not causal effect)"
    )

    return (
        f"{xgb_pred:.1f} HRC",
        interval_text,
        f"{b2_pred:.1f} HRC",
        warning_text,
        tmp_path,
        top_text,
    )


def create_demo(models, meta=None):
    """Create the local Gradio interface block."""
    q90 = float(meta.get("conformal_q90", 5.08)) if meta else 5.08
    with gr.Blocks(title="Materials Data Lab - Research Prototype") as demo:
        gr.Markdown("# Materials Data Lab — Tempering Hardness Predictor")
        gr.Markdown(
            f"model v{__version__} · champion: XGB_tuned · OOF q90=±{q90:.2f} HRC · "
            f"bounds SHA-256: `{BOUNDS_PROVENANCE['csv_sha256'][:12]}...`"
        )
        gr.Markdown(
            "> **Research prototype only:** Observed 1D bounds do not prove multivariate support; "
            "the dataset lacks initial quenched hardness, cooling medium, and section geometry; "
            "per-family OOF empirical coverage varies widely (severely under-covered in `9264`, "
            "`0.31%C plain carbon`, `5160`, `Nitriding Steel`, and `E52100`)."
        )

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
                temper_temp_c = gr.Slider(minimum=100.0, maximum=704.4, value=400.0, label="Temperature (°C)")
                temper_time_s = gr.Slider(minimum=10.0, maximum=115200.0, step=10, value=3600.0, label="Time (s)")

                predict_btn = gr.Button("Predict Hardness", variant="primary")

                gr.Markdown("### Predictions")
                with gr.Row():
                    with gr.Column():
                        xgb_out = gr.Textbox(label="Champion: XGBoost (point prediction)", text_align="center")
                        interval_out = gr.Textbox(
                            label="OOF Empirical Prediction Band (not externally guaranteed)",
                            text_align="center",
                        )
                    b2_out = gr.Textbox(label="Physics Baseline (B2, assumed C=19.5)", text_align="center")

                warning_out = gr.Textbox(
                    label="Data Boundary & Scope Status (1D observed vs. app input scope; metallurgical suitability unverified)",
                    interactive=False,
                )

                with gr.Accordion(
                    "Model Feature Attribution (SHAP — correlated inputs, not causal metallurgical effect)",
                    open=False,
                ):
                    gr.Markdown(
                        "_Note: Alloying elements co-vary across steel grades; SHAP values explain model output "
                        "decomposition and do not represent isolated physical causality._"
                    )
                    top_text_out = gr.Textbox(label="Top 3 Model Attributions", interactive=False)
                    shap_img_out = gr.Image(label="SHAP Waterfall (Model Output)", type="filepath")

        inputs = [
            c_wt, mn_wt, p_wt, s_wt, si_wt, ni_wt, cr_wt, mo_wt, v_wt, al_wt, cu_wt,
            temper_temp_c, temper_time_s,
        ]
        outputs = [xgb_out, interval_out, b2_out, warning_out, shap_img_out, top_text_out]

        def predict_wrapper(*args):
            return make_prediction(models, meta, *args)

        predict_btn.click(fn=predict_wrapper, inputs=inputs, outputs=outputs)

        gr.Markdown("---")
        gr.Markdown(
            "Data source: Tempering data for carbon and low alloy steels (Raiipa Technologies, CC BY 4.0) | "
            "Educational/research prototype, not for production QC | "
            "[GitHub Repository](https://github.com/0nkagh/materials-data-lab)"
        )

    return demo
