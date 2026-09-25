import json
from materials_data_lab.api import get_models
from materials_data_lab.recommender import recommend_recipe

steels = {
    "4140": {"c_wt": 0.40, "mn_wt": 0.85, "p_wt": 0.015, "s_wt": 0.015, "si_wt": 0.25, "ni_wt": 0.0, "cr_wt": 1.0, "mo_wt": 0.20, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0},
    "1045": {"c_wt": 0.45, "mn_wt": 0.75, "p_wt": 0.02, "s_wt": 0.02, "si_wt": 0.20, "ni_wt": 0.0, "cr_wt": 0.0, "mo_wt": 0.0, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0},
    "4340": {"c_wt": 0.40, "mn_wt": 0.70, "p_wt": 0.01, "s_wt": 0.01, "si_wt": 0.25, "ni_wt": 1.8, "cr_wt": 0.8, "mo_wt": 0.25, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0},
    "1080": {"c_wt": 0.80, "mn_wt": 0.75, "p_wt": 0.02, "s_wt": 0.02, "si_wt": 0.20, "ni_wt": 0.0, "cr_wt": 0.0, "mo_wt": 0.0, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0},
    "5160": {"c_wt": 0.60, "mn_wt": 0.85, "p_wt": 0.02, "s_wt": 0.02, "si_wt": 0.20, "ni_wt": 0.0, "cr_wt": 0.8, "mo_wt": 0.0, "v_wt": 0.0, "al_wt": 0.0, "cu_wt": 0.0}
}

targets = [30.0, 35.0, 40.0, 45.0, 50.0]
models, meta = get_models()

out_md = []
out_md.append("# Materials Data Lab — V5-A: Recommender Validation")
out_md.append("")
out_md.append("| Steel | Target (HRC) | Recipe T (°C) | Recipe t (s) | Pred HRC | Round-Trip Err | Status |")
out_md.append("|-------|--------------|---------------|--------------|----------|----------------|--------|")

not_achievable_examples = []
example_4140_45 = None

for name, comp in steels.items():
    for target in targets:
        res = recommend_recipe(models, meta, comp, target)
        if res["recommendable"]:
            err = abs(res["predicted_hrc"] - target)
            status = "PASS" if err <= 0.5 else f"WARN ({err:.2f})"
            out_md.append(f"| {name} | {target} | {res['T_c']} | {res['t_s']} | {res['predicted_hrc']:.2f} | {err:.2f} | {status} |")
            if name == "4140" and target == 45.0:
                example_4140_45 = res
        else:
            out_md.append(f"| {name} | {target} | - | - | - | - | NOT_ACHIEVABLE |")
            not_achievable_examples.append((name, target, res["notes"][-1]))

out_md.append("\n## NOT_ACHIEVABLE EXAMPLES\n")
for ex in not_achievable_examples:
    out_md.append(f"- **{ex[0]}** (Target: {ex[1]}): {ex[2]}")

out_md.append("\n## EXAMPLE 4140 -> 45 HRC\n```json\n")
out_md.append(json.dumps(example_4140_45, indent=2))
out_md.append("\n```\n")

with open("reports/phase_v5a_recommender.md", "w", encoding="utf-8") as f:
    f.write("\n".join(out_md))

