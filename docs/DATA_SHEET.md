# Data Sheet

## Dataset Origin
- **Source Name:** Tempering data for carbon and low alloy steels
- **Publisher:** Raiipa Technologies (Kaggle: `rgerschtzsauer`)
- **URL:** [Kaggle Dataset](https://www.kaggle.com/datasets/rgerschtzsauer/tempering-data-for-carbon-and-low-alloy-steels)
- **License:** CC BY 4.0
- **Integrity:** Read-only access with SHA-256 verification (`3f88fdccf4bb2f3c03be6eeca03001b4bd8a2b0de5cd4818195486182a4762b0`).

## Data Shape and Statistics
- **Structure:** `1466` rows, `17` columns (`34` unique `Steel type` strings after `.str.strip()`, `37` distinct 11-element chemistry vectors).
- **Duplicates:** `0` exact 17-column duplicates and `0` identical 13-input (`11 elements + T + t`) duplicates.
- **Missing Policy:** Literal `"?"` values in `Initial hardness (HRC) - post quenching` are parsed as `NaN` (`949/1466` missing, all in `Grange and Baughman, 1956`). Univariate bounds below are computed over non-null values (`n = 1466` for all 13 model features and `final_hrc`; `n = 517` for `initial_hrc`).

## Observed Univariate Ranges vs. Application Supported Input Scope
> **Correction Note (2026-11-17):** Earlier revisions of this table (`0.19–0.77% C`, `0.0–1.95% Ni`, etc.) did not match the raw CSV and conflicted with `reports/phase3_eda.md`. The observed dataset bounds below are pinned directly to the canonical `src/materials_data_lab/data_bounds.py` (`CSV SHA-256 = 3f88fdccf4bb2f3c03be6eeca03001b4bd8a2b0de5cd4818195486182a4762b0`). Note that **observed dataset range**, **application supported input scope** (`SUPPORTED_INPUT_BOUNDS` in `data_bounds.py` / Pydantic bounds in `api.py`), and **true metallurgical suitability** are three distinct concepts: the `0 < T <= 800.0 °C` check reflects the repository's configured software input cap (aligned with historical `thresholds.py`), **not** a universal metallurgical boundary, physical impossibility, or composition-independent austenitizing limit (`Ac1` and transformation behavior depend on steel composition and conditions). Passing 1D marginal min/max checks does **not** verify joint multivariate support or metallurgical suitability.

### Process & Target Ranges
| Variable | Unit | Observed Dataset `[min, max]` (`n=1466`) | Application Supported Input Scope (`api.py` / `data_bounds.py`) |
| :--- | :--- | :--- | :--- |
| `temper_temp_c` | °C | `100.0` to `704.4` | `(0.0, 800.0]` finite (app scope, not universal `Ac1`/physical limit) |
| `temper_time_s` | s | `10.0` to `115200.0` | `> 0.0` finite |
| `final_hrc` (target) | HRC | `0.9` to `68.5` | N/A (`recommend` target `[0.0, 100.0]`) |
| `initial_hrc` (excluded `D-08`) | HRC | `46.5` to `67.0` (`n=517`, `949` `NaN`) | Excluded from global MVP features |

### Observed Chemistry Ranges (%wt, `n=1466`)
All element inputs must be finite real numbers within this application's supported input scope (`[0.0, 100.0]` %wt per element and `sum(all 11 elements) <= 100.0` %wt; distinct from both observed dataset support and metallurgical suitability). Marginal observed dataset bounds are:

| Element | Observed Min (%wt) | Observed Max (%wt) |
| :--- | :---: | :---: |
| `c_wt` (C) | `0.25` | `1.15` |
| `mn_wt` (Mn) | `0.30` | `1.85` |
| `p_wt` (P) | `0.007` | `0.054` |
| `s_wt` (S) | `0.005` | `0.055` |
| `si_wt` (Si) | `0.06` | `1.62` |
| `ni_wt` (Ni) | `0.0` | `3.41` |
| `cr_wt` (Cr) | `0.0` | `1.57` |
| `mo_wt` (Mo) | `0.0` | `0.36` |
| `v_wt` (V) | `0.0` | `0.16` |
| `al_wt` (Al) | `0.0` | `1.26` |
| `cu_wt` (Cu) | `0.0` | `0.08` |

## Known Caveats & Structural Limitations
1. **Values `< 20 HRC` (`E140_EXTRAPOLATION_SUSPECT`):** `123` rows fall below `20 HRC` (`Grange=92`, `Hollomon=21`, `Penha=10`), outside ASTM E18 / E140 (`238 HV`) reliable range. Preserved per `D-02` with quality flags.
2. **Missing `initial_hrc` Data:** `949` of `1466` rows (`64.7%`, entire Grange subset) lack `initial_hrc`. Excluded from global modeling (`D-08`), and synthetic imputation rejected (`D-15`).
3. **Missing Process & Provenance Metadata:** Quenching medium (water, oil, air), austenitizing conditions, sample geometry/section size, and furnace/heat IDs are absent; `Penha (2010)` primary citation remains `AMBIGUOUS` (`docs/references.md`).

## Dictionary
For column definitions and units, see [data_dictionary.md](data_dictionary.md).
