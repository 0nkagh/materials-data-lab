# Data Sheet

## Dataset Origin
- **Source Name:** Tempering data for carbon and low alloy steels
- **Publisher:** Raiipa Technologies (Kaggle: rgerschtzsauer)
- **URL:** [Kaggle Dataset](https://www.kaggle.com/datasets/rgerschtzsauer/tempering-data-for-carbon-and-low-alloy-steels)
- **License:** CC BY 4.0
- **Integrity:** SHA-256 validation enforced during loading.

## Data Shape and Statistics
- **Structure:** 1466 rows, 17 columns (post-cleaning)
- **Duplicates:** 0 duplicate rows found after structural validation (Cleaning rule D-06).

### Physical Ranges (Observed Support)
- **Temperatures ($T$):** 100°C to 705°C
- **Time ($t$):** 10 seconds to 115,200 seconds
- **Post-Temper Hardness (HRC):** Min and Max bounds based directly on the raw dataset.

### Chemistry Ranges (wt%)
| Element | Min | Max |
| :--- | :--- | :--- |
| C | 0.19 | 0.77 |
| Mn | 0.28 | 1.70 |
| P | 0.007 | 0.051 |
| S | 0.003 | 0.046 |
| Si | 0.12 | 1.51 |
| Ni | 0.0 | 1.95 |
| Cr | 0.0 | 1.10 |
| Mo | 0.0 | 0.52 |
| V | 0.0 | 0.03 |
| Al | 0.0 | 0.081 |
| Cu | 0.0 | 0.23 |

## Known Caveats & Structural Limitations
1. **HV to HRC Conversion Limits:** The dataset contains 82 rows that fall outside the standard ASTM E140 valid conversion bounds for Vickers (HV) to Rockwell C (HRC). These rows were preserved (rule D-01) but noted as potential noise factors at boundary extremes.
2. **Missing `initial_hrc` Data:** 949 out of 1466 rows (approx. 65%) are entirely missing `initial_hrc` measurements (only 517/1466 have values). Consequently, pre-tempering hardness cannot be used as an active feature in global training.
3. **Missing Cooling Mediums:** Crucial metallurgical context like quenching method (water, oil, air) and geometrical dimensions of the steel samples are not present in this dataset.

## Dictionary
For column definitions and unit details, refer strictly to [data_dictionary.md](data_dictionary.md).
