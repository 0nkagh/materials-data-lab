# Materials Data Lab — Phase V2-D: Steel Class Analysis

## Sınıf Dağılımı ve Köken
| Steel Class | Total Rows | Source Breakdown |
| :--- | :--- | :--- |
| **plain_carbon** | 579 | Grange and Baughman, 1956: 309, Hollomon and Jaffe, 1945: 270 |
| **UNKNOWN_CLASS** | 201 | Grange and Baughman, 1956: 150, Penha, 2010: 51 |
| **Mo_steel** | 160 | Grange and Baughman, 1956: 160 |
| **Ni_Cr_Mo** | 101 | Grange and Baughman, 1956: 50, Penha, 2010: 51 |
| **Cr_steel** | 101 | Grange and Baughman, 1956: 50, Penha, 2010: 51 |
| **Cr_V** | 101 | Grange and Baughman, 1956: 50, Penha, 2010: 51 |
| **Cr_Mo** | 93 | Grange and Baughman, 1956: 50, Penha, 2010: 43 |
| **Mn_steel** | 50 | Grange and Baughman, 1956: 50 |
| **Si_steel** | 40 | Grange and Baughman, 1956: 40 |
| **nitriding** | 40 | Grange and Baughman, 1956: 40 |

## MAE Kıyası: XGB_tuned vs RF_tuned
| Steel Class | Rows | XGB_tuned MAE | RF_tuned MAE | Winner |
| :--- | :--- | :--- | :--- | :--- |
| **Cr_Mo** | 93 | 2.186 | 2.255 | XGB |
| **Cr_V** | 101 | 1.782 | 1.994 | XGB |
| **Cr_steel** | 101 | 3.030 | 2.520 | RF |
| **Mn_steel** | 50 | 1.315 | 1.251 | RF |
| **Mo_steel** | 160 | 1.751 | 2.197 | XGB |
| **Ni_Cr_Mo** | 101 | 2.066 | 1.921 | RF |
| **Si_steel** | 40 | 9.261 | 7.438 | RF |
| **UNKNOWN_CLASS** | 201 | 2.094 | 2.147 | XGB |
| **nitriding** | 40 | 4.073 | 4.339 | XGB |
| **plain_carbon** | 579 | 2.160 | 2.927 | XGB |

## A/B Mini Deney (One-hot `steel_class`)
- XGBoost Base S2 MAE: 2.353
- XGBoost One-hot S2 MAE: 2.331
- **Sonuç**: Adding one-hot class improved MAE by 0.0212 HRC. This is <0.1 HRC, so the gain is not material.