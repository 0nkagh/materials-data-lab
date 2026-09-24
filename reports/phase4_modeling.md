# Materials Data Lab — Phase 4: Modelleme MVP

## Yöntem Özeti ve Kararlar
- D-08: initial_hrc MVP'de kullanılmadı (eksik veri oranı yüksek).
- Feature Engineering: log10(temper_time_s) kullanıldı.
- Physics Baseline: p_hj = (T_K/1000) * (19.5 + log10(t_saat)) [C=19.5 ASSUMED, REPORT_ONLY]

## Gözlem: RandomSplit (S1) vs GroupSplit (S2)
> **ZORUNLU NOT**: S1'de aynı çeliğin satırları train ve testte kalabildiğinden sonuçlar iyimser olabilir; S2 ("daha önce görülmemiş çelik") daha gerçekçi üst sınırdır. Nedensellik iddiası taşımaz, model gözlemidir.

### S1: RandomSplit (Test Size 0.2)
| Model | MAE | RMSE | R² |
| :--- | :-: | :-: | :-: |
| B1_Naive | 11.8058 | 14.8410 | -0.0530 |
| B2_Physics | 4.5569 | 5.8928 | 0.8340 |
| M1_RF | 1.5153 | 2.3470 | 0.9737 |

### S2: GroupSplit (GroupKFold = 5, steel_type grouped)
| Model | MAE (mean ± std) | RMSE (mean ± std) | R² (mean ± std) |
| :--- | :-: | :-: | :-: |
| B1_Naive | 11.5082 ± 0.6881 | 14.2481 ± 0.7138 | -0.0686 ± 0.1130 |
| B2_Physics | 4.6735 ± 1.0928 | 5.7600 ± 1.1441 | 0.8098 ± 0.1020 |
| M1_RF | 2.6872 ± 0.3886 | 3.5735 ± 0.5561 | 0.9292 ± 0.0280 |

## B2 vs M1 Karşılaştırması
**Gözlem**: RandomForest (M1) genel hatlarıyla fiziksel baseline'dan (B2) daha düşük hata oranlarına ulaşmıştır. Ancak, fiziksel denklemin (p_hj) yalnızca tek bir değişkene dayalı basit bir Linear Regression modeli (B2) ile gösterdiği başarı dikkat çekicidir.

## RF Feature Importance (MDI) - İlk 10
> **UYARI**: Korele metallurjik özelliklerde MDI yanıltıcı olabilir [REPORT_ONLY]

- **temper_temp_c**: 0.7649
- **c_wt**: 0.0869
- **log_time**: 0.0833
- **cr_wt**: 0.0344
- **si_wt**: 0.0087
- **s_wt**: 0.0049
- **mo_wt**: 0.0046
- **p_wt**: 0.0043
- **mn_wt**: 0.0042
- **ni_wt**: 0.0033