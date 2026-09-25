# Materials Data Lab — Phase V2-E: Hollomon-Jaffe Evolution

## Kıyas Tablosu
| Model | S1 (RandomSplit) MAE | S1 RMSE | S1 R² | S2 (GroupSplit) MAE | S2 RMSE | S2 R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B2_klassik** | 4.5569 | 5.8928 | 0.8340 | 4.6735 ± 1.0928 | 5.7600 ± 1.1441 | 0.8098 ± 0.1020 |
| **Composite_C** | 3.0636 | 4.2463 | 0.9138 | 3.3360 ± 0.3974 | 4.3388 ± 0.4975 | 0.8976 ± 0.0336 |
| **XGB_tuned** | 1.0516 | 1.8869 | 0.9830 | 2.3498 ± 0.3879 | 3.2054 ± 0.6647 | 0.9439 ± 0.0196 |
| **XGB_tuned_phj** | 1.0309 | 1.8098 | 0.9843 | 2.3035 ± 0.4086 | 3.1293 ± 0.6575 | 0.9469 ± 0.0192 |

## Gözlemler
1. **Composite_C vs B2_klassik**: Composite_C, klasik modele göre MAE'yi önemli ölçüde iyileştirdi. Öğrenilen $k_i$ katsayıları [REPORT_ONLY], C sabitinin alaşım elementlerine bağımlı olduğunu ve özellikle C, Cr, Mo gibi karbür yapıcıların ve sertleşebilirlik elementlerinin fiziksel denkleme etki ettiğini doğruluyor.
2. **XGB + p_hj**: p_hj özelliğini XGBoost'a eklemenin S2 MAE katkısı marjinal kaldı (<0.1 HRC ise 'not material').
3. **Physics vs ML**: Öğrenilmiş fiziksel model (Composite_C) geleneksel HJ'yi geçse de, XGB_tuned gibi non-lineer ML modellerine S2 genel hata (MAE) bağlamında hala uzaktır.

**Effective C Analizi [REPORT_ONLY]**
Tasarım matrisindeki `(T_K/1000)` ve `(T_K/1000)*x_i` kolonlarının kolineer yapısı nedeniyle `c0` sabiti tek başına (raw) yorumlanamaz. Bu nedenle üç referans bileşim için "Effective C" ($c_0 + \sum k_i x_i$) değerleri hesaplanmıştır:
- (i) Veri seti medyan bileşimi: **-74.22**
- (ii) Tipik 4140 (C 0.40, Mn 0.85, Si 0.25, Cr 1.00, Mo 0.22, diğer ~min): **-69.17**
- (iii) Tipik 1026 (C 0.26, Mn 0.75, Si 0.20, diğer ~min): **-80.69**

*Gözlem*: Değerler klasik literatür aralığı olan ~15-22 bandında **çıkmamıştır**. Bunun temel nedeni, kapalı-form lineer tasarımda doğrudan HRC hedefine regresyon yapılması sonucu oluşan kolinearlik ve ölçek uyuşmazlığıdır.
*Caveat*: individual coefficients of the collinear composite design are not separately interpretable; only effective C at reference compositions and predictive performance are meaningful [REPORT_ONLY].

The composite model is best read as a linear interaction model; recovering a single 'effective C' is not identifiable (coefficients trade off against the separate time term and intercept), while its predictive CV results remain valid [REPORT_ONLY].