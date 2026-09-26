# Materials Data Lab — Phase 7A: Baseline Merdiveni

Bu rapor V7 Audit Supplement kapsamında, Dummy ve basit doğrusal modellerle mevcut şampiyon modelin (XGB_tuned) salt-okunur olarak (S2 GroupKFold) kıyaslanmasını içerir.

## Yöntem
- Protokol: S2 GroupKFold(n_splits=5, groups=steel_type)
- Metrikler: OOF (pooled) MAE, OOF (pooled) RMSE, OOF (pooled) R², Mutlak Hata (Residüel) p50/p90/p99.
- Özellik Seti: Mevcut şampiyon özellik seti kullanılmıştır. XGB_Hp modeline ek olarak sabit-C ile (C=19.5) Hollomon-Jaffe (Hp) özelliği eklenmiştir.

## Rekonsiliyasyon Notu (Agregasyon Yöntemleri)
Önceki raporlarda (örn. Phase 5) MAE metriği **fold-mean** (katman ortalaması) olarak verilmiştir (XGB_tuned için `2.35 ± 0.39`). Bu raporda (Phase 7) ise **pooled OOF** (tüm OOF tahminlerin tek havuzda toplanıp hesaplanması) kullanılmıştır (XGB_tuned için `2.48`). GroupKFold ile bölünen grupların satır sayıları (fold sizes) dengesiz olduğu için, fold-mean küçük gruplara fazla ağırlık verirken, pooled OOF tüm satırlara eşit ağırlık verir. Bu durum bir çelişki değil, iki farklı agregasyon perspektifidir.

## Sonuçlar Tablosu

| Model | OOF (pooled) MAE | OOF (pooled) RMSE | OOF (pooled) R² | |resid| p50 | |resid| p90 | |resid| p99 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DummyRegressor (mean) | 11.59 | 14.19 | -0.017 | 9.94 | 23.33 | 34.20 |
| DummyRegressor (median) | 11.51 | 14.27 | -0.027 | 9.55 | 23.55 | 35.88 |
| Ridge (std-scale ile) | 3.86 | 4.94 | 0.877 | 3.24 | 7.57 | 14.20 |
| RF_tuned (mevcut) | 2.71 | 3.66 | 0.932 | 2.16 | 5.62 | 11.41 |
| **XGB şampiyon (mevcut)** | **2.48** | **3.33** | **0.944** | **1.94** | **4.98** | **11.16** |
| XGB+Hp (sabit-C) | 2.48 | 3.34 | 0.944 | 1.99 | 5.06 | 11.33 |

*Not: XGB_Hp modeli (Hp özelliği eklenmiş model), XGB şampiyon modelinden daha iyi performans göstermemiştir (+0.005 MAE farkı ile pratik olarak eşdeğer veya gürültü sınırındadır). D-12 0.15 HRC kuralına göre reddedilmiştir.*
