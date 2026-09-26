# Materials Data Lab — Phase 7B: Ablation Merdiveni

Fiziksel / metalurjik özellik mühendisliğinin katkılarını adım adım görmek için yapılan çıkartma/ekleme çalışması.

## Yöntem
- Şampiyon Model (XGB_tuned) temel alınarak özellikleri adım adım ekliyoruz.
- Protokol: S2 GroupKFold (steel_type), random_state=42.

## Sonuçlar Tablosu

| Basamak | OOF (pooled) MAE | OOF (pooled) RMSE | OOF (pooled) R² | |resid| p50 | |resid| p90 | |resid| p99 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (1) Yalnız 11 kimya | 10.60 | 12.95 | 0.150 | 9.17 | 19.78 | 32.08 |
| (2) + temper_temp_c, temper_time_s | 2.48 | 3.33 | 0.944 | 1.94 | 4.98 | 11.16 |
| (3) + log_time | 2.48 | 3.33 | 0.944 | 1.94 | 4.98 | 11.16 |
| (4) + Hp sabit-C | 2.48 | 3.34 | 0.944 | 1.99 | 5.06 | 11.33 |
| (5) threshold-rejected (D-18) | - | - | - | - | - | - |

## Sonuç
Ağaç tabanlı (XGBoost) bir model kullanıldığından, `temper_time_s` eklendikten sonra bunun monotonik dönüşümü olan `log_time` özelliğinin eklenmesi modelin gücünü artırmamıştır (Basamak 2 ve 3'ün OOF MAE değerleri virgülden sonra 4 haneye kadar tamamen özdeştir: 2.4762 HRC). Bu durum, log_time dönüşümünün ağaçlar için tamamen gereksiz olduğunu kanıtlamaktadır. Benzer şekilde, formülize edilmiş sabit-C'li Hp parametresi (Basamak 4) ağacın zaten sıcaklık ve zaman ile kurduğu non-lineer etkileşimi geçememiştir (MAE = 2.48, +0.005 gürültü/kötüleşme).

Fiziksel özellikler, D-12 eşiğine (0.15 HRC) göre **"not material"** hükmü almıştır. Ağaç modeli zaten kendi iç etkileşimlerini veriden optimal şekilde öğrenebilmektedir.
