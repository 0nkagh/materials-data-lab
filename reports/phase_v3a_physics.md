# Materials Data Lab — V3-A: Physics-guided XGBoost & Conformal Intervals

## Yöntem Özeti ve Kararlar
- D-15: initial_hrc imputasyonu REDDİ (Verinin %64'ü eksik, sentetik doldurma overconfidence yaratır).
- D-16: MLflow/W&B REDDİ (Statik karar defteri / markdown raporlama karmaşıklığı yeterince düşük tutuyor).
- D-17: Monotone constraints deneyi.
- D-18: Metalurjik özellik deneyleri (CE, DI, V+Mo, Si_interaction).
- D-19: Split-conformal out-of-fold %90 güven aralığı.

## D-17: Monotone Constraints
Sıcaklık ve zaman özellikleri -1 (negatif) olarak kısıtlandı.

| Model | S1 MAE | S2 MAE |
| :--- | :-: | :-: |
| XGB_tuned (Base) | 1.052 | 2.350 |
| XGB_monotonic | 1.242 | 2.494 |

**Ekstrapolasyon Testi (Tipik 4140 Çeliği - 3600s)**
| Temp | Base Pred (HRC) | Monotonic Pred (HRC) |
| :--- | :-: | :-: |
| 750°C | 22.5 | 22.2 |
| 800°C | 22.5 | 22.2 |

## D-18: Metalurjik Özelliklerin Etkisi
Tüm aday özellikler XGB_tuned modeline tek tek eklenerek (S2 MAE) test edildi.

| Özellik | MAE | Gain (HRC) | Karar (>0.15 threshold) |
| :--- | :-: | :-: | :-: |
| *Base Model* | 2.350 | - | - |
| carbon_equivalent | 2.309 | +0.040 | NOT MATERIAL |
| simplified_DI | 2.222 | +0.128 | NOT MATERIAL |
| secondary_hardening_proxy | 2.344 | +0.006 | NOT MATERIAL |
| si_interaction | 2.348 | +0.002 | NOT MATERIAL |
| combined | 2.326 | +0.024 | NOT MATERIAL |

## D-19: Split-Conformal Güven Aralıkları
Out-of-fold %90 absolut residüel (q): **5.080 HRC**
Genel ampirik kapsama oranı: **90.0%**

**Band Bazlı Kapsama (Coverage):**
- <25 HRC: 89.7%
- 25-35 HRC: 90.2%
- 35-45 HRC: 90.6%
- >45 HRC: 89.7%