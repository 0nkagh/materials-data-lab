# Materials Data Lab — Phase 3: Sayısal Keşifsel Veri Analizi (EDA)

> **NOT**: Bu rapordaki korelasyon bulguları nedensellik belirtmez. Yalnızca istatistiksel gözlemlerdir.
> Analiz temiz yükleme katmanı (clean_loader) kullanılarak türetilmiş görünüm üzerinde gerçekleştirilmiştir.

## 1. Final HRC Dağılımı
**Genel**:
- n=1466, Min=0.90, P25=32.10, Median=43.10, P75=51.80, Max=68.50, Mean=41.47, Std=14.08
**Kaynak Bazında**:
- Grange and Baughman, 1956: n=949, Min=0.90, P25=29.70, Median=39.20, P75=48.00, Max=62.70, Mean=37.95, Std=12.33
- Hollomon and Jaffe, 1945: n=270, Min=1.50, P25=37.50, Median=50.50, P75=62.25, Max=68.50, Mean=47.82, Std=16.49
- Penha, 2010: n=247, Min=7.30, P25=40.75, Median=51.60, P75=58.00, Max=64.00, Mean=48.06, Std=12.70

## 2. Sıcaklık ve Zaman Dağılımları
### Tempering Temperature (°C)
**Genel**: n=1466, Min=100.00, P25=260.00, Median=426.70, P75=593.30, Max=704.40, Mean=422.02, Std=176.09
- Grange and Baughman, 1956: n=949, Min=204.40, P25=315.60, Median=482.20, P75=593.30, Max=704.40, Mean=456.58, Std=153.91
- Hollomon and Jaffe, 1945: n=270, Min=100.00, P25=200.00, Median=300.00, P75=500.00, Max=700.00, Mean=355.70, Std=196.86
- Penha, 2010: n=247, Min=100.00, P25=200.00, Median=300.00, P75=500.00, Max=700.00, Mean=361.74, Std=194.71
### Tempering Time (s)
**Genel**: n=1466, Min=10.00, P25=600.00, Median=3600.00, P75=14400.00, Max=115200.00, Mean=21969.75, Std=34177.62
- Grange and Baughman, 1956: n=949, Min=40.00, P25=900.00, Median=3600.00, P75=28800.00, Max=115200.00, Mean=23862.51, Std=34653.87
- Hollomon and Jaffe, 1945: n=270, Min=10.00, P25=90.00, Median=900.00, P75=9000.00, Max=86400.00, Mean=19280.00, Std=33791.51
- Penha, 2010: n=247, Min=10.00, P25=90.00, Median=900.00, P75=9000.00, Max=86400.00, Mean=17637.81, Std=32239.36

## 3. Kompozisyon Özetleri (%wt)
| Element | Min | Median | Max |
| :--- | :-: | :-: | :-: |
| c_wt | 0.2500 | 0.4200 | 1.1500 |
| mn_wt | 0.3000 | 0.7400 | 1.8500 |
| p_wt | 0.0070 | 0.0170 | 0.0540 |
| s_wt | 0.0050 | 0.0240 | 0.0550 |
| si_wt | 0.0600 | 0.2100 | 1.6200 |
| ni_wt | 0.0000 | 0.0100 | 3.4100 |
| cr_wt | 0.0000 | 0.0600 | 1.5700 |
| mo_wt | 0.0000 | 0.0000 | 0.3600 |
| v_wt | 0.0000 | 0.0000 | 0.1600 |
| al_wt | 0.0000 | 0.0000 | 1.2600 |
| cu_wt | 0.0000 | 0.0000 | 0.0800 |

## 4. Spearman Korelasyonları (final_hrc ile)
Gözlem: Sıcaklık ve log10(zaman) ile final HRC arasındaki korelasyonlar.
| Kaynak | vs Sıcaklık | vs log10(Zaman) |
| :--- | :-: | :-: |
| **Genel** | -0.8893 | -0.2434 |
| Grange and Baughman, 1956 | -0.8961 | -0.0754 |
| Hollomon and Jaffe, 1945 | -0.8500 | -0.2589 |
| Penha, 2010 | -0.8242 | -0.4331 |

## 5. Keşifsel Hollomon-Jaffe Korelasyonu
Hesaplama: `p_hj = (T_K / 1000) * (19.5 + log10(t_saat))`
> **C=19.5 ASSUMED [REPORT_ONLY]** - Modelleme değildir.
| Kaynak | vs P_HJ (Spearman) |
| :--- | :-: |
| **Genel** | -0.9285 |
| Grange and Baughman, 1956 | -0.9236 |
| Hollomon and Jaffe, 1945 | -0.8908 |
| Penha, 2010 | -0.8944 |

## 6. Eksiklik Deseni (Initial HRC Missing)
Gözlem: Hangi kaynakta `initial_hrc` hücresi eksiktir.
**Missing: False**
- Grange and Baughman, 1956: 0
- Hollomon and Jaffe, 1945: 270
- Penha, 2010: 247
**Missing: True**
- Grange and Baughman, 1956: 949
- Hollomon and Jaffe, 1945: 0
- Penha, 2010: 0

## 7. Bayrak Tutarlılık Kontrolü
Gözlem: Phase 1 ile sayıların tutarlılığı.
- Beklenen: SUSPECT 303 / WARNING 278 / INFO 604 / E140 123
- Gerçekleşen: SUSPECT 303 / WARNING 278 / INFO 604 / E140 123
- **Sonuç**: Tüm sayılar Phase 1 ile birebir eşleşiyor.