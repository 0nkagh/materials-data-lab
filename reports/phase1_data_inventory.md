# Materials Data Lab — Phase 1 Data Inventory Report

> **Salt-Okunur Veri Envanteri**: Bu rapor ham veriyi değiştirmeden oluşturulmuştur. Veri temizleme, dönüştürme, özellik mühendisliği veya modelleme içermez.

## 1. Dosya Meta Bilgileri
- **Dosya Yolu**: `C:\Users\agah\Documents\materials-data-lab\data\raw\Tempering data for carbon and low alloy steels - Raiipa.csv`
- **Boyut**: `156,258` bytes
- **SHA-256 Hash**: `3f88fdccf4bb2f3c03be6eeca03001b4bd8a2b0de5cd4818195486182a4762b0`
- **Analiz Zamanı**: `2026-09-24T10:36:57.329030+00:00`
- **Python Sürümü**: `3.12.8 (tags/v3.12.8:2dc476b, Dec  3 2024, 19:30:04) [MSC v.1942 64 bit (AMD64)]`
- **Pandas Sürümü**: `3.0.5`

## 2. Veri Boyutları ve Genel Yapı
- **Satır Sayısı**: `1466`
- **Kolon Sayısı**: `17`
- **Mükerrer Satır Sayısı (Duplicate Rows)**: `0`
- **Tekrarlanan Kolon Adları**: `Yok (tüm kolon adları benzersiz)`
- **Sabit / Tek Değerli Kolonlar**: `Yok`

### Kolon Adları Listesi (Dosyadaki Haliyle Aynen)
| # | Kolon Adı | Çıkarılan Dtype | Tip Kontrolü |
| :-: | :--- | :--- | :--- |
| 1 | `Source` | `str` | String |
| 2 | `Steel type` | `str` | String |
| 3 | `Initial hardness (HRC) - post quenching` | `str` | String |
| 4 | `Tempering time (s)` | `int64` | Numeric |
| 5 | `Tempering temperature (ºC)` | `float64` | Numeric |
| 6 | `C (%wt)` | `float64` | Numeric |
| 7 | `Mn (%wt)` | `float64` | Numeric |
| 8 | `P (%wt)` | `float64` | Numeric |
| 9 | `S (%wt)` | `float64` | Numeric |
| 10 | `Si (%wt)` | `float64` | Numeric |
| 11 | `Ni (%wt)` | `float64` | Numeric |
| 12 | `Cr (%wt)` | `float64` | Numeric |
| 13 | `Mo (%wt)` | `float64` | Numeric |
| 14 | `V (%wt)` | `float64` | Numeric |
| 15 | `Al (%wt)` | `float64` | Numeric |
| 16 | `Cu (%wt)` | `float64` | Numeric |
| 17 | `Final hardness (HRC) - post tempering` | `float64` | Numeric |

## 3. Detaylı Kolon Envanteri (Ham Veri & Belirteç Sayımları)
| Kolon Adı | Ham Dtype | Dolu Satır | Ham Eksik (NaN) | Literal '?' | Benzersiz Değer | Min | Max | Ortalama |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `Source` | `str` | 1466 | 0 | 0 | 3 | — | — | — |
| `Steel type` | `str` | 1466 | 0 | 0 | 34 | — | — | — |
| `Initial hardness (HRC) - post quenching` | `str` | 1466 | 0 | 949 | 11 | 46.5000 | 67.0000 | 61.4936 |
| `Tempering time (s)` | `int64` | 1466 | 0 | 0 | 16 | 10.0000 | 115200.0000 | 21969.7544 |
| `Tempering temperature (ºC)` | `float64` | 1466 | 0 | 0 | 21 | 100.0000 | 704.4000 | 422.0241 |
| `C (%wt)` | `float64` | 1466 | 0 | 0 | 29 | 0.2500 | 1.1500 | 0.5116 |
| `Mn (%wt)` | `float64` | 1466 | 0 | 0 | 26 | 0.3000 | 1.8500 | 0.7407 |
| `P (%wt)` | `float64` | 1466 | 0 | 0 | 21 | 0.0070 | 0.0540 | 0.0172 |
| `S (%wt)` | `float64` | 1466 | 0 | 0 | 23 | 0.0050 | 0.0550 | 0.0238 |
| `Si (%wt)` | `float64` | 1466 | 0 | 0 | 21 | 0.0600 | 1.6200 | 0.2394 |
| `Ni (%wt)` | `float64` | 1466 | 0 | 0 | 13 | 0.0000 | 3.4100 | 0.3628 |
| `Cr (%wt)` | `float64` | 1466 | 0 | 0 | 19 | 0.0000 | 1.5700 | 0.3897 |
| `Mo (%wt)` | `float64` | 1466 | 0 | 0 | 10 | 0.0000 | 0.3600 | 0.0802 |
| `V (%wt)` | `float64` | 1466 | 0 | 0 | 2 | 0.0000 | 0.1600 | 0.0055 |
| `Al (%wt)` | `float64` | 1466 | 0 | 0 | 2 | 0.0000 | 1.2600 | 0.0344 |
| `Cu (%wt)` | `float64` | 1466 | 0 | 0 | 4 | 0.0000 | 0.0800 | 0.0060 |
| `Final hardness (HRC) - post tempering` | `float64` | 1466 | 0 | 0 | 446 | 0.9000 | 68.5000 | 41.4680 |

## 4. İkincil Sayısal Dönüşüm Görünümü (Coerced Numeric View)
> Bu bölüm `pd.to_numeric(errors='coerce')` ile elde edilen ikincil analizdir. Ham veriyi DEĞİŞTİRMEZ.

| Kolon Adı | Ham Eksik | Coerce ile Yeni NaN | Toplam Coerced NaN | +Infinity | -Infinity |
| :--- | :-: | :-: | :-: | :-: | :-: |
| `Source` | 0 | 1466 | 1466 | 0 | 0 |
| `Steel type` | 0 | 1466 | 1466 | 0 | 0 |
| `Initial hardness (HRC) - post quenching` | 0 | 949 | 949 | 0 | 0 |
| `Tempering time (s)` | 0 | 0 | 0 | 0 | 0 |
| `Tempering temperature (ºC)` | 0 | 0 | 0 | 0 | 0 |
| `C (%wt)` | 0 | 0 | 0 | 0 | 0 |
| `Mn (%wt)` | 0 | 0 | 0 | 0 | 0 |
| `P (%wt)` | 0 | 0 | 0 | 0 | 0 |
| `S (%wt)` | 0 | 0 | 0 | 0 | 0 |
| `Si (%wt)` | 0 | 0 | 0 | 0 | 0 |
| `Ni (%wt)` | 0 | 0 | 0 | 0 | 0 |
| `Cr (%wt)` | 0 | 0 | 0 | 0 | 0 |
| `Mo (%wt)` | 0 | 0 | 0 | 0 | 0 |
| `V (%wt)` | 0 | 0 | 0 | 0 | 0 |
| `Al (%wt)` | 0 | 0 | 0 | 0 | 0 |
| `Cu (%wt)` | 0 | 0 | 0 | 0 | 0 |
| `Final hardness (HRC) - post tempering` | 0 | 0 | 0 | 0 | 0 |

## 5. Fiziksel Makul Aralık ve Eşik Kontrolleri (Threshold Audit)
> Bulgular veri kalitesi veya süreç anomalilerini belirlemek için bayraklandırılmıştır. Ham veri DEĞİŞTİRİLMEMİŞTİR.

### Bayrak Sayımları Özeti
- **SUSPECT Toplam Bayrak Sayısı**: `303`
  - *E140_EXTRAPOLATION_SUSPECT (<20 HRC)*: `123`
  - *HARDNESS_OUT_OF_RANGE_SUSPECT*: `123`
  - *P_UPPER_BOUND_SUSPECT (>0.050 %wt)*: `17`
  - *Al_UPPER_BOUND_SUSPECT (>0.15 %wt, Nitrasyon)*: `40`
- **WARNING Toplam Bayrak Sayısı**: `278`
  - *TEMP_BELOW_WINDOW_WARNING (100–150 °C)*: `110`
  - *TEMP_AC1_NEAR_WARNING (700–800 °C)*: `107`
  - *P_WARNING_BAND (0.040–0.050 %wt)*: `44`
  - *S_WARNING_BAND (0.050–0.060 %wt)*: `17`
- **INFO Toplam Bayrak Sayısı**: `604`
  - *TIME_SHORT_INDUCTION_INFO (<300 s)*: `258`
  - *TIME_LONG_FURNACE_INFO (>14400 s)*: `346`
  - *INFO_HIGH_ALLOY (sum > 8.0 %wt)*: `0`
- **En Az Bir Bayrak Taşıyan Satır Sayısı**: `814` / `1466`

### Kolon Bazında Bayrak Dağılımı
| Kolon | SUSPECT | WARNING | INFO |
| :--- | :-: | :-: | :-: |
| `Final hardness (HRC) - post tempering` | 246 | 0 | 0 |
| `Tempering temperature (ºC)` | 0 | 217 | 0 |
| `Tempering time (s)` | 0 | 0 | 604 |
| `P (%wt)` | 17 | 44 | 0 |
| `S (%wt)` | 0 | 17 | 0 |
| `Al (%wt)` | 40 | 0 | 0 |

## 6. Kaggle Veri Kartı Beklentileri ile Karşılaştırma
- **Satır & Kolon Sayısı**: Beklenen ~1466 satır, 17 kolon -> **Gerçek: 1466 satır, 17 kolon (Birebir uyumlu)**.
- **Kolon Adları**: Gerçek kolon adları beklentilerle karşılaştırıldığında küçük harf ve özel karakter uyumları:
  - `Steel type` dosyada küçük harf `'type'` şeklindedir (veri kartında `'Steel Type'` yazılmıştı).
  - `Tempering temperature (ºC)` başlığındaki derece işareti UTF-8 ordinal göstergesi (`º` / U+00BA) kodlamasındadır.
- **Başlangıç Sertliği Eksiklikleri**: Beklenen ~%65 oranında `?` -> **Gerçek: 949 satırda (%64.73) literal `?` mevcuttur**.
  - Tüm `?` değerleri `Grange and Baughman, 1956` alt kümesinde yer almaktadır.
- **Final Sertlik < 20 HRC (E140 Dönüşüm Şüphesi)**: Veri kartında 82 örneğin <238 HV değerinden dönüştürüldüğü not edilmişti;
  ancak dosyada `< 20 HRC` olan **123 satır** bulunmaktadır (Grange: 92, Hollomon: 21, Penha: 10).

---
*Materials Data Lab Phase 1 Envanter Raporu başarıyla tamamlanmıştır.*