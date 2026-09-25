# Materials Data Lab — Phase 5: Değerlendirme Derinliği ve Model Kartı

## Faz 1-4 Özeti
- **Phase 1-2** (`69dadb4`): Proje iskeleti oluşturuldu, veri envanteri ve kural bazlı kalite bayrakları (breakdown) yapıldı.
- **Phase 3-4** (`9f6689d`): Onaylanmış karar günlüğü ile temiz veri yükleyici (clean_loader) geliştirildi; özellik mühendisliği (log_time) ve S1/S2 ayrım stratejileriyle modelleme MVP'si oluşturuldu.

## Model Kartı
- **Veri Kaynağı**: Kaggle üzerinden Raiipa Technologies (CC BY 4.0), literatür temelli (Grange, Hollomon, Penha).
- **Özellikler**: 11 bileşim kolonu, temperleme sıcaklığı, log10(temperleme zamanı).
- **Stratejiler**: S1 (RandomSplit - naif performans) ve S2 (GroupSplit - dış çelik genellemesi). S2 daha gerçekçidir.
- **Metrikler**: RF için S2 R²=0.93, MAE=2.69. Fizik baseline (B2) için S2 R²=0.81, MAE=4.67.
- **Limitasyonlar**: 
  1. (i) tek veri kaynağı kümesi — genellenebilirlik sınırlı,
  2. (ii) C=19.5 ASSUMED [REPORT_ONLY],
  3. (iii) initial_hrc hariç tutuldu (D-08),
  4. (iv) MDI korele-feature uyarısı,
  5. (v) Tuning (V2-C1): `RandomizedSearchCV` ile yapılan hiperparametre araması sonucunda kazanım <0.1 HRC olmuştur (MAE: 2.687 → 2.639; kazanç 0.048 HRC). Default MVP parametrelerinin halihazırda yeterli olduğu raporlanmış olup, `demo.py` default RF modeliyle hizmet vermeye devam etmektedir. Tuning yapılması modelin MDI/Permutation sıralamasını (özellik önemini) anlamlı ölçüde değiştirmemiştir.
  6. (vi) rezidüeller kaynaklar arasında dengesiz olabilir (Grange kaynağında hata 2.32 iken, Hollomon ve Penha kaynaklarında hata 3.35+ seviyesindedir).
  7. Permutation importance cross-check (Faz 6a) eklendi; sonuç özeti.

### Challenger Study (V2-C2)
- XGBoost, Random Forest şampiyonluğuna karşı test edilmiştir. XGBoost Tuned modeli, RF Tuned modeline kıyasla GroupKFold altında daha düşük ortalama MAE elde etmiş (S2 MAE: 2.35 ± 0.39 vs 2.64 ± 0.24) ve D-12 eşiğini (>0.15) aşarak yeni şampiyon olmuştur.
- **SHAP Gözlemi**: XGBoost şampiyon modelinin SHAP analizi, MDI/Permutation sıralamasıyla genel uyum göstermektedir. Sıcaklık ve zaman özellikleri hala en dominatör olup, alaşımlar ikincil etkiye sahiptir.
- **Demo Notu**: The demo serves XGB_tuned as of V2-F.
### Physics Evolution Study (V2-E)
- Kang & Lee (2014) yaklaşımından esinlenilerek Hollomon-Jaffe C sabiti bileşimden öğrenildi (Learned-C Composite, D-13). Bu model, klasik B2 modeline göre S2 MAE hatasını ~4.67'den ~3.34 HRC'ye düşürdü.
- Klasik p_hj özelliğinin şampiyon XGBoost modeline (XGB_tuned) eklenmesi S2 MAE'de <0.1 HRC (0.046 HRC) kazanç sağladı. Bu marjinal kazanç, D-14 kuralı (şampiyon güncelleme eşiği) gereği model değişikliğini tetiklememiştir.

### Physics-Guided Study & Conformal Intervals (V3-A)
- **Monotonic Constraints (D-17)**: XGBoost modelinde sıcaklık ve zamana monotonik olarak azalan kısıt eklendiğinde performans S2'de (2.35'ten 2.49'a) düştü ve ekstrapolasyonda ek bir iyileşme sağlamadı. Bu yüzden modelden dışlandı.
- **Metalurjik Özellikler (D-18)**: Karbon Eşdeğeri (CE), İdeal Kritik Çap log-toplamı (DI), İkincil Sertleşme (V+Mo) ve Si-Sıcaklık etkileşim özellikleri modele ayrı ayrı beslendi. Hiçbiri >0.15 HRC barajını aşamadı (en iyisi DI proxy ile 0.12 HRC kazançtı). Şampiyon model özellik değiştirmeden yoluna devam etti.
  > Note: simplified DI reached 0.128 HRC gain — below the pre-registered 0.15 threshold; not adopted to avoid threshold-shopping (D-12/D-18 consistency).
- **Split-Conformal Güven Aralıkları (D-19)**: Kullanıcılara nokta tahmini yerine kalibre edilmiş belirsizlik vermek amacıyla out-of-fold mutlak residüellerin %90'lık quantile değeri hesaplandı. Model, tüm sertlik bantlarında homojen ve tutarlı bir şekilde hedeflenen ~%90 kapsama oranına (overall 89.9%) **±5.08 HRC** güven aralığı bandı ile ulaştı.


## Sonuçların Doğru Okunması
> Korelasyon nedensellik ifade etmez. Ayrıca S1'deki (Random Split) R² skorları aynı çelik türünün hem eğitim hem teste sızmasından ötürü **iyimserdir**. S2 skoru daha güvenilirdir.

## MAE Kırılımları (Out-of-Fold, S2 GroupKFold)

### Kaynağa Göre MAE
| Kaynak | RF MAE (n) | B2 MAE (n) |
| :--- | :--- | :--- |
| Grange and Baughman, 1956 | 2.32 (949) | 4.35 (949) |
| Hollomon and Jaffe, 1945 | 3.38 (270) | 5.72 (270) |
| Penha, 2010 | 3.35 (247) | 4.82 (247) |

### HRC Bandına Göre MAE
| HRC Bandı | RF MAE (n) | B2 MAE (n) |
| :--- | :--- | :--- |
| <25 | 3.27 (203) | 6.43 (203) |
| 25-40 | 2.96 (428) | 4.07 (428) |
| 40-55 | 2.27 (580) | 4.54 (580) |
| >=55 | 2.72 (255) | 4.64 (255) |

### Sıcaklık Bandına Göre MAE
| Sıcaklık Bandı (°C) | RF MAE (n) | B2 MAE (n) |
| :--- | :--- | :--- |
| <300 | 2.34 (393) | 4.31 (393) |
| 300-500 | 2.38 (487) | 4.21 (487) |
| 500-650 | 3.15 (479) | 5.08 (479) |
| >=650 | 3.34 (107) | 6.40 (107) |

## RF (S2) En Kötü 10 Tahmin (Out-of-Fold)
| Steel Type | Kaynak | Temp (°C) | Time (s) | Gerçek | Tahmin | Rezidüel |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0,98%C - plain carbon steel | Hollomon and Jaffe, 1945 | 602.0 | 86400 | 4.5 | 24.0 | 19.5 |
| 0,31%C - plain carbon steel | Hollomon and Jaffe, 1945 | 150.0 | 90 | 46.0 | 59.7 | 13.7 |
| 0,31%C - plain carbon steel | Hollomon and Jaffe, 1945 | 700.0 | 10 | 21.5 | 34.6 | 13.1 |
| 0,31%C - plain carbon steel | Hollomon and Jaffe, 1945 | 700.0 | 90 | 16.0 | 28.7 | 12.7 |
| AISI-SAE E52100 | Penha, 2010 | 600.0 | 90 | 51.7 | 39.1 | -12.6 |
| AISI-SAE 5160 | Penha, 2010 | 500.0 | 86400 | 25.0 | 37.6 | 12.6 |
| AISI-SAE 9264 | Grange and Baughman, 1956 | 648.9 | 86400 | 26.3 | 13.8 | -12.5 |
| 0,31%C - plain carbon steel | Hollomon and Jaffe, 1945 | 100.0 | 10 | 47.5 | 59.7 | 12.2 |
| AISI-SAE E52100 | Penha, 2010 | 600.0 | 900 | 47.2 | 35.0 | -12.2 |
| 0,31%C - plain carbon steel | Hollomon and Jaffe, 1945 | 600.0 | 90 | 22.5 | 34.5 | 12.0 |

**Gözlem**: B2'nin (fiziksel baseline) en zayıf olduğu bölgeler ile RF modelinin zorlandığı bölgeler (genellikle uç sertlik/sıcaklık değerleri) paralellik göstermektedir.