# Data Dictionary (Draft) — Materials Data Lab

Bu veri sözlüğü, `Tempering data for carbon and low alloy steels - Raiipa.csv` veri setindeki her sütunun tanımını, beklenen birimini, referans aralıklarını ve köken (provenance) sınıflandırmasını belgeler. Kaynaktan doğrudan doğrulanamayan tüm alanlar katı kural gereği `UNKNOWN` olarak bırakılmıştır; hiçbir alanda yapay veri uydurulmamıştır.

---

## 1. Köken (Provenance) Taksonomisi
- **`STANDARD_CROSSCHECKED`**: Resmi teknik standartlar (ASTM, ISO, SAE) veya hakemli literatür çapraz kontrolleri ile doğrulanmış bilgi.
- **`REPORT_ONLY`**: Tekil literatür derlemesi, veri kartı açıklaması veya heuristik kural; bağımsız tam metin standart çapraz kontrolü yapılmamış.
- **`UNKNOWN`**: Mevcut veri ve dokümantasyondan kesin olarak tespit edilemeyen veya doğrulanması mümkün olmayan bilgi.

---

## 2. Sütun Tanımları ve Özellikleri

*(Not: Tablodaki kimyasal kompozisyon "Referans Aralıkları" SAE J403/J404 tarzı tipik spec bağlamını gösterir `[REPORT_ONLY]`; bu aralıklar dışındaki uç değerler veri kalitesi amacıyla "bayrak kuralları" ile etiketlenir.)*

| Kolon Adı (Dosyadaki Haliyle) | Tahmini Anlam | Birim | Referans Aralık | Köken (Provenance) | Notlar |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Source` | Verinin alındığı birincil literatür çalışması | `UNKNOWN` | Grange & Baughman (1956), Hollomon & Jaffe (1945), Penha (2010) | `REPORT_ONLY` | Kategorik metin alanı. Penha (2010) künyesi belirsizdir (`docs/references.md`). |
| `Steel type` | Çelik kalite standardı veya alaşım kodu | `UNKNOWN` | AISI-SAE serisi (34 benzersiz tip) | `STANDARD_CROSSCHECKED` | SAE J403/J404 standart alaşım kodlamaları (örn. AISI-SAE 1045, 4140, 4340). |
| `Initial hardness (HRC) - post quenching` | Su verme sonrası, menevişleme öncesi başlangıç sertliği | `HRC` | 20.0 – 70.0 HRC | `STANDARD_CROSSCHECKED` | ASTM E18 / ISO 6508 standardı. Grange (1956) kaynaklı 949 satırda (%64.7) literal `?` olarak mevcuttur. Hollomon ve Penha kaynaklarında 46.5–67.0 HRC aralığındadır. |
| `Tempering time (s)` | Meneviş fırınında tutma / izotermal bekleme süresi | `s` | > 0 s (Tipik fırın rejimi: 300 – 14400 s) | `REPORT_ONLY` | 10 s ile 115200 s (32 saat) arası değerler. <300 s (kısa/endüksiyon rejimi, 258 satır); >14400 s (uzun rejim, 346 satır). |
| `Tempering temperature (ºC)` | Menevişleme sıcaklığı | `ºC` | Tipik pencere: 150 – 700 ºC (Fiziksel sınır: >0 ve ≤800 ºC) | `STANDARD_CROSSCHECKED` | Kolon başlığındaki derece karakteri ordinal gösterge (`º`, U+00BA) olarak kodlanmıştır. 100–150 ºC (pencere altı) ve 700–800 ºC (Ac1 yakını) değerleri içerir. |
| `C (%wt)` | Karbon kütlesel yüzdesi | `%wt` | 0.0 – 1.20 %wt | `REPORT_ONLY` | SAE J403/J404 heuristiği. Çelikte martenzit sertliğini ve sertleşebilirliği belirleyen temel alaşım elementi. |
| `Mn (%wt)` | Mangan kütlesel yüzdesi | `%wt` | 0.0 – 2.10 %wt | `REPORT_ONLY` | SAE J403/J404 heuristiği. Sertleşebilirlik ve kükürt bağlama etkisi. |
| `P (%wt)` | Fosfor kütlesel yüzdesi | `%wt` | ≤ 0.050 %wt (0.040–0.050 WARNING bandı) | `REPORT_ONLY` | SAE J403/J404 heuristiği. Kalıntı/empürite elementi. >0.050 %wt (17 satır) SUSPECT, 0.040–0.050 %wt (44 satır) WARNING. |
| `S (%wt)` | Kükürt kütlesel yüzdesi | `%wt` | ≤ 0.060 %wt (0.050–0.060 WARNING bandı) | `REPORT_ONLY` | SAE J403/J404 heuristiği. Kalıntı elementi. 0.050–0.060 %wt (17 satır) WARNING bandında. |
| `Si (%wt)` | Silisyum kütlesel yüzdesi | `%wt` | 0.0 – 2.50 %wt | `REPORT_ONLY` | SAE J403/J404 heuristiği. Deoksidasyon ve katı eriyik sertleşmesi elementi; temper gevrekleşmesini geciktirici etki. |
| `Ni (%wt)` | Nikel kütlesel yüzdesi | `%wt` | 0.0 – 5.00 %wt | `REPORT_ONLY` | SAE J403/J404 heuristiği. Tokluk ve derin sertleşebilirlik artırıcı. |
| `Cr (%wt)` | Krom kütlesel yüzdesi | `%wt` | 0.0 – 2.50 %wt | `REPORT_ONLY` | SAE J403/J404 heuristiği. Karbür oluşturucu ve sertleşebilirlik elementi. |
| `Mo (%wt)` | Molibden kütlesel yüzdesi | `%wt` | 0.0 – 1.00 %wt | `REPORT_ONLY` | SAE J403/J404 heuristiği. İkincil sertleşme ve temper gevrekleşmesini önleme. |
| `V (%wt)` | Vanadyum kütlesel yüzdesi | `%wt` | 0.0 – 0.50 %wt | `REPORT_ONLY` | SAE J403/J404 heuristiği. Güçlü karbür oluşturucu ve tane inceltici. |
| `Al (%wt)` | Alüminyum kütlesel yüzdesi | `%wt` | 0.0 – 0.15 %wt | `REPORT_ONLY` | SAE J403/J404 heuristiği. >0.15 %wt (40 satır) SUSPECT; nitrasyon çeliği (nitralloy) varlığı şüphesi notu. |
| `Cu (%wt)` | Bakır kütlesel yüzdesi | `%wt` | 0.0 – 0.40 %wt | `REPORT_ONLY` | SAE J403/J404 heuristiği. Korozyon ve hava direnci elementi. |
| `Final hardness (HRC) - post tempering` | Menevişleme sonrası ölçülen final sertlik | `HRC` | 20.0 – 70.0 HRC | `STANDARD_CROSSCHECKED` | ASTM E18 / ISO 6508 standardı. <20 HRC olan 123 satır mevcuttur; bu satırlar ASTM E140 Tablo 1'e göre 238 HV altından matematiksel dönüştürme şüphesiyle `E140_EXTRAPOLATION_SUSPECT` olarak işaretlenir. Note: the dataset card records no HV->HRC conversions for Hollomon & Jaffe (1945), so the lowest-HRC rows in that subset are plausibly true measurements; the <20 HRC region should be read as UNKNOWN-origin (conversion or measurement), consistent with Phase 2 decision D-02. |

---

## 3. Kapsam Dışı Bağlam Notları (Contextual Metallurgical Notes)
*(Bu notlar modelleme veya filtreleme amacıyla değil, metalurjik alan bilgisini belgelemek üzere sunulmuştur; köken: `REPORT_ONLY`)*

1. **Temperleme Evreleri (Tempering Stages)**:
   - **Evre 1 (~100–200 ºC)**: Aşırı doymuş tetragonal martenzitten geçiş karbürü (epsilon/eta karbür, $\text{Fe}_{2.4}\text{C}$) çökelmesi ve karbon oranının matriste ~%0.25 C civarına düşmesi.
   - **Evre 2 (~200–300 ºC)**: Su verme sonrası yapıda kalmış kararsız kalıntı östenitin (retained austenite) beynite veya ferrit + sementite dönüşümü.
   - **Evre 3 (~250–350 ºC)**: Geçiş karbürlerinin ve matrisin kararlı sementit ($\text{Fe}_3\text{C}$) fazına dönüşmesi, tetragonal yapının KHM ferrite gevşemesi.
   - **Evre 4 (>400 ºC)**: Sementit taneciklerinin irileşmesi/küreselleşmesi (coarsening/spheroidization). Karbür oluşturucu alaşım elementleri (Cr, Mo, V) içeren çeliklerde ikincil sertleşme (secondary hardening) karbürlerinin çökelmesi.

2. **Gevrikleşme Bölgeleri (Embrittlement Ranges)**:
   - **Temper Martensite Embrittlement (TME / 350 ºC Gevrekliği / Blue Brittleness)**: ~250–400 ºC aralığında temperlenen çeliklerde görülen tokluk düşüşü (tane sınırlarında karbür filmi çökelmesi).
   - **Temper Embrittlement (TE / İki Basamaklı Gevrelik)**: ~350–575 ºC aralığında uzun süre tutma veya bu aralıktan yavaş soğuma sırasında kalıntı elementlerin (P, Sb, Sn, As) eski östenit tane sınırlarına segregasyonu sonucu oluşan gevrelik.
   - *Not*: Veri setinde Charpy darbe enerjisi veya tokluk ölçümü bulunmamaktadır; bu bölgeler sertlik ölçümlerinde süreksizlik yaratmaz, ancak mekanik tasarım bağlamı için önemlidir.

3. **Su Verme Sertliği - Karbon İlişkisi (As-Quenched Hardness)**:
   - Su verilmiş martenzitik çeliklerin maksimum sertliği esas olarak çözünmüş karbon yüzdesi tarafından belirlenir (Hodge & Orehoski 1946 modelleri). Düşük alaşım elementleri martenzit sertliğini doğrudan artırmaktan ziyade kritik soğuma hızını düşürerek sertleşebilirliği (hardenability) artırır.

---

## 4. Steel Class System (`steel_class`)
Proje kapsamında, `Steel type` sütunu standart SAE etiketlerine dayanılarak daha üst seviye çelik sınıflarına (steel_class) dönüştürülür `[STANDARD_CROSSCHECKED]`. Bilinmeyen formattaki isimler `UNKNOWN_CLASS` olarak işaretlenir.

| Çelik Sınıfı (steel_class) | SAE/AISI Kodu | Tanım |
| :--- | :--- | :--- |
| **plain_carbon** | 10xx | Sade karbon çelikleri |
| **Mn_steel** | 13xx | Mangan alaşımlı çelikler |
| **Mo_steel** | 40xx | Molibden alaşımlı çelikler |
| **Cr_Mo** | 41xx | Krom-Molibden alaşımlı çelikler |
| **Ni_Cr_Mo** | 43xx, 86xx, 87xx | Nikel-Krom-Molibden alaşımlı çelikler |
| **Cr_steel** | 51xx, 52xx | Krom alaşımlı çelikler |
| **Cr_V** | 61xx | Krom-Vanadyum alaşımlı çelikler |
| **Si_steel** | 92xx | Silisyum alaşımlı çelikler |
| **nitriding** | Nitriding Steel | Yüzey sertleştirme (nitrasyon) için özel alaşım |
| **UNKNOWN_CLASS** | - | Standart dışı, eksik veya eşleşmeyen kodlar |

Note: Si-alloy steels (e.g. 9260-type) show the highest prediction error for both models, consistent with Si delaying cementite precipitation during tempering (REPORT_ONLY observation).
