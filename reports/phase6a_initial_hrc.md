# Phase 6a: initial_hrc Alt-Küme Çalışması

**Alt Küme Doğrulaması**: Toplam 517 satır. Kaynak kırılımı: {"Hollomon and Jaffe, 1945": 270, "Penha, 2010": 247}. Grup (steel_type) sayısı: 11. (GroupKFold 5 kat ile çalıştırıldı).

## Sonuçlar (Out-of-Fold, S2)
| Model | MAE (mean ± std) | RMSE (mean ± std) | R² (mean ± std) |
| :--- | :--- | :--- | :--- |
| B2 (Fizik, p_hj) | 5.37 ± 2.14 | 6.54 ± 2.24 | 0.74 ± 0.22 |
| M1 Set A (11 Comp + Temp + Time) | 4.02 ± 2.21 | 4.91 ± 2.53 | 0.83 ± 0.20 |
| M1 Set B (Set A + initial_hrc) | 3.53 ± 2.14 | 4.45 ± 2.41 | 0.86 ± 0.17 |

## Gözlem
initial_hrc özelliği eklendiğinde (Set B), modele aynı veride görülmemiş çelikler üzerinde MAE'nin değişip değişmediği gözlemsel olarak tespit edilmiştir (Tabloya bakınız; modest stability improvement gözlenmiştir).

## ZORUNLU CAVEAT BÖLÜMÜ
1. **Seçim Yanlılığı**: Bu alt küme, initial_hrc bilgisi eksik olan TÜM Grange (1956) verisini dışlar. Rastgele bir örneklem DEĞİLDİR.
2. **Kaynak Sınırlaması**: Yalnızca 2 kaynak (Hollomon ve Penha) içerir.
3. **Küçük Grup Sayısı**: Bu alt kümede yalnızca 11 farklı çelik tipi (grup) mevcuttur.
4. **Gözlemsel Limitasyon**: Sonuçlar tamamen gözlemseldir; farklı bir veri alt kümesinde eğitildiklerinden ötürü MVP (Faz 4) metrikleriyle doğrudan kıyaslanamaz.