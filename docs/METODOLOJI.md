# Metodoloji Notu — Makine Öğrenmesi Destekli Lojistik Anahat Optimizasyonu (MVP)

**Ekip:** AYZEK · **Aşama:** Temel İşlevli Çözüm (MVP) · **Tahmin haftası:** 11–17 Mayıs 2026

## 1. Özet Sonuçlar

| Metrik | Değer |
|---|---|
| Talep tahmin başarısı (WAPE) | **%21.1** (≈%79 isabet) |
| Tahmin başarısı (MAE) | 2.440 desi |
| Talep kapsama oranı | **%99.99** |
| Toplam operasyonel maliyet | **8.675.737,50 TL** |
| Maliyet optimalliği (ILP'ye göre) | optimalin **%2.6** içinde |

Çözüm üç katmandan oluşur: **(1) Talep Tahmini → (2) Araç Planlama → (3) Doğrulama.** Her katman izole, test edilebilir bir modüldür; `python main.py` ile uçtan uca çalışır ve üç çıktı dosyasını üretir.

## 2. Genel Mimari

```
Ham veri → [Katman 1: Tahmin] → kanonik desi tablosu ─┬→ Tahminlenen_Talep.xlsx
                                                      │
        [Mesafe matrisi (Haversine)] ────────────────┤
                                                      ▼
                              [Katman 2: Atama (kiralık→uğrama→spot)]
                                                      │
                              [Katman 3: invariant + kalite kapısı]
                                                      ▼
                              Araç_Planlama.xlsx + Karşılanamayan_Yük.xlsx
```

Tahmin tablosu **tek kanonik kaynaktır**: hem teslim dosyasına hem atama motorunun girdisine aynı sayılardan türetilir (zero-floor + sabit ondalık yuvarlama).

## 3. Katman 1 — Talep Tahmini

**Yöntem:** Tek global **LightGBM** regresyonu + **direct multi-step** kurgu + **5-seed ensemble**.

- **Tek global model:** 89 ayrı model yerine hat kimliği (çıkış+varış kategorik) bir özelliktir; az veri olan hatlar diğerlerinden bilgi paylaşımıyla öğrenir.
- **Direct multi-step (vs recursive):** Her hedef gün, sabit bir origin'den (ilk tahmin gününden bir önceki gün) ufuk *h* kadar ileriye **doğrudan** tahmin edilir; üretilen tahmin geri beslenmez. Recursive'in hata birikimini kırar. Walk-forward CV: recursive %24.1 → **direct %21.1** (her fold'da daha iyi).
- **Seed ensemble:** Aynı model 5 farklı tohumla eğitilip ortalanır; şansa bağlı gürültüyü siler (%24.8 → %24.1).
- **Özellikler:** origin-sabitli lag'ler (v0, v−1, v−6, v−7, v−13), 7/14 günlük hareketli ortalama/std, takvim (haftanın günü, ay, ayın günü, ay başı/sonu), resmi tatil/bayram bayrakları, hat kimliği.

**Metrik seçimi — WAPE (MAPE değil):** Veride bazı hat-günlerin gerçek desi'si çok küçüktür (5–50). Klasik MAPE bu küçük bölenlerde patlar (yanıltıcı %400+). Bu yüzden baş metrik **WAPE = Σ|hata|/Σgerçek** ve MAE'dir; MAPE yalnız referans olarak raporlanır.

**Doğrulama (walk-forward CV):** Genişleyen pencere; her fold, geçmişle eğitilen modelle ardışık 7 günlük bloğu üretir — gerçek 11–17 Mayıs görevini birebir taklit eder (tek-adım CV'nin iyimser hatası yaşanmaz).

**Denenip elenen yaklaşımlar (ölçülerek):** log dönüşüm (%28.9), ek feature lag28/rolling28 (%27.4), daha sade/regülarize parametreler, L2 kayıp, direct+recursive harman (5-seed'de %24.8) — hepsi baz modelden kötü çıktı; mevcut kurulum bu veri için optimale yakındır.

## 4. Katman 2 — Araç Planlama (Atama)

Sıralı greedy atama, kurallara birebir uyumlu:

1. **Kiralık araçlar (öncelikli, zorunlu):** Datasette tanımlı rota/sayı/tiple, doluluğa bakılmaksızın plana eklenir (maliyeti önceden ödenmiş). Atanan desi = gerçek yük (hayalet yük önlemi).
2. **Uğrama (multi-stop):** Bir spot aracın güzergâhında ikinci bir hattı da taşıması. Maliyet **fiilen kat edilen kümülatif mesafeden** hesaplanır (Soru-Cevap'ta kesinleşti). Sapma eşiği: kümülatif mesafe ≤ direkt × 1.15. Yalnızca ayrı araçlardan ucuzsa seçilir. **Konsolidasyon yapılmaz** (MVP'de yasak); hub/araç-arası aktarma yoktur.
3. **Spot atama (Greedy+):** Kalan yük, **desi başına en ucuz** araç tipiyle, **%10 doluluk kuralına** tabi doldurulur. Kapasiteyi aşan büyük talepler birden çok tam araca **bölünür**. %10 eşiği altında kalan küçük artık yük karşılanamayan tablosuna düşer.

**Maliyet fonksiyonu (tek kaynak):** Spot = sabit günlük maliyet + spot_km × mesafe; Kiralık = günlük kira + kiralık_km × mesafe. Tüm modüller bu tek fonksiyondan geçer.

**Mesafe:** 18 TM koordinatından **Haversine (kuş uçuşu)** 18×18 matris; düzeltme katsayısı yok, gerçek karayolu kullanılmaz.

## 5. Katman 3 — Doğrulama (Kalite Kapısı)

- **Sanity-check:** 89 hat / 130 gün / 18 TM / negatif desi yok — pipeline başlamadan veri doğrulanır.
- **Mesafe matrisi doğrulama:** 18×18, simetrik, köşegen=0, tüm değerler ≥ 0.
- **Invariant:** her hat-gün için `atanan desi + karşılanamayan desi = kanonik tahmin` (assert ile zorlanır).
- **Kalite kapısı:** tahmin ve araç planı çıktıları çapraz kontrol edilir; formatlar ve tutarlılık teslim öncesi garanti edilir.
- **ILP doğrulama valfi:** Rota-tabanlı set-covering ILP, greedy atamanın optimumdan uzaklığını ölçer → greedy, optimalin **%2.6** içinde (greedy yaklaşımının yeterliliğini kanıtlar).

## 6. Sonuç ve Kural Uyumu

Çözüm tüm MVP kısıtlarını sağlar: spot %10 doluluk, kiralık zorunlu+öncelikli, kuş uçuşu mesafe, uğrama serbest / konsolidasyon yok, dönüş rotası kapsam dışı, her gün bağımsız. Tahmin başarısı talebin doğal hafta-üstü oynaklığının (%33) altındadır; atama optimale yakındır; talebin %99.99'u taşınır. Sistem hem doğru hem kural-uyumlu hem de optimale yakın bir MVP teslimidir.
