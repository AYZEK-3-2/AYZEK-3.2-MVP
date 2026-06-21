# AYZEK 3.2 — MVP Lojistik Optimizasyon Sistemi

**TEKNOFEST 2026 — Makine Öğrenmesi Destekli Lojistik Anahat Optimizasyonu**  
*Dataset A · Tahmin + Atama Hibrit Çözümü*

---

## Genel Bakış

Bu proje, 18 transfer merkezi (TM) ve 89 sabit çıkış→varış hattı üzerinde çalışan bir **linehaul (middle-mile) operasyonunu** optimize eder. Sistem iki temel soruyu birden cevaplar:

1. **Tahmin:** Önümüzdeki 7 gün için her hatta kaç desi yük taşınacak?
2. **Atama:** Bu yükü en düşük maliyetle hangi araçlara nasıl dağıtalım?

### Mimari — Üç Katman

```
Ham Veri (130 gün)
       │
       ▼
┌─────────────────────────────────┐
│  Katman 1: LightGBM Tahmini    │  → 7 günlük desi tahmini (yansız, zero-floor)
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Katman 2: Atama Motoru         │
│  ① Kiralık araç yerleştirme    │  → Zorunlu, doluluğa bakılmaksızın
│  ② Uğrama değerlendirmesi      │  → Yalnız spot, maliyet kıyasıyla
│  ③ Greedy+ spot atama          │  → %10 doluluk kuralı burada
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Katman 3: ILP Doğrulama       │  → Rota-tabanlı optimum kanıtı
└─────────────────────────────────┘
       │
       ▼
Çıktı: Tahminlenen_Talep.xlsx
       Araç_Planlama.xlsx
       Karşılanamayan_Yük.xlsx
```

---

## Dosya Yapısı

```
AYZEK-3.2-PROJECT/
│
├── data/
│   ├── raw/                          # Ham xlsx dosyaları (repoya dahil, ASCII isimli)
│   │   ├── Desi_talep.xlsx           # 10.770 satır: 89 hat × 130 gün
│   │   ├── Koordinatlar_v2.xlsx      # 18 TM koordinatı
│   │   ├── Arac_Kapasite_Maliyet.xlsx
│   │   └── Kiralik_Araclar.xlsx      # 14 kiralık araç, 12 hat
│   └── processed/                    # Ara çıktılar (runtime'da üretilir)
│       └── distance_matrix.csv       # 18×18 Haversine mesafe matrisi
│
├── src/
│   ├── config.py                     # Tüm sabitler, dosya yolları, araç parametreleri
│   ├── data_loader.py                # Ham veri okuma ve standartlaştırma
│   ├── output_writer.py              # xlsx üreticisi — format, stil, toplam maliyet satırı
│   │
│   ├── layer1_prediction/
│   │   ├── feature_engineering.py   # Lag, rolling ortalama, takvim, tatil özellikleri
│   │   ├── model.py                  # LightGBM eğitimi + walk-forward CV
│   │   └── predict.py                # Direct multi-step tahmin (5-seed ensemble)
│   │
│   ├── layer2_assignment/
│   │   ├── distance.py               # Haversine matrisi + kümülatif rota mesafesi
│   │   ├── rental_placer.py          # Kiralık araç zorunlu yerleştirme
│   │   ├── waypoint.py               # Uğrama aday üretimi ve maliyet kıyası
│   │   └── spot_assigner.py          # Greedy+ spot atama + artık yük yönetimi
│   │
│   └── layer3_validation/
│       ├── ilp_validator.py          # Rota-tabanlı ILP doğrulama valfi (PuLP)
│       ├── invariant.py              # atanan + karşılanamayan = tahmin kontrolü
│       └── quality_gate.py           # Teslim öncesi çapraz format doğrulama
│
├── output/                           # Teslim dosyaları (pipeline sonrası oluşur)
├── notebooks/                        # EDA ve deney notebook'ları
│
├── main.py                           # Uçtan uca pipeline çalıştırıcı
├── docs/METODOLOJI.md / .pdf         # Jüri için metodoloji notu
└── requirements.txt
```

---

## Kurulum

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt
```

Ham veri dosyaları repoya dahildir (`data/raw/`); ayrı bir kopyalama adımı gerekmez.
Klonlayıp doğrudan `python main.py` çalıştırabilirsiniz.

**Gereksinimler:** Python 3.10+, pandas, numpy, lightgbm, scikit-learn, openpyxl, pulp

---

## Çalıştırma

```bash
# Tam pipeline (tahmin + atama + çıktı)
python main.py

# Cross-validation ile (birkaç dakika sürer)
python -c "from main import run_pipeline; run_pipeline(run_cv=True)"

# ILP doğrulama valfi aktif
python -c "from main import run_pipeline; run_pipeline(run_ilp=True)"
```

Pipeline sırası `main.py` içinde 10 adım olarak belgelenmiştir.

---

## Veri Seti

| Dosya | Boyut | İçerik |
|---|---|---|
| Desi_talep | 10.770 satır | 89 O→D hattı × 130 gün (01.01–10.05.2026) |
| Koordinatlar_v2 | 18 satır | TM enlem/boylam |
| Araç_Kapasite_Maliyet | 4 satır | Tır, Kamyon, Hafif Kamyon, Kamyonet |
| Kiralık_Araçlar | 12 satır | 14 kiralık araç (8 Tır + 4 Kamyon) |

**Tahmin hedefi:** 11–17 Mayıs 2026 (7 gün)

---

## Araç Parametreleri

| Araç | Kapasite (desi) | Kira/gün | Kira km | Spot sabit | Spot km |
|---|---|---|---|---|---|
| Tır | 22.400 | 7.000 TL | 13 TL | 11.700 TL | 25 TL |
| Kamyon | 12.000 | 5.000 TL | 10 TL | 7.638 TL | 21 TL |
| Hafif Kamyon | 7.200 | 5.000 TL | 10 TL | 8.750 TL | 20 TL |
| Kamyonet | 5.600 | 3.750 TL | 6 TL | 4.750 TL | 18 TL |

---

## Temel Tasarım Kararları

### Tahmin (Katman 1)
- **Tek global LightGBM modeli** — 89 hat için ayrı model yerine hat kimliği feature olarak verilir
- **Direct multi-step + 5-seed ensemble:** Her hedef gün sabit bir origin'den doğrudan tahmin edilir; tahmin geri beslenmez (recursive'in hata birikimi olmaz). CV WAPE ~%21.1
- **Zero-floor:** `max(0, tahmin)` — negatif desi atama motorunu çökertmez
- **Kanonik yuvarlama:** tahmin üretilince 2 haneye sabitlenir; hem çıktı dosyası hem atama aynı değerden beslenir

### Atama (Katman 2)
- **Katman sırası zorunludur:** kiralık → uğrama → spot
- **Hayalet yük koruması:** atanan desi = gerçek yük, araç kapasitesi değil
- **Uğrama yalnız spot havuzunda** — kiralık araçlar tanımlı rotasında çalışır (kural ihlali önlemi)
- **Pik yük araç seçimi:** araç son bacağa göre değil rotanın maksimum anlık yüküne göre seçilir
- **Kümülatif bacak mesafesi:** uğramalı rotada maliyet doğrudan O→D değil, tüm bacakların Haversine toplamıdır
- **Büyük talep bölme:** en büyük araç kapasitesini (22.400 desi) aşan yük birden çok tam araca bölünür (atılmaz) → kapsama %99.99
- **Artık yük terminali:** yalnızca %10 eşiği (560 desi) altındaki küçük artık yük atanmaz, unfulfilled tablosuna yazılır

### Doğrulama (Katman 3)
- **Rota-tabanlı ILP** (set-partitioning): uğramalı rotaları da içerdiği için gerçek optimum kanıtı sağlar
- Yalnızca örnek günlerde çalıştırılır; günlük iş için Greedy+ yeterlidir

---

## İnvariant Kontrolü

Her hat-gün için `main.py` şu eşitliği doğrular:

```
atanan_desi + karşılanamayan_desi = kanonik_tahmin
```

İhlal varsa pipeline uyarı verir ve teslim öncesi yakalanır.

---

## Çıktı Formatı

### Tahminlenen_Talep.xlsx
| Tarih | Çıkış TM | Varış TM | Tahmin Edilen Desi |
|---|---|---|---|
| 2026-05-11 | İstanbul | Ankara TM | 1250.50 |

### Araç_Planlama.xlsx
| Tarih | Araç Tipi | Çıkış TM | Varış TM | Güzergah | Atanan Desi | Maliyet |
|---|---|---|---|---|---|---|
| 2026-05-11 | Kiralık Tır | İstanbul | Yalova | İstanbul→Yalova | 4800.00 | 7606.00 |

### Karşılanamayan_Yük.xlsx
| Tarih | Çıkış TM | Varış TM | Karşılanamayan Desi | Sebep |
|---|---|---|---|---|
| 2026-05-11 | Mersin | Kütahya | 420.00 | %10 spot eşiği altında |

---

## Mimari Doküman

Detaylı sistem tasarımı, kısıt gerekçeleri ve risk matrisi için:
`docs/MVP_Sistem_Mimarisi_v5.md`
