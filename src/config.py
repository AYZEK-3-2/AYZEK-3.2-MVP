# -*- coding: utf-8 -*-
"""
Proje genelinde kullanılan sabitler, dosya yolları ve parametreler.
Tek kaynak: tüm modüller buradan import eder.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Dizin yolları
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR   = PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# Ham veri dosyaları
# ---------------------------------------------------------------------------
RAW_FILES = {
    "desi_talep":    DATA_RAW / "Desi_talep.xlsx",
    "koordinatlar":  DATA_RAW / "Koordinatlar_v2.xlsx",
    "arac_maliyet":  DATA_RAW / "Araç_Kapasite_Maliyet.xlsx",
    "kiralik_arac":  DATA_RAW / "Kiralık_Araçlar.xlsx",
}

# ---------------------------------------------------------------------------
# Araç tipleri  (Araç_Kapasite_Maliyet.xlsx'ten türetilmiş)
# ---------------------------------------------------------------------------
VEHICLE_TYPES = {
    "Tır": {
        "kapasite":      22_400,   # desi
        "kira_gun":       7_000,   # TL/gün
        "kira_km":           13,   # TL/km
        "spot_sabit":    11_700,   # TL/gün (sabit maliyet)
        "spot_km":           25,   # TL/km
    },
    "Kamyon": {
        "kapasite":      12_000,
        "kira_gun":       5_000,
        "kira_km":           10,
        "spot_sabit":     7_638,
        "spot_km":           21,
    },
    "Hafif Kamyon": {
        "kapasite":       7_200,
        "kira_gun":       5_000,
        "kira_km":           10,
        "spot_sabit":     8_750,
        "spot_km":           20,
    },
    "Kamyonet": {
        "kapasite":       5_600,
        "kira_gun":       3_750,
        "kira_km":            6,
        "spot_sabit":     4_750,
        "spot_km":           18,
    },
}

# Büyükten küçüğe sıralı araç listesi (spot atamasında kullanılır)
VEHICLE_ORDER = ["Tır", "Kamyon", "Hafif Kamyon", "Kamyonet"]

# ---------------------------------------------------------------------------
# Kısıt parametreleri
# ---------------------------------------------------------------------------
MIN_DOLULUK_ORAN    = 0.10   # %10 spot doluluk kuralı
UGRAMA_SAPMA_ESIGI  = 1.15   # uğrama: maksimum mesafe sapması (%15)
MAX_ARA_DURAK       = 2      # uğramada maksimum ara durak sayısı

# ---------------------------------------------------------------------------
# Tahmin parametreleri
# ---------------------------------------------------------------------------
TAHMIN_BASLANGIC = "2026-05-11"
TAHMIN_BITIS     = "2026-05-17"
EGITIM_BITIS     = "2026-05-10"   # eğitim verisinin son günü

LGBM_PARAMS = {
    "objective":    "regression_l1",   # MAE — yansız tahmin için
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves":    63,
    "min_child_samples": 20,
    "subsample":    0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs":       -1,
    "verbose":      -1,
}

# CV: time-series walk-forward katlamaları
CV_N_SPLITS = 5
CV_GAP_DAYS = 7    # eğitim→test arası boşluk (sızıntı önlemi)

# ---------------------------------------------------------------------------
# Çıktı dosyaları
# ---------------------------------------------------------------------------
OUTPUT_FILES = {
    "tahmin":        OUTPUT_DIR / "Tahminlenen_Talep.xlsx",
    "plan":          OUTPUT_DIR / "Araç_Planlama.xlsx",
    "karsılanamaz":  OUTPUT_DIR / "Karşılanamayan_Yük.xlsx",
}

# Ondalık basamak — kanonik yuvarlama
DECIMAL_PLACES = 2
