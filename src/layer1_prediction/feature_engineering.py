# -*- coding: utf-8 -*-
"""
Ham desi_talep serisinden LightGBM feature'larını türetir.

Feature grupları:
  - Lag: 1, 7, 14 gün önceki desi (haftalık çevrim + kısa dönem)
  - Hareketli istatistikler: 7 ve 14 günlük rolling mean/std
  - Takvim: haftanın günü, ay, ayın günü, ay başı/sonu
  - Tatil/bayram bayrakları: talep sıçramalarını açıklar
  - Hat kimliği: cikis_tm + varis_tm (kategorik)
"""
import pandas as pd
import numpy as np
from src.config import EGITIM_BITIS


# Türkiye resmi tatil ve dini bayram günleri (2026, ±1 gün tampon dahil değil)
TATIL_GUNLERI = [
    "2026-01-01",  # Yılbaşı
    "2026-04-23",  # Ulusal Egemenlik
    "2026-05-01",  # İşçi Bayramı
    "2026-05-19",  # Atatürk'ü Anma
    # Ramazan Bayramı 2026 (tahmini: 20-22 Mart)
    "2026-03-20", "2026-03-21", "2026-03-22",
    # Kurban Bayramı 2026 (tahmini: 27-30 Mayıs)
    "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parametre
    ---------
    df : load_desi_talep() çıktısı — kolonlar: cikis_tm, varis_tm, tarih, desi

    Döndürür
    --------
    df_feat : her hat-gün için feature + hedef (desi) içeren DataFrame.
              Lag gerektiren ilk satırlar NaN olacağından dropna() uygulanır.
    """
    df = df.copy()
    df = df.sort_values(["cikis_tm", "varis_tm", "tarih"]).reset_index(drop=True)

    tatiller = pd.to_datetime(TATIL_GUNLERI)

    # Her hat grubu içinde lag ve rolling hesapla
    grp = df.groupby(["cikis_tm", "varis_tm"], sort=False)

    df["lag_1"]  = grp["desi"].shift(1)
    df["lag_7"]  = grp["desi"].shift(7)
    df["lag_14"] = grp["desi"].shift(14)

    df["roll7_mean"]  = grp["desi"].transform(lambda x: x.shift(1).rolling(7,  min_periods=4).mean())
    df["roll7_std"]   = grp["desi"].transform(lambda x: x.shift(1).rolling(7,  min_periods=4).std())
    df["roll14_mean"] = grp["desi"].transform(lambda x: x.shift(1).rolling(14, min_periods=7).mean())
    df["roll14_std"]  = grp["desi"].transform(lambda x: x.shift(1).rolling(14, min_periods=7).std())

    # Takvim
    df["gun_haftada"]  = df["tarih"].dt.dayofweek          # 0=Pzt … 6=Paz
    df["ay"]           = df["tarih"].dt.month
    df["ay_gunu"]      = df["tarih"].dt.day
    df["ay_basi"]      = (df["tarih"].dt.day <= 3).astype(int)
    df["ay_sonu"]      = (df["tarih"].dt.day >= 28).astype(int)

    # Tatil bayrağı
    df["tatil"]        = df["tarih"].isin(tatiller).astype(int)

    # Hat kimliği (kategorik olarak tutulur; LightGBM kategorikle çalışır)
    df["cikis_tm"]  = df["cikis_tm"].astype("category")
    df["varis_tm"]  = df["varis_tm"].astype("category")

    df = df.dropna(subset=["lag_1", "lag_7", "lag_14"]).reset_index(drop=True)
    return df


FEATURE_COLS = [
    "cikis_tm", "varis_tm",
    "lag_1", "lag_7", "lag_14",
    "roll7_mean", "roll7_std", "roll14_mean", "roll14_std",
    "gun_haftada", "ay", "ay_gunu", "ay_basi", "ay_sonu",
    "tatil",
]
TARGET_COL = "desi"
