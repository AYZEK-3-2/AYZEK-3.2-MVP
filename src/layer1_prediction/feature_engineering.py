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


# ===========================================================================
# Direct multi-step feature seti (birincil yöntem)
# ===========================================================================
# Recursive yöntem 7 günü zincirleme üretir (her gün bir önceki tahmine dayanır)
# → hata birikir. Direct yöntemde her hedef gün, SABİT bir origin'den (= ilk
# tahmin gününden bir önceki gün) ufuk h kadar ileriye DOĞRUDAN tahmin edilir;
# tahmin geri beslenmez. Tüm lag/rolling origin'e sabitlenir, sadece hedef günün
# takvimi ileri bakar. CV'de recursive %24.1 → direct %21.1 (her fold'da daha iyi).

DIRECT_HORIZONS = range(1, 8)   # 1..7 günlük ufuklar (11–17 Mayıs)


def build_direct_features(df: pd.DataFrame, horizons=DIRECT_HORIZONS) -> pd.DataFrame:
    """
    Her (hat, hedef gün, ufuk h) satırı için origin = hedef − h gününe sabitlenmiş
    lag/rolling özellikleri + hedef günün takvim/tatil özelliklerini üretir.

    Parametre
    ---------
    df       : load_desi_talep() çıktısı (cikis_tm, varis_tm, tarih, desi)
    horizons : tahmin ufukları (varsayılan 1..7)

    Döndürür
    --------
    Genişletilmiş DataFrame. Eğitimde dropna(["v0","vo7"]) uygulanmalı; tahminde
    origin'e göre filtrelenir (bkz. direct_predict).
    """
    df = df.copy().sort_values(["cikis_tm", "varis_tm", "tarih"]).reset_index(drop=True)
    tatiller = pd.to_datetime(TATIL_GUNLERI)

    frames = []
    for h in horizons:
        d = df.copy()
        d["h"] = h
        g = d.groupby(["cikis_tm", "varis_tm"], sort=False)["desi"]
        # Origin-sabitli lag'ler (origin = hedef − h): shift(h) origin'e iner
        d["v0"]   = g.shift(h)         # origin günündeki değer (son bilinen)
        d["vo1"]  = g.shift(h + 1)
        d["vo6"]  = g.shift(h + 6)
        d["vo7"]  = g.shift(h + 7)     # origin'den bir hafta önce
        d["vo13"] = g.shift(h + 13)
        d["rom7"]  = g.transform(lambda x: x.shift(h).rolling(7,  min_periods=4).mean())
        d["ros7"]  = g.transform(lambda x: x.shift(h).rolling(7,  min_periods=4).std())
        d["rom14"] = g.transform(lambda x: x.shift(h).rolling(14, min_periods=7).mean())
        d["ros14"] = g.transform(lambda x: x.shift(h).rolling(14, min_periods=7).std())
        frames.append(d)

    out = pd.concat(frames, ignore_index=True)

    # Hedef günün takvimi (ileri bakar — hedef tarih bilinir, sızıntı değil)
    out["gun_haftada"] = out["tarih"].dt.dayofweek
    out["ay"]          = out["tarih"].dt.month
    out["ay_gunu"]     = out["tarih"].dt.day
    out["ay_basi"]     = (out["tarih"].dt.day <= 3).astype(int)
    out["ay_sonu"]     = (out["tarih"].dt.day >= 28).astype(int)
    out["tatil"]       = out["tarih"].isin(tatiller).astype(int)

    out["cikis_tm"]    = out["cikis_tm"].astype("category")
    out["varis_tm"]    = out["varis_tm"].astype("category")
    return out


DIRECT_FEATURE_COLS = [
    "cikis_tm", "varis_tm", "h",
    "v0", "vo1", "vo6", "vo7", "vo13",
    "rom7", "ros7", "rom14", "ros14",
    "gun_haftada", "ay", "ay_gunu", "ay_basi", "ay_sonu",
    "tatil",
]
