# -*- coding: utf-8 -*-
"""
Ham xlsx dosyalarini okur, temizler ve standart DataFrame'lere donusturur.
Kolon isimleri pozisyona gore atanir — xlsx'teki Turkce basliklar fark etmez.
"""
import pandas as pd
from src.config import RAW_FILES, DECIMAL_PLACES


def load_desi_talep() -> pd.DataFrame:
    """
    Desi_talep.xlsx -> temiz DataFrame.
    Kolonlar: cikis_tm, varis_tm, tarih (datetime), desi (float, >=0)

    Xlsx yapisi (pozisyon bazli):
      0: Cikis TM  1: Varis TM  2: Tarih  3: Desi (toplam)
    """
    df = pd.read_excel(RAW_FILES["desi_talep"])
    df.columns = ["cikis_tm", "varis_tm", "tarih", "desi"]
    df["tarih"] = pd.to_datetime(df["tarih"])
    df["desi"]  = pd.to_numeric(df["desi"], errors="coerce").clip(lower=0).round(DECIMAL_PLACES)
    df = df.dropna(subset=["tarih", "desi"])
    return df.sort_values(["tarih", "cikis_tm", "varis_tm"]).reset_index(drop=True)


def load_koordinatlar() -> pd.DataFrame:
    """
    Koordinatlar v2.xlsx -> DataFrame.
    Kolonlar: tm (str), enlem (float), boylam (float)

    Xlsx yapisi: Transfer Merkezi | Enlem | Boylam
    """
    df = pd.read_excel(RAW_FILES["koordinatlar"])
    df.columns = ["tm", "enlem", "boylam"]
    df["tm"]     = df["tm"].astype(str).str.strip()
    df["enlem"]  = pd.to_numeric(df["enlem"], errors="coerce")
    df["boylam"] = pd.to_numeric(df["boylam"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def load_arac_maliyet() -> pd.DataFrame:
    """
    Arac_Kapasite_Maliyet.xlsx -> DataFrame.
    Kolonlar: arac_tipi, kapasite, kira_gun, kira_km, spot_sabit, spot_km

    Xlsx yapisi (6 kolon):
      Arac Adi | Kapasite (desi) | Kiralik/gun | Kiralik/km | Spot sabit/gun | Spot/km
    """
    df = pd.read_excel(RAW_FILES["arac_maliyet"])
    df.columns = ["arac_tipi", "kapasite", "kira_gun", "kira_km", "spot_sabit", "spot_km"]
    for col in ["kapasite", "kira_gun", "kira_km", "spot_sabit", "spot_km"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["arac_tipi"] = df["arac_tipi"].astype(str).str.strip()
    return df.dropna().reset_index(drop=True)


def load_kiralik_araclar() -> pd.DataFrame:
    """
    Kiralik_Araclar.xlsx -> DataFrame.
    Kolonlar: cikis_tm, varis_tm, arac_sayisi, arac_tipi

    Xlsx yapisi:
      Cikis Transfer Merkezi | Varis Transfer Merkezi | Arac sayisi | Arac Turu
    """
    df = pd.read_excel(RAW_FILES["kiralik_arac"])
    df.columns = ["cikis_tm", "varis_tm", "arac_sayisi", "arac_tipi"]
    df["cikis_tm"]   = df["cikis_tm"].astype(str).str.strip()
    df["varis_tm"]   = df["varis_tm"].astype(str).str.strip()
    df["arac_sayisi"] = pd.to_numeric(df["arac_sayisi"], errors="coerce").fillna(1).astype(int)
    df["arac_tipi"]  = df["arac_tipi"].astype(str).str.strip()
    return df.dropna(subset=["cikis_tm", "varis_tm"]).reset_index(drop=True)


def load_all() -> dict:
    """Tum ham veri setlerini bir dict olarak dondurur."""
    return {
        "desi_talep":      load_desi_talep(),
        "koordinatlar":    load_koordinatlar(),
        "arac_maliyet":    load_arac_maliyet(),
        "kiralik_araclar": load_kiralik_araclar(),
    }
