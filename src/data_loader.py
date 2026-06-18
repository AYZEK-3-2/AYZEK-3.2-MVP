# -*- coding: utf-8 -*-
"""
Ham xlsx dosyalarını okur, temizler ve standart DataFrame'lere dönüştürür.
Tüm downstream modüller bu modülden veri alır.
"""
import pandas as pd
from pathlib import Path
from src.config import RAW_FILES, DECIMAL_PLACES


def load_desi_talep() -> pd.DataFrame:
    """
    Desi_talep.xlsx → temiz DataFrame.
    Kolonlar: cikis_tm, varis_tm, tarih (datetime), desi (float, ≥0)
    """
    df = pd.read_excel(RAW_FILES["desi_talep"])
    df.columns = ["cikis_tm", "varis_tm", "tarih", "desi"]
    df["tarih"] = pd.to_datetime(df["tarih"])
    df["desi"]  = df["desi"].clip(lower=0).round(DECIMAL_PLACES)
    return df.sort_values(["tarih", "cikis_tm", "varis_tm"]).reset_index(drop=True)


def load_koordinatlar() -> pd.DataFrame:
    """
    Koordinatlar_v2.xlsx → DataFrame.
    Kolonlar: tm (str), enlem (float), boylam (float)
    """
    df = pd.read_excel(RAW_FILES["koordinatlar"])
    df.columns = ["tm", "enlem", "boylam"]
    return df


def load_arac_maliyet() -> pd.DataFrame:
    """
    Araç_Kapasite_Maliyet.xlsx → DataFrame.
    Kolonlar: arac_tipi, kapasite, kira_gun, kira_km, spot_sabit, spot_km
    """
    df = pd.read_excel(RAW_FILES["arac_maliyet"])
    df.columns = ["arac_tipi", "kapasite", "kira_gun", "kira_km", "spot_sabit", "spot_km"]
    return df


def load_kiralik_araclar() -> pd.DataFrame:
    """
    Kiralık_Araçlar.xlsx → DataFrame.
    Kolonlar: cikis_tm, varis_tm, arac_sayisi, arac_tipi
    """
    df = pd.read_excel(RAW_FILES["kiralik_arac"])
    df.columns = ["cikis_tm", "varis_tm", "arac_sayisi", "arac_tipi"]
    return df


def load_all() -> dict:
    """Tüm ham veri setlerini bir dict olarak döndürür."""
    return {
        "desi_talep":     load_desi_talep(),
        "koordinatlar":   load_koordinatlar(),
        "arac_maliyet":   load_arac_maliyet(),
        "kiralik_araclar": load_kiralik_araclar(),
    }


def sanity_check(data: dict) -> None:
    """
    Yüklenen ham verinin beklenen yapıya uyduğunu doğrular.
    Herhangi bir ihlalde ValueError fırlatır — pipeline başlamadan hata yakalanır.

    Beklentiler (Çarşamba arayüz toplantısında sabitlendi):
      - 89 benzersiz (cikis_tm, varis_tm) hattı
      - 130 benzersiz tarih
      - 18 TM koordinatı
      - Desi sütununda negatif değer yok
      - Toplam 10.770 satır (bazı hat-gün kombinasyonları boş, tüm 89×130 dolu değil)
    """
    df   = data["desi_talep"]
    koord = data["koordinatlar"]

    hatlar  = df[["cikis_tm", "varis_tm"]].drop_duplicates()
    gunler  = df["tarih"].nunique()
    n_hat   = len(hatlar)
    n_tm    = len(koord)
    n_satirlar = len(df)

    hatalar = []

    if n_hat != 89:
        hatalar.append(f"Hat sayısı beklenen 89, bulunan {n_hat}")

    if gunler != 130:
        hatalar.append(f"Gün sayısı beklenen 130, bulunan {gunler}")

    if n_tm != 18:
        hatalar.append(f"TM sayısı beklenen 18, bulunan {n_tm}")

    neg = (df["desi"] < 0).sum()
    if neg > 0:
        hatalar.append(f"Negatif desi değeri var: {neg} satır")

    BEKLENEN_SATIRLAR = 10_770
    if n_satirlar != BEKLENEN_SATIRLAR:
        hatalar.append(
            f"Toplam satır beklenen {BEKLENEN_SATIRLAR}, bulunan {n_satirlar}"
        )

    if hatalar:
        raise ValueError("Sanity check başarısız:\n" + "\n".join(f"  • {h}" for h in hatalar))

    print(
        f"[sanity_check] OK — {n_hat} hat | {gunler} gün | {n_tm} TM | "
        f"{n_satirlar} satır | negatif desi yok"
    )
