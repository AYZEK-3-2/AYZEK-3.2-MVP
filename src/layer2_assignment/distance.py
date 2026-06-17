# -*- coding: utf-8 -*-
"""
18 TM koordinatından Haversine (kuş uçuşu) mesafe matrisi üretir.
Soru-Cevap oturumunda kesinleştiği üzere: gerçek karayolu yok, katsayısız.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import DATA_PROC


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """İki koordinat arasındaki Haversine mesafesini km cinsinden döndürür."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_distance_matrix(df_koordinat: pd.DataFrame) -> pd.DataFrame:
    """
    Parametre
    ---------
    df_koordinat : kolonlar → tm, enlem, boylam

    Döndürür
    --------
    DataFrame (18×18) — satır=çıkış TM, sütun=varış TM, değer=km
    """
    tms = df_koordinat["tm"].tolist()
    n   = len(tms)
    mat = np.zeros((n, n))

    for i, row_i in df_koordinat.iterrows():
        for j, row_j in df_koordinat.iterrows():
            mat[i, j] = haversine_km(
                row_i["enlem"], row_i["boylam"],
                row_j["enlem"], row_j["boylam"],
            )

    dist_df = pd.DataFrame(mat, index=tms, columns=tms)
    return dist_df


def save_distance_matrix(dist_df: pd.DataFrame) -> Path:
    out = DATA_PROC / "distance_matrix.csv"
    dist_df.to_csv(out, encoding="utf-8-sig")
    return out


def load_distance_matrix() -> pd.DataFrame:
    path = DATA_PROC / "distance_matrix.csv"
    return pd.read_csv(path, index_col=0, encoding="utf-8-sig")


def route_distance(dist_df: pd.DataFrame, stops: list[str]) -> float:
    """
    Çok-duraklı rotanın kümülatif Haversine mesafesini döndürür.
    stops = [O, ara1, ara2, ..., D]
    Maliyet hesabında DOĞRUdan O→D değil, tüm bacak toplamı kullanılır.
    """
    total = 0.0
    for k in range(len(stops) - 1):
        total += dist_df.loc[stops[k], stops[k + 1]]
    return total
