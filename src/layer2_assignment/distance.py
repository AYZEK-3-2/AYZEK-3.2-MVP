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


def validate_distance_matrix(dist_df: pd.DataFrame) -> None:
    """
    Üretilen mesafe matrisinin beklenen yapıya uyduğunu doğrular.
    Herhangi bir ihlalde ValueError fırlatır.

    Kontroller:
      - Boyut 18×18
      - Köşegen (aynı TM→TM) sıfır
      - Simetrik (A→B == B→A)
      - Tüm değerler ≥ 0
    """
    hatalar = []

    if dist_df.shape != (18, 18):
        hatalar.append(f"Boyut beklenen (18,18), bulunan {dist_df.shape}")

    diagonal = np.diag(dist_df.values)
    if not np.allclose(diagonal, 0):
        hatalar.append("Köşegen sıfır değil — aynı TM'ye mesafe 0 olmalı")

    if not np.allclose(dist_df.values, dist_df.values.T, atol=1e-6):
        hatalar.append("Matris simetrik değil — A→B mesafesi B→A'dan farklı")

    if (dist_df.values < 0).any():
        hatalar.append("Negatif mesafe değeri var")

    if hatalar:
        raise ValueError("Mesafe matrisi doğrulama başarısız:\n" + "\n".join(f"  • {h}" for h in hatalar))

    print(f"[validate_distance_matrix] OK — {dist_df.shape[0]}×{dist_df.shape[1]} | simetrik | köşegen=0 | tüm değerler ≥ 0")


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
