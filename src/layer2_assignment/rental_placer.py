# -*- coding: utf-8 -*-
"""
Atama Adım 1: Kiralık araçları plana zorunlu olarak yerleştirir.

Kurallar:
 - Kiralık araçlar her gün, doluluğa bakılmaksızın çıkar (parası ödenmiş).
 - Atanan desi = gerçek yük (kapasite değil) — hayalet yük önlemi.
 - Kapasite aşımı olursa kalan yük spot'a devredilir (atama motoru halleder).
 - Kiralık araçlara uğrama uygulanmaz (kural ihlali önlemi).
"""
import pandas as pd
from src.config import VEHICLE_TYPES
from src.layer2_assignment.cost_function import vehicle_cost


def place_rental_vehicles(
    df_kiralik: pd.DataFrame,
    df_tahmin: pd.DataFrame,
    dist_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parametre
    ---------
    df_kiralik  : load_kiralik_araclar() çıktısı
    df_tahmin   : recursive_predict() çıktısı — kolonlar: tarih, cikis_tm, varis_tm, tahmin
    dist_df     : Haversine mesafe matrisi

    Döndürür
    --------
    plan_rows  : kiralık araç satırları (Araç Planlama formatı)
    df_kalan   : hat-gün bazlı kalan (taşınamayan) desi — spot katmanına devredilir
    """
    plan_rows = []
    kalan_rows = []

    dates = df_tahmin["tarih"].unique()

    for gun in dates:
        df_gun_talep = df_tahmin[df_tahmin["tarih"] == gun].copy()
        talep_map = {
            (r["cikis_tm"], r["varis_tm"]): r["tahmin"]
            for _, r in df_gun_talep.iterrows()
        }
        kalan_map = dict(talep_map)  # başlangıçta tüm talep kalan

        for _, k_row in df_kiralik.iterrows():
            cikis  = k_row["cikis_tm"]
            varis  = k_row["varis_tm"]
            n_arac = k_row["arac_sayisi"]
            tip    = k_row["arac_tipi"]
            kapasite = VEHICLE_TYPES[tip]["kapasite"]

            # kalan_map kullan — aynı hat için birden fazla kiralık satır varsa
            # önceki atamanın düşüldüğü güncel kalanı görmek gerekir.
            mevcut_kalan = kalan_map.get((cikis, varis), 0.0)
            toplam_kap   = kapasite * n_arac
            atanan       = min(mevcut_kalan, toplam_kap)   # gerçek yük

            mesafe = dist_df.loc[cikis, varis] if cikis != varis else 0.0
            maliyet_birim = vehicle_cost(tip, mesafe, kiralik=True, n_arac=1)
            toplam_maliyet = vehicle_cost(tip, mesafe, kiralik=True, n_arac=n_arac)

            for _ in range(n_arac):
                plan_rows.append({
                    "Tarih":      gun,
                    "Araç Tipi":  f"Kiralık {tip}",
                    "Çıkış TM":   cikis,
                    "Varış TM":   varis,
                    "Güzergah":   f"{cikis}→{varis}",
                    "Atanan Desi": round(atanan / n_arac, 2),
                    "Maliyet":    round(maliyet_birim, 2),
                })

            # Kalan yükü güncelle
            kalan_map[(cikis, varis)] = max(0.0, mevcut_kalan - atanan)

        # Kalan yük tablosu
        for (cikis, varis), kalan in kalan_map.items():
            kalan_rows.append({
                "tarih":    gun,
                "cikis_tm": cikis,
                "varis_tm": varis,
                "kalan":    round(kalan, 2),
            })

    plan_df  = pd.DataFrame(plan_rows)
    kalan_df = pd.DataFrame(kalan_rows)
    return plan_df, kalan_df
