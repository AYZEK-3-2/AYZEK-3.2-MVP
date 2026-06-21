# -*- coding: utf-8 -*-
"""
Atama Adım 3: Greedy+ spot araç ataması.

- %10 doluluk kuralı yalnızca burada, atama anında uygulanır.
- En küçük spot araç Kamyonet (kap. 5600), %10 eşiği = 560 desi.
- 560 desi altında kalan artık yük → unfulfilled tablosuna eklenir.
- Araç seçimi: desi başına efektif maliyeti minimize eden tip seçilir.
"""
import pandas as pd
import numpy as np
from src.config import VEHICLE_TYPES, VEHICLE_ORDER, MIN_DOLULUK_ORAN
from src.layer2_assignment.distance import route_distance
from src.layer2_assignment.cost_function import vehicle_cost


def _best_spot_vehicle(desi: float, mesafe_km: float) -> tuple[str, float] | tuple[None, None]:
    """
    %10 doluluk kuralını sağlayan araç tiplerinden desi başına en ucuzunu seç.
    Döndürür (araç_tipi, maliyet) ya da (None, None) eşik sağlanmazsa.
    """
    best_tip, best_cost = None, float("inf")
    for tip in VEHICLE_ORDER:
        kap = VEHICLE_TYPES[tip]["kapasite"]
        min_desi = kap * MIN_DOLULUK_ORAN
        if desi < min_desi:
            continue
        if desi > kap:
            continue
        cost = vehicle_cost(tip, mesafe_km, kiralik=False)
        desi_basi = cost / desi
        if desi_basi < best_cost:
            best_cost = desi_basi
            best_tip  = tip
    return best_tip, best_cost


def assign_spot(
    kalan_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    ugrama_plan: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parametre
    ---------
    kalan_df     : rental_placer çıktısı — kalan desi (uğrama sonrası güncellenmiş olabilir)
    dist_df      : mesafe matrisi
    ugrama_plan  : waypoint.find_waypoint_candidates() sonrası seçilen uğrma rotaları

    Döndürür
    --------
    plan_df      : spot araç plan satırları
    unfulfilled  : karşılanamayan artık yük satırları
    """
    plan_rows   = []
    unful_rows  = []

    # Uğrama planındaki hat-günleri kalan'dan düşülmüş kabul edilir
    # (waypoint seçimi kalan_df'i zaten güncellemeli — akış main.py'da yönetilir)
    working = kalan_df[kalan_df["kalan"] > 0].copy()

    for _, row in working.iterrows():
        gun    = row["tarih"]
        cikis  = row["cikis_tm"]
        varis  = row["varis_tm"]
        kalan  = row["kalan"]
        mesafe = dist_df.loc[cikis, varis] if cikis != varis else 0.0

        # Kalan yükü araçlara doldur (her araç %10 kuralına tabi)
        remaining = kalan
        while remaining > 0:
            tip, _ = _best_spot_vehicle(remaining, mesafe)

            if tip is None:
                # %10 eşiği altında — artık yük
                unful_rows.append({
                    "Tarih":               gun,
                    "Çıkış TM":            cikis,
                    "Varış TM":            varis,
                    "Karşılanamayan Desi": round(remaining, 2),
                    "Sebep":               "%10 spot eşiği altında",
                })
                break

            kap    = VEHICLE_TYPES[tip]["kapasite"]
            atanan = min(remaining, kap)
            maliyet = vehicle_cost(tip, mesafe, kiralik=False)

            plan_rows.append({
                "Tarih":       gun,
                "Araç Tipi":   f"Spot {tip}",
                "Çıkış TM":    cikis,
                "Varış TM":    varis,
                "Güzergah":    f"{cikis}→{varis}",
                "Atanan Desi": round(atanan, 2),
                "Maliyet":     round(maliyet, 2),
            })
            remaining = round(remaining - atanan, 2)

    plan_df  = pd.DataFrame(plan_rows)
    unful_df = pd.DataFrame(unful_rows)
    return plan_df, unful_df
