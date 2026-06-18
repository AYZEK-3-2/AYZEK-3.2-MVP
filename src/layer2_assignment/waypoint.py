# -*- coding: utf-8 -*-
"""
Atama Adım 2: Uğrama (multi-stop) aday üretimi ve karar motoru.

Kurallar (MVP Mimarisi §6.2):
 - Yalnızca SPOT havuzuna uygulanır; kiralık araçlara kesinlikle uygulanmaz.
 - Sapma eşiği: kümülatif rota mesafesi ≤ O→D mesafesi × 1.15
 - Araç pik yüke göre seçilir (bottleneck), son bacağa göre değil.
 - Kapasite kontrolü bacak bazlı (dinamik); statik toplam değil.
 - Maliyet karşılaştırmasında kümülatif bacak mesafesi kullanılır.
 - Ancak ayrı araçlardan ucuzsa uğrama uygulanır.
"""
import itertools
import pandas as pd
import numpy as np
from src.config import VEHICLE_TYPES, VEHICLE_ORDER, UGRAMA_SAPMA_ESIGI, MAX_ARA_DURAK
from src.layer2_assignment.distance import route_distance
from src.layer2_assignment.cost_function import vehicle_cost


def _spot_cost(tip: str, mesafe_km: float) -> float:
    return vehicle_cost(tip, mesafe_km, kiralik=False)


def _min_vehicle_for_load(peak_load: float) -> str | None:
    """Pik yükü taşıyabilecek en küçük (en ucuz) araç tipini döndürür."""
    for tip in VEHICLE_ORDER:
        if VEHICLE_TYPES[tip]["kapasite"] >= peak_load:
            return tip
    return None  # hiçbir araç yetmez


def _leg_loads(loads_per_stop: dict, stops: list[str]) -> list[float]:
    """
    Çok-duraklı rotada her bacaktaki anlık (dinamik) yükü hesaplar.
    loads_per_stop = {tm: (yuklenecek, inecek)} sözlüğü.
    """
    current = 0.0
    leg_loads = []
    for i in range(len(stops) - 1):
        origin = stops[i]
        yuklenecek, inecek = loads_per_stop.get(origin, (0.0, 0.0))
        current += yuklenecek - inecek
        leg_loads.append(current)
    return leg_loads


def find_waypoint_candidates(
    kalan_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    all_tms: list[str],
) -> list[dict]:
    """
    Kalan yük DataFrame'inden uğrama adaylarını üretir.

    Döndürür
    --------
    candidates : her aday için
        {
          "tarih", "stops": [O, ara, ..., D],
          "loads": {tm: (yuklenecek, inecek)},
          "toplam_desi", "tip", "maliyet", "direkt_maliyet"
        }
    """
    candidates = []

    for gun, df_gun in kalan_df[kalan_df["kalan"] > 0].groupby("tarih"):
        routes = list(df_gun.itertuples(index=False))

        for r1, r2 in itertools.combinations(routes, 2):
            # O1→D1 ve O2→D2 hatlarından uğrama fırsatı ara
            for (origin, dest, load_od), (origin2, dest2, load_o2d2) in [
                ((r1.cikis_tm, r1.varis_tm, r1.kalan),
                 (r2.cikis_tm, r2.varis_tm, r2.kalan)),
                ((r2.cikis_tm, r2.varis_tm, r2.kalan),
                 (r1.cikis_tm, r1.varis_tm, r1.kalan)),
            ]:
                # Senaryo: araç O→ara_durak→D gider; ara_durak = origin2
                # Koşul: origin2, O ve D arasında coğrafi olarak yol üstünde
                ara = origin2
                if ara == origin or ara == dest:
                    continue

                stops = [origin, ara, dest]
                kum_mesafe = route_distance(dist_df, stops)
                direkt_mesafe = dist_df.loc[origin, dest]

                if kum_mesafe > direkt_mesafe * UGRAMA_SAPMA_ESIGI:
                    continue  # sapma eşiğini aşıyor

                # Bacak bazlı yük: O'da tüm yük biner, ara'da r2 yükü iner,
                # r2 kendi Origin'indeyse r2 yükü de biner (basit senaryo)
                leg1 = load_od + load_o2d2   # O→ara: ikisi de araçta
                leg2 = load_od               # ara→D: yalnızca O→D yükü

                peak = max(leg1, leg2)
                tip  = _min_vehicle_for_load(peak)
                if tip is None:
                    continue

                ugrama_maliyet = _spot_cost(tip, kum_mesafe)

                # Direkt maliyet: iki ayrı araç
                tip1 = _min_vehicle_for_load(load_od)
                tip2 = _min_vehicle_for_load(load_o2d2)
                if tip1 is None or tip2 is None:
                    continue
                direkt_maliyet = (
                    _spot_cost(tip1, direkt_mesafe)
                    + _spot_cost(tip2, dist_df.loc[origin2, dest2])
                )

                if ugrama_maliyet < direkt_maliyet:
                    candidates.append({
                        "tarih":          gun,
                        "stops":          stops,
                        "loads":          {origin: (load_od + load_o2d2, 0),
                                           ara:    (0, load_o2d2),
                                           dest:   (0, load_od)},
                        "toplam_desi":    load_od + load_o2d2,
                        "tip":            tip,
                        "maliyet":        round(ugrama_maliyet, 2),
                        "direkt_maliyet": round(direkt_maliyet, 2),
                        "tasarruf":       round(direkt_maliyet - ugrama_maliyet, 2),
                        "hat1":           (origin, dest),
                        "hat2":           (origin2, dest2),
                        "load1":          load_od,
                        "load2":          load_o2d2,
                    })

    # Tasarrufa göre sıralı döndür
    candidates.sort(key=lambda x: x["tasarruf"], reverse=True)
    return candidates
