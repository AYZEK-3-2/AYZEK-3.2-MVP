# -*- coding: utf-8 -*-
"""
Atama Adım 3: Greedy+ spot araç ataması.

Kurallar (MVP Mimarisi §6.3 + Ekip Görev Dağılımı §5):
 - %10 doluluk kuralı yalnızca burada, atama anında uygulanır.
 - En küçük spot araç Kamyonet (kap 5.600 desi), %10 eşiği = 560 desi.
 - 560 desi altında kalan artık yük → unfulfilled tablosuna eklenir (terminal kural).
 - Araç seçimi: desi başına etkin maliyeti minimize eden tip seçilir.
 - Büyük talep (> tek araç kapasitesi) için araç tekrar tekrar doldurulur.
 - Parçalamada 560 altı artık bırakılmaz; uygun büyüklükte araç tercih edilir.
"""
import pandas as pd
from src.config import VEHICLE_TYPES, VEHICLE_ORDER, MIN_DOLULUK_ORAN
from src.layer2_assignment.cost_function import vehicle_cost, cost_per_desi

# Sabit eşik: en küçük aracın %10 kapasitesi
_MIN_LOAD = VEHICLE_TYPES["Kamyonet"]["kapasite"] * MIN_DOLULUK_ORAN  # 560 desi


def _select_vehicle(remaining: float, mesafe_km: float):
    """
    `remaining` desi için en ucuz (desi başına maliyet minimumu) spot araç tipini seç.

    Kurallar:
     1. %10 doluluk: aracın gerçek yükü ≥ kapasite × 0.10
     2. Parçalama sonrası artık yük ya 0 ya da ≥ 560 olmalı — aradaki değer bırakma.
     3. Büyük talep (remaining > maks kapasite) → en büyük araçla kısmi doldur.

    Döndürür
    --------
    (tip, atanan_desi, maliyet)  ya da  None (terminal kural devreye girer)
    """
    if remaining < _MIN_LOAD:
        return None  # terminal kural — 560 altı, spot atanmaz

    best = None
    best_cpp = float("inf")   # cost per desi

    for tip in VEHICLE_ORDER:
        kap = VEHICLE_TYPES[tip]["kapasite"]
        min_desi = kap * MIN_DOLULUK_ORAN

        actual_load = min(remaining, kap)

        # %10 doluluk kontrolü
        if actual_load < min_desi:
            continue

        # Parçalama artık kontrolü: bu araçla doldurduktan sonra kalan
        leftover = remaining - actual_load
        if 0 < leftover < _MIN_LOAD:
            # Bu araç 560-altı artık bırakır → atla (§5.3)
            # Ancak hiçbir araç bu sorunsalı çözemiyorsa daha sonra fallback devreye girer
            continue

        cpp = cost_per_desi(tip, mesafe_km, kiralik=False, loaded_desi=actual_load)
        if cpp < best_cpp:
            best_cpp = cpp
            best = (tip, actual_load, vehicle_cost(tip, mesafe_km, kiralik=False))

    if best is not None:
        return best

    # Fallback: 560-altı artık kaçınılamıyorsa (tüm kombinasyonlar artık bırakıyor),
    # en iyi tek araçla doldur; kalan < 560 unfulfilled olarak işaretlenecek.
    for tip in VEHICLE_ORDER:
        kap = VEHICLE_TYPES[tip]["kapasite"]
        min_desi = kap * MIN_DOLULUK_ORAN
        actual_load = min(remaining, kap)
        if actual_load >= min_desi:
            return (tip, actual_load, vehicle_cost(tip, mesafe_km, kiralik=False))

    return None  # hiçbir araç %10 kuralını sağlayamıyor


def assign_spot(
    kalan_df: pd.DataFrame,
    dist_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Kalan yükü Greedy+ yöntemiyle spot araçlara atar.

    Parametre
    ---------
    kalan_df  : rental_placer (ve isteğe bağlı waypoint) sonrası kalan desi
                kolonlar: tarih, cikis_tm, varis_tm, kalan
    dist_df   : Haversine mesafe matrisi (18×18)

    Döndürür
    --------
    plan_df      : spot araç plan satırları — Araç Planlama formatı
    unfulfilled  : karşılanamayan artık yük satırları
    """
    plan_rows  = []
    unful_rows = []

    working = kalan_df[kalan_df["kalan"] > 0].copy()

    for _, row in working.iterrows():
        gun    = row["tarih"]
        cikis  = row["cikis_tm"]
        varis  = row["varis_tm"]
        mesafe = dist_df.loc[cikis, varis] if cikis != varis else 0.0

        remaining = round(float(row["kalan"]), 2)

        while remaining > 0:
            result = _select_vehicle(remaining, mesafe)

            if result is None:
                # Terminal kural: %10 eşiği sağlanamıyor → unfulfilled
                unful_rows.append({
                    "Tarih":               gun,
                    "Çıkış TM":            cikis,
                    "Varış TM":            varis,
                    "Karşılanamayan Desi": round(remaining, 2),
                    "Sebep":               "%10 spot eşiği altında",
                })
                break

            tip, atanan, maliyet = result

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
