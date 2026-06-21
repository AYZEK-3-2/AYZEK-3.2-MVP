# -*- coding: utf-8 -*-
"""
Rota-tabanlı ILP doğrulama valfi (§6.4).

Formülasyon: set-partitioning — önce geçerli aday rotaları (doğrudan O→D +
uğramalı) üretir; ILP her talebi tam bir kez karşılayan en ucuz kombinasyonu
seçer. Greedy+ ile fark raporlanır.

Bağımlılık: PuLP (pip install pulp)
Yalnızca örnek günlerde çalıştırılır — tüm 7 gün için değil.
"""
import pandas as pd
from itertools import combinations

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


def validate_day(
    gun: pd.Timestamp,
    kalan_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    greedy_cost: float,
) -> dict:
    """
    Tek bir gün için rota-tabanlı ILP çözer ve Greedy+ maliyetiyle kıyaslar.

    Döndürür
    --------
    {
      "gun": ...,
      "ilp_maliyet": float,
      "greedy_maliyet": float,
      "fark_pct": float,
      "durum": "optimal" | "infeasible" | "pulp_eksik"
    }
    """
    if not PULP_AVAILABLE:
        return {"gun": gun, "durum": "pulp_eksik", "greedy_maliyet": greedy_cost}

    df_gun = kalan_df[(kalan_df["tarih"] == gun) & (kalan_df["kalan"] > 0)].copy()
    if df_gun.empty:
        return {"gun": gun, "ilp_maliyet": 0, "greedy_maliyet": greedy_cost, "fark_pct": 0}

    from src.config import VEHICLE_TYPES, VEHICLE_ORDER, MIN_DOLULUK_ORAN, UGRAMA_SAPMA_ESIGI
    from src.layer2_assignment.distance import route_distance
    from src.layer2_assignment.waypoint import find_waypoint_candidates
    from src.layer2_assignment.cost_function import vehicle_cost
    from src.layer2_assignment.spot_assigner import _best_spot_vehicle, _best_full_vehicle, _MAX_KAP

    def _direct_decompose(desi, km):
        """
        Tek bir O→D talebini araçlara böler — greedy spot_assigner ile BİREBİR aynı
        kural: kapasiteyi aşan kısım tam araçlarla bölünür, son parça %10 eşiğine
        tabidir. Döndürür: (toplam_maliyet, karsilanan_desi).
        """
        total, served, rem = 0.0, 0.0, desi
        while rem > 0:
            if rem > _MAX_KAP:
                tip = _best_full_vehicle(km)
                atanan = VEHICLE_TYPES[tip]["kapasite"]
            else:
                tip, _ = _best_spot_vehicle(rem, km)
                if tip is None:       # %10 eşiği altı artık yük → karşılanamaz
                    break
                atanan = min(rem, VEHICLE_TYPES[tip]["kapasite"])
            total += vehicle_cost(tip, km, kiralik=False)
            served += atanan
            rem = round(rem - atanan, 2)
        return total, served

    # Aday rota üretimi
    demands = {
        (r["cikis_tm"], r["varis_tm"]): r["kalan"]
        for _, r in df_gun.iterrows()
    }
    routes = []  # her rota: {"stops", "covered_demands", "tip", "cost"}

    # Doğrudan O→D rotaları — her talep için tam-kapsama (çok araçlı) tek seçenek.
    # Karşılanan desi 0 ise (tümü %10 altı) rota üretilmez; o talep zaten karşılanamaz.
    for (cikis, varis), desi in demands.items():
        km = dist_df.loc[cikis, varis]
        cost, served = _direct_decompose(desi, km)
        if served > 0:
            routes.append({
                "stops":            [cikis, varis],
                "covered_demands":  {(cikis, varis): desi},
                "tip":              "decompose",
                "cost":             cost,
            })

    # Uğramalı rotalar (waypoint adaylarından)
    wp_candidates = find_waypoint_candidates(df_gun, dist_df, list(dist_df.index))
    for cand in wp_candidates:
        routes.append({
            "stops":           cand["stops"],
            "covered_demands": {cand["hat1"]: cand["load1"], cand["hat2"]: cand["load2"]},
            "tip":             cand["tip"],
            "cost":            cand["maliyet"],
        })

    if not routes:
        return {"gun": gun, "durum": "infeasible", "greedy_maliyet": greedy_cost}

    # ILP
    prob = pulp.LpProblem(f"route_ilp_{gun.date()}", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(routes))]

    # Amaç: toplam maliyet minimize
    prob += pulp.lpSum(x[i] * routes[i]["cost"] for i in range(len(routes)))

    # Kısıt: her talep tam bir kez karşılanmalı
    for demand_key in demands:
        covering = [i for i, r in enumerate(routes) if demand_key in r["covered_demands"]]
        if covering:
            prob += pulp.lpSum(x[i] for i in covering) >= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return {"gun": gun, "durum": "infeasible", "greedy_maliyet": greedy_cost}

    ilp_cost = pulp.value(prob.objective)
    fark_pct  = (greedy_cost - ilp_cost) / (ilp_cost + 1e-9) * 100

    return {
        "gun":             gun,
        "ilp_maliyet":     round(ilp_cost, 2),
        "greedy_maliyet":  round(greedy_cost, 2),
        "fark_pct":        round(fark_pct, 2),
        "durum":           "optimal",
    }
