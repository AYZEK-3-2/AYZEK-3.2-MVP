# -*- coding: utf-8 -*-
"""
Atama motoru birim testleri.

Çalıştırma: python -m pytest tests/test_assignment.py -v
ya da:      python tests/test_assignment.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.config import VEHICLE_TYPES
from src.layer2_assignment.cost_function import vehicle_cost, cost_per_desi
from src.layer2_assignment.spot_assigner import assign_spot, _select_vehicle


# ---------------------------------------------------------------------------
# Yardımcı: küçük mesafe matrisi
# ---------------------------------------------------------------------------
def dummy_dist(dist_km: float = 400.0, tms=("A", "B", "C")) -> pd.DataFrame:
    n = len(tms)
    mat = {t: {t2: (0.0 if t == t2 else dist_km) for t2 in tms} for t in tms}
    return pd.DataFrame(mat).T


# ---------------------------------------------------------------------------
# Maliyet fonksiyonu testleri
# ---------------------------------------------------------------------------
def test_spot_cost_tir_400km():
    """Spot Tır 400 km: 11.700 + 25×400 = 21.700 TL"""
    result = vehicle_cost("Tır", 400, kiralik=False)
    assert result == 21_700.0, f"Beklenen 21700, gelen {result}"
    print("OK Spot Tır 400 km maliyet doğru")

def test_spot_cost_kamyon_400km():
    """Spot Kamyon 400 km: 7.638 + 21×400 = 16.038 TL"""
    result = vehicle_cost("Kamyon", 400, kiralik=False)
    assert result == 16_038.0, f"Beklenen 16038, gelen {result}"
    print("OK Spot Kamyon 400 km maliyet doğru")

def test_rental_cost_tir_400km():
    """Kiralık Tır 400 km: 7.000 + 13×400 = 12.200 TL"""
    result = vehicle_cost("Tır", 400, kiralik=True)
    assert result == 12_200.0, f"Beklenen 12200, gelen {result}"
    print("OK Kiralık Tır 400 km maliyet doğru")

def test_rental_n_arac():
    """2 kiralık Kamyon 400 km: 2 × (5.000 + 10×400) = 18.000 TL"""
    result = vehicle_cost("Kamyon", 400, kiralik=True, n_arac=2)
    assert result == 18_000.0, f"Beklenen 18000, gelen {result}"
    print("OK 2 kiralık Kamyon 400 km maliyet doğru")


# ---------------------------------------------------------------------------
# Araç seçim testleri
# ---------------------------------------------------------------------------
def test_select_large_demand():
    """25.000 desi için Tır seçilmeli (kısmi doldur), None döndürmemeli."""
    result = _select_vehicle(25_000, 400)
    assert result is not None, "25.000 desi için araç bulunamadı — kritik bug!"
    tip, atanan, _ = result
    assert tip == "Tır", f"Beklenen Tır, gelen {tip}"
    assert atanan == 22_400, f"Beklenen 22400, gelen {atanan}"
    print(f"OK 25.000 desi -> Tır (kısmi: 22.400 atandı)")

def test_select_below_threshold():
    """559 desi -> None (terminal kural)"""
    result = _select_vehicle(559, 400)
    assert result is None, f"559 desi terminal kural tetiklemeli, gelen {result}"
    print("OK 559 desi -> terminal kural (None)")

def test_select_6050_avoids_tiny_remainder():
    """
    6.050 desi -> Kamyonet seçilirse 5.600 dolup 450 artık kalır (< 560).
    Motor bunu önlemeli ve Kamyon seçmeli.
    """
    result = _select_vehicle(6_050, 400)
    assert result is not None
    tip, atanan, _ = result
    leftover = 6_050 - atanan
    assert leftover == 0 or leftover >= 560, (
        f"Motor 560-altı artık bıraktı: {leftover} desi. tip={tip}, atanan={atanan}"
    )
    print(f"OK 6.050 desi -> {tip} ({atanan} atandı, artık={leftover})")

def test_select_exact_560():
    """Tam 560 desi (eşikte) -> Kamyonet seçilmeli"""
    result = _select_vehicle(560, 400)
    assert result is not None, "560 desi araç bulamamalı değil"
    tip, atanan, _ = result
    assert atanan == 560
    print(f"OK 560 desi -> {tip}")


# ---------------------------------------------------------------------------
# assign_spot entegrasyon testi
# ---------------------------------------------------------------------------
def test_assign_spot_large_demand():
    """
    30.000 desi (> Tır kapasitesi) -> 1 Tır (22.400) + 1 Tır (7.600) = 30.000 atanmalı.
    """
    dist_df = dummy_dist(400)
    kalan_df = pd.DataFrame([{
        "tarih":    pd.Timestamp("2026-05-11"),
        "cikis_tm": "A",
        "varis_tm": "B",
        "kalan":    30_000.0,
    }])
    plan, unful = assign_spot(kalan_df, dist_df)
    assert unful.empty, f"Karşılanamayan yük olmamalıydı: {unful}"
    assert abs(plan["Atanan Desi"].sum() - 30_000) < 0.1
    print(f"OK 30.000 desi -> {len(plan)} araç, toplam {plan['Atanan Desi'].sum()} desi")

def test_assign_spot_below_threshold():
    """400 desi (< 560) -> unfulfilled olmalı, plan boş olmalı."""
    dist_df = dummy_dist(400)
    kalan_df = pd.DataFrame([{
        "tarih":    pd.Timestamp("2026-05-11"),
        "cikis_tm": "A",
        "varis_tm": "B",
        "kalan":    400.0,
    }])
    plan, unful = assign_spot(kalan_df, dist_df)
    assert plan.empty, "Plan boş olmalıydı"
    assert len(unful) == 1
    assert unful.iloc[0]["Karşılanamayan Desi"] == 400.0
    print("OK 400 desi -> unfulfilled (terminal kural)")

def test_invariant_large():
    """Büyük talep: atanan + karşılanamayan = kalan (invariant)."""
    dist_df = dummy_dist(300)
    kalan = 15_432.5
    kalan_df = pd.DataFrame([{
        "tarih":    pd.Timestamp("2026-05-12"),
        "cikis_tm": "A",
        "varis_tm": "B",
        "kalan":    kalan,
    }])
    plan, unful = assign_spot(kalan_df, dist_df)
    atanan_toplam = plan["Atanan Desi"].sum() if not plan.empty else 0
    unful_toplam  = unful["Karşılanamayan Desi"].sum() if not unful.empty else 0
    delta = abs(atanan_toplam + unful_toplam - kalan)
    assert delta < 0.1, f"İnvariant ihlali! delta={delta}"
    print(f"OK İnvariant OK: {atanan_toplam} + {unful_toplam} ~= {kalan}")


# ---------------------------------------------------------------------------
# Koşturucu
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_spot_cost_tir_400km,
        test_spot_cost_kamyon_400km,
        test_rental_cost_tir_400km,
        test_rental_n_arac,
        test_select_large_demand,
        test_select_below_threshold,
        test_select_6050_avoids_tiny_remainder,
        test_select_exact_560,
        test_assign_spot_large_demand,
        test_assign_spot_below_threshold,
        test_invariant_large,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Sonuc: {passed} gecti / {failed} basarisiz")
