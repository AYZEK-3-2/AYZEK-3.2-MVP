# -*- coding: utf-8 -*-
"""
Hacer'in atama motorunu 3 senaryoyla denetle (pytest).

Senaryo 1 — Küçük talep (560 altı)
    Talep %10 eşiğinin altında → unfulfilled tablosuna gitmeli, araç atanmamalı.

Senaryo 2 — Büyük talep
    Tek Tır kapasitesini (22.400) aşmayan yük → spot Tır atanmalı.

Senaryo 3 — Kiralık kapasiteyi aşan talep
    Kiralık Tır (22.400) kapasiteyi doldurur, fazlası spot'a devredilmeli.
"""
import pandas as pd
import pytest

from src.layer2_assignment.rental_placer import place_rental_vehicles
from src.layer2_assignment.spot_assigner import assign_spot
from src.layer3_validation.invariant import check_invariant, validate_unfulfilled_schema

# ---------------------------------------------------------------------------
# Sabitler ve mock yardımcıları
# ---------------------------------------------------------------------------

TARIH = pd.Timestamp("2026-05-11")
TM_A  = "İstanbul"
TM_B  = "Ankara TM"

DIST_DF = pd.DataFrame(
    [[0.0, 350.0], [350.0, 0.0]],
    index=[TM_A, TM_B],
    columns=[TM_A, TM_B],
)


def _tahmin(desi: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "tarih": TARIH, "cikis_tm": TM_A, "varis_tm": TM_B, "tahmin": desi,
    }])


def _bos_kiralik() -> pd.DataFrame:
    return pd.DataFrame(columns=["cikis_tm", "varis_tm", "arac_sayisi", "arac_tipi"])


def _kiralik_tir() -> pd.DataFrame:
    return pd.DataFrame([{
        "cikis_tm": TM_A, "varis_tm": TM_B, "arac_sayisi": 1, "arac_tipi": "Tır",
    }])


# ---------------------------------------------------------------------------
# Senaryo 1: Küçük talep — 560 altı → unfulfilled
# ---------------------------------------------------------------------------

def test_kucuk_talep_unfulfilled_a_gider():
    df_tahmin = _tahmin(400.0)
    kiralik_plan, kalan = place_rental_vehicles(_bos_kiralik(), df_tahmin, DIST_DF)
    _, unful = assign_spot(kalan, DIST_DF)

    assert not unful.empty, "400 desi unfulfilled'a gitmeli"
    assert abs(unful["Karşılanamayan Desi"].sum() - 400.0) < 0.1
    validate_unfulfilled_schema(unful)
    check_invariant(df_tahmin, kiralik_plan, unful)


# ---------------------------------------------------------------------------
# Senaryo 2: Büyük talep — spot Tır atanmalı
# ---------------------------------------------------------------------------

def test_buyuk_talep_spot_arac_atanir():
    df_tahmin = _tahmin(15_000.0)
    kiralik_plan, kalan = place_rental_vehicles(_bos_kiralik(), df_tahmin, DIST_DF)
    spot_plan, unful = assign_spot(kalan, DIST_DF)

    assert not spot_plan.empty, "15.000 desi için araç atanmalı"
    assert spot_plan["Atanan Desi"].sum() > 0
    assert unful.empty or unful["Karşılanamayan Desi"].sum() < 0.1

    tam_plan = pd.concat([kiralik_plan, spot_plan], ignore_index=True)
    check_invariant(df_tahmin, tam_plan, unful)


# ---------------------------------------------------------------------------
# Senaryo 3: Kiralık kapasiteyi aşan talep → fazlası spot'a
# ---------------------------------------------------------------------------

def test_kiralik_kapasiteyi_asan_talep_spota_devredilir():
    df_tahmin = _tahmin(30_000.0)
    kiralik_plan, kalan = place_rental_vehicles(_kiralik_tir(), df_tahmin, DIST_DF)
    spot_plan, unful = assign_spot(kalan, DIST_DF)

    kiralik_atanan = kiralik_plan["Atanan Desi"].sum()
    spot_atanan    = spot_plan["Atanan Desi"].sum() if not spot_plan.empty else 0.0

    assert kiralik_atanan <= 22_400, "Kiralık Tır kapasiteyi aşamaz"
    assert spot_atanan > 0, "Kalan yük spot'a devredilmeli"
    assert abs(kiralik_atanan + spot_atanan - 30_000.0) < 0.1

    tam_plan = pd.concat([kiralik_plan, spot_plan], ignore_index=True)
    check_invariant(df_tahmin, tam_plan, unful)
