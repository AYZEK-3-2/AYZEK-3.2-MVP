# -*- coding: utf-8 -*-
"""
Maliyet fonksiyonu — izole modül.

Tüm araç maliyeti hesapları bu tek fonksiyondan geçer.
Girdi: araç tipi, mesafe (km), kiralık mı spot mu, araç sayısı
Çıktı: toplam TL

Maliyet formülleri (Araç_Kapasite_Maliyet + Soru-Cevap'tan kesinleşen):
  Kiralık : günlük kira  + kiralık_km  × mesafe
  Spot    : spot_sabit   + spot_km     × mesafe   (sabit günlük maliyet dahil!)

Uyarı: spot için sadece km hesabı YAPMA — spot_sabit yoksa tüm maliyetler hatalı çıkar.
"""
from src.config import VEHICLE_TYPES


def vehicle_cost(
    tip: str,
    mesafe_km: float,
    kiralik: bool,
    n_arac: int = 1,
) -> float:
    """
    Parametre
    ---------
    tip       : "Tır" | "Kamyon" | "Hafif Kamyon" | "Kamyonet"
    mesafe_km : rota mesafesi km — uğramalı rotada kümülatif bacak toplamı kullanılmalı
    kiralik   : True → kiralık formülü; False → spot formülü
    n_arac    : araç adedi (maliyeti bu faktörle çarpar)

    Döndürür
    --------
    float : toplam TL (n_arac × birim maliyet)

    Elle doğrulama örnekleri (400 km):
      Spot Tır    : 11.700 + 25 × 400 = 21.700 TL
      Spot Kamyon : 7.638  + 21 × 400 = 16.038 TL
      Kiralık Tır : 7.000  + 13 × 400 = 12.200 TL
    """
    v = VEHICLE_TYPES[tip]
    if kiralik:
        birim = v["kira_gun"] + v["kira_km"] * mesafe_km
    else:
        birim = v["spot_sabit"] + v["spot_km"] * mesafe_km
    return round(birim * n_arac, 2)


def cost_per_desi(tip: str, mesafe_km: float, kiralik: bool, loaded_desi: float) -> float:
    """Birim desi başına etkin maliyet — araç seçim karşılaştırmasında kullanılır."""
    if loaded_desi <= 0:
        return float("inf")
    return vehicle_cost(tip, mesafe_km, kiralik) / loaded_desi
