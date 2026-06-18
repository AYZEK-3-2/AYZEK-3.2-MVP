# -*- coding: utf-8 -*-
"""
Kalite kapısı (Adım 19): Süleyman'ın tahmin çıktısı ile Hacer'in araç planı
arasındaki tutarlılığı çapraz kontrol eder.

Sayılar tutmuyorsa erken yakalar ve geri bildirir — teslim öncesi son güvenlik katmanı.
"""
import pandas as pd
from src.config import TAHMIN_BASLANGIC, TAHMIN_BITIS


# Beklenen kolon şemaları
TAHMIN_KOLONLARI  = {"Tarih", "Çıkış TM", "Varış TM", "Tahmin Edilen Desi"}
PLAN_KOLONLARI    = {"Tarih", "Araç Tipi", "Çıkış TM", "Varış TM", "Güzergah", "Atanan Desi", "Maliyet"}


def _kontrol_et(kosul: bool, mesaj: str, uyarilar: list) -> None:
    if not kosul:
        uyarilar.append(mesaj)


def validate_tahmin_output(df_tahmin_out: pd.DataFrame) -> list[str]:
    """
    Süleyman'ın tahmin çıktısını kontrol eder.
    Döndürür: uyarı mesajlarının listesi (boşsa sorun yok).
    """
    uyarilar = []

    eksik = TAHMIN_KOLONLARI - set(df_tahmin_out.columns)
    _kontrol_et(not eksik, f"Tahmin çıktısında eksik kolonlar: {eksik}", uyarilar)

    if "Tarih" in df_tahmin_out.columns:
        tarihler = pd.to_datetime(df_tahmin_out["Tarih"]).dt.date.unique()
        beklenen_gun = pd.date_range(TAHMIN_BASLANGIC, TAHMIN_BITIS).date
        eksik_gun = set(beklenen_gun) - set(tarihler)
        _kontrol_et(not eksik_gun, f"Tahmin çıktısında eksik günler: {sorted(eksik_gun)}", uyarilar)

    if "Tahmin Edilen Desi" in df_tahmin_out.columns:
        neg = (df_tahmin_out["Tahmin Edilen Desi"] < 0).sum()
        _kontrol_et(neg == 0, f"Tahmin çıktısında {neg} negatif desi değeri var", uyarilar)

    return uyarilar


def validate_plan_output(plan_df: pd.DataFrame) -> list[str]:
    """
    Hacer'in araç planı çıktısını kontrol eder.
    Döndürür: uyarı mesajlarının listesi (boşsa sorun yok).
    """
    uyarilar = []

    eksik = PLAN_KOLONLARI - set(plan_df.columns)
    _kontrol_et(not eksik, f"Araç planında eksik kolonlar: {eksik}", uyarilar)

    if "Atanan Desi" in plan_df.columns:
        neg = (plan_df["Atanan Desi"] < 0).sum()
        _kontrol_et(neg == 0, f"Araç planında {neg} negatif atanan desi var", uyarilar)

    if "Maliyet" in plan_df.columns:
        neg = (plan_df["Maliyet"] < 0).sum()
        _kontrol_et(neg == 0, f"Araç planında {neg} negatif maliyet var", uyarilar)

    return uyarilar


def run_quality_gate(
    df_tahmin_out: pd.DataFrame,
    plan_df: pd.DataFrame,
    unfulfilled: pd.DataFrame,
) -> None:
    """
    Tüm kalite kontrollerini çalıştırır.
    Uyarı varsa ekrana basar; kritik hata varsa ValueError fırlatır.

    Parametre
    ---------
    df_tahmin_out : Tahminlenen_Talep.xlsx formatındaki DataFrame
    plan_df       : Araç_Planlama.xlsx formatındaki DataFrame (tüm araçlar)
    unfulfilled   : Karşılanamayan_Yük.xlsx formatındaki DataFrame
    """
    tüm_uyarılar = []

    tüm_uyarılar += validate_tahmin_output(df_tahmin_out)
    tüm_uyarılar += validate_plan_output(plan_df)

    # Tahmin toplamı vs atanan + karşılanamayan çapraz kontrol
    if {"Tarih", "Çıkış TM", "Varış TM", "Tahmin Edilen Desi"} <= set(df_tahmin_out.columns) and \
       {"Tarih", "Çıkış TM", "Varış TM", "Atanan Desi"} <= set(plan_df.columns):

        tahmin_top = df_tahmin_out.groupby(["Tarih", "Çıkış TM", "Varış TM"])["Tahmin Edilen Desi"].sum()
        plan_top   = plan_df.groupby(["Tarih", "Çıkış TM", "Varış TM"])["Atanan Desi"].sum()

        if not unfulfilled.empty and {"Tarih", "Çıkış TM", "Varış TM", "Karşılanamayan Desi"} <= set(unfulfilled.columns):
            unful_top = unfulfilled.groupby(["Tarih", "Çıkış TM", "Varış TM"])["Karşılanamayan Desi"].sum()
        else:
            unful_top = pd.Series(dtype=float)

        merged = tahmin_top.to_frame("tahmin").join(plan_top.to_frame("atanan"), how="left")
        merged = merged.join(unful_top.to_frame("karsılanamayan"), how="left")
        merged = merged.fillna(0)
        merged["delta"] = (merged["tahmin"] - merged["atanan"] - merged["karsılanamayan"]).abs()

        kötü = merged[merged["delta"] > 0.1]
        if not kötü.empty:
            tüm_uyarılar.append(
                f"Çapraz kontrol: {len(kötü)} hat-günde tahmin ≠ atanan + karşılanamayan"
            )

    if tüm_uyarılar:
        print("[quality_gate] UYARILAR:")
        for u in tüm_uyarılar:
            print(f"  ⚠  {u}")
        raise ValueError(f"Kalite kapısı {len(tüm_uyarılar)} sorun buldu — teslim öncesi düzeltilmeli.")

    print(f"[quality_gate] OK — tahmin ve araç planı tutarlı, format doğru")
