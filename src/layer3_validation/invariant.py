# -*- coding: utf-8 -*-
"""
Invariant kontrolü: her hat-gün için
  atanan_desi + karsılanamayan_desi == kanonik_tahmin

Eşitlik bozulursa ValueError fırlatır — pipeline teslim öncesi durur.
"""
import pandas as pd

# Adım 17: Karşılanamayanlar tablosunun zorunlu kolon yapısı.
# Bu sözlük pipeline genelinde tek kaynak; hem Hacer'in üretimi hem
# Emrullah'ın doğrulaması aynı şemayı kullanır.
UNFULFILLED_SCHEMA = {
    "Tarih":               "datetime64[ns]",
    "Çıkış TM":            "object",
    "Varış TM":            "object",
    "Karşılanamayan Desi": "float64",
    "Sebep":               "object",
}


def validate_unfulfilled_schema(unfulfilled: pd.DataFrame) -> None:
    """
    Karşılanamayan tablo yapısını doğrular.
    Eksik kolon veya dtype uyumsuzluğu varsa ValueError fırlatır.
    """
    eksik = [k for k in UNFULFILLED_SCHEMA if k not in unfulfilled.columns]
    if eksik:
        raise ValueError(
            f"Karşılanamayan tablosunda eksik kolonlar: {eksik}\n"
            f"Beklenen: {list(UNFULFILLED_SCHEMA.keys())}"
        )

    tip_hatalar = []
    for kolon, beklenen in UNFULFILLED_SCHEMA.items():
        gercek = unfulfilled[kolon].dtype
        if beklenen == "datetime64[ns]" and not pd.api.types.is_datetime64_any_dtype(unfulfilled[kolon]):
            tip_hatalar.append(f"  • {kolon}: beklenen datetime, bulunan {gercek}")
        elif beklenen == "float64" and not pd.api.types.is_numeric_dtype(unfulfilled[kolon]):
            tip_hatalar.append(f"  • {kolon}: beklenen float64, bulunan {gercek}")

    if tip_hatalar:
        raise ValueError("Karşılanamayan tablosunda tip uyumsuzluğu:\n" + "\n".join(tip_hatalar))

    print(f"[validate_unfulfilled_schema] OK — {len(unfulfilled)} satır, şema ve tipler doğru")


def check_invariant(
    df_tahmin: pd.DataFrame,
    tam_plan: pd.DataFrame,
    unfulfilled: pd.DataFrame,
    tolerans: float = 0.1,
) -> None:
    """
    Parametre
    ---------
    df_tahmin    : tahmin çıktısı — kolonlar: tarih, cikis_tm, varis_tm, tahmin
    tam_plan     : kiralik + ugrama + spot plan satırları — kolon: Atanan Desi
    unfulfilled  : karşılanamayan yük — kolon: Karşılanamayan Desi
    tolerans     : kayan nokta farkı için kabul eşiği (varsayılan 0.1 desi)

    Hata
    ----
    ValueError — ihlal varsa ilk 5 satırı göstererek fırlatır.
    """
    if not tam_plan.empty and "Tarih" in tam_plan.columns:
        atanan = (
            tam_plan.groupby(["Tarih", "Çıkış TM", "Varış TM"])["Atanan Desi"]
            .sum()
            .reset_index()
            .rename(columns={"Tarih": "tarih", "Çıkış TM": "cikis_tm",
                             "Varış TM": "varis_tm", "Atanan Desi": "atanan"})
        )
    else:
        atanan = pd.DataFrame(columns=["tarih", "cikis_tm", "varis_tm", "atanan"])

    if not unfulfilled.empty:
        unful = (
            unfulfilled.groupby(["Tarih", "Çıkış TM", "Varış TM"])["Karşılanamayan Desi"]
            .sum()
            .reset_index()
            .rename(columns={"Tarih": "tarih", "Çıkış TM": "cikis_tm",
                             "Varış TM": "varis_tm", "Karşılanamayan Desi": "karsılanamayan"})
        )
    else:
        unful = pd.DataFrame(columns=["tarih", "cikis_tm", "varis_tm", "karsılanamayan"])

    check = df_tahmin.merge(atanan, on=["tarih", "cikis_tm", "varis_tm"], how="left")
    check = check.merge(unful,   on=["tarih", "cikis_tm", "varis_tm"], how="left")
    check["atanan"]         = check["atanan"].fillna(0)
    check["karsılanamayan"] = check["karsılanamayan"].fillna(0)
    check["delta"] = (check["tahmin"] - check["atanan"] - check["karsılanamayan"]).abs()

    ihlaller = check[check["delta"] > tolerans]

    if not ihlaller.empty:
        raise ValueError(
            f"İnvariant ihlali: {len(ihlaller)} hat-günde "
            f"atanan + karşılanamayan ≠ tahmin\n"
            + ihlaller[["tarih", "cikis_tm", "varis_tm",
                         "tahmin", "atanan", "karsılanamayan", "delta"]]
              .head()
              .to_string(index=False)
        )

    print(f"[check_invariant] OK — {len(check)} hat-gün kontrolü geçti")
