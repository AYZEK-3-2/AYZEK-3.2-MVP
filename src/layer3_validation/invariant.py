# -*- coding: utf-8 -*-
"""
Invariant kontrolü: her hat-gün için
  atanan_desi + karsılanamayan_desi == kanonik_tahmin

Eşitlik bozulursa ValueError fırlatır — pipeline teslim öncesi durur.
"""
import pandas as pd


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
    atanan = (
        tam_plan.groupby(["Tarih", "Çıkış TM", "Varış TM"])["Atanan Desi"]
        .sum()
        .reset_index()
        .rename(columns={"Tarih": "tarih", "Çıkış TM": "cikis_tm",
                         "Varış TM": "varis_tm", "Atanan Desi": "atanan"})
    )

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
