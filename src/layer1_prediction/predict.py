# -*- coding: utf-8 -*-
"""
Tahmin üretimi — iki strateji:

  - direct_predict   : BİRİNCİL yöntem. Sabit origin'den her ufka doğrudan tahmin;
                       hata birikmez (CV WAPE ~%21.1).
  - recursive_predict: BASELINE/alternatif. Her günün tahmini bir sonraki günün
                       lag_1'i olarak beslenir (lag tuzağı çözümü; CV WAPE ~%24.1).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import DECIMAL_PLACES, TAHMIN_BASLANGIC, TAHMIN_BITIS
from src.layer1_prediction.feature_engineering import (
    FEATURE_COLS, TARGET_COL, TATIL_GUNLERI, build_features,
    build_direct_features, DIRECT_FEATURE_COLS,
)


def direct_predict(
    model,
    df_history: pd.DataFrame,
    predict_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Direct multi-step tahmin (birincil yöntem).

    Sabit origin = (ilk tahmin günü − 1). Her hedef gün, origin'e sabitlenmiş
    feature'larla doğrudan tahmin edilir; üretilen tahmin geri beslenmez, böylece
    recursive'in hata birikimi olmaz.

    Parametre
    ---------
    model         : eğitilmiş LGBMRegressor veya listesi (seed ensemble → ortalama)
    df_history    : geçmiş desi verisi
    predict_dates : tahmin günleri (None → config TAHMIN_BASLANGIC..BITIS)

    Döndürür
    --------
    DataFrame — kolonlar: tarih, cikis_tm, varis_tm, tahmin
    """
    models = list(model) if isinstance(model, (list, tuple)) else [model]

    if predict_dates is None:
        predict_dates = pd.date_range(TAHMIN_BASLANGIC, TAHMIN_BITIS)
    predict_dates = pd.DatetimeIndex(predict_dates)
    origin = predict_dates.min() - pd.Timedelta(days=1)

    hist = df_history[["cikis_tm", "varis_tm", "tarih", "desi"]]
    hist = hist[hist["tarih"] <= origin]
    routes = hist[["cikis_tm", "varis_tm"]].drop_duplicates()

    # Hedef günler için placeholder satırlar (desi NaN). Feature'lar yalnız
    # origin ve öncesine baktığından bu NaN'lar tahmine sızmaz.
    ph = pd.concat([routes.assign(tarih=d, desi=np.nan) for d in predict_dates],
                   ignore_index=True)
    full = pd.concat([hist, ph], ignore_index=True)

    feat = build_direct_features(full)
    feat["orig"] = feat["tarih"] - pd.to_timedelta(feat["h"], unit="D")
    mask = feat["tarih"].isin(predict_dates) & (feat["orig"] == origin)
    today = feat[mask].copy()

    preds = np.mean([m.predict(today[DIRECT_FEATURE_COLS]) for m in models], axis=0)
    preds = np.maximum(preds, 0.0)                    # zero-floor
    preds = np.round(preds, DECIMAL_PLACES)           # kanonik yuvarlama

    out = today[["tarih", "cikis_tm", "varis_tm"]].copy()
    out["tahmin"] = preds
    return out.sort_values(["tarih", "cikis_tm", "varis_tm"]).reset_index(drop=True)


def recursive_predict(
    model,
    df_history: pd.DataFrame,
    predict_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Parametre
    ---------
    model         : eğitilmiş LGBMRegressor veya LGBMRegressor listesi.
                    Liste verilirse her günün tahmini modellerin ortalamasıdır
                    (seed ensemble — bkz. config.SEED_LIST).
    df_history    : geçmiş desi verisi (tarih ≤ son eğitim günü)
    predict_dates : tahmin yapılacak günler (None → config'den alır)

    Döndürür
    --------
    DataFrame — kolonlar: tarih, cikis_tm, varis_tm, tahmin
    """
    # Tek model de liste de kabul edilir; ortalama için listeye normalize et
    models = list(model) if isinstance(model, (list, tuple)) else [model]

    if predict_dates is None:
        predict_dates = pd.date_range(TAHMIN_BASLANGIC, TAHMIN_BITIS)

    tatiller = pd.to_datetime(TATIL_GUNLERI)

    # Geçmiş veriyi çalışma kopyasına al; tahminler buraya eklenir
    df_work = df_history[["cikis_tm", "varis_tm", "tarih", "desi"]].copy()

    results = []

    for pred_date in predict_dates:
        # Tüm hatlar için feature setini oluştur
        df_feat = build_features(df_work)
        df_today = df_feat[df_feat["tarih"] == pred_date].copy()

        if df_today.empty:
            # Geçmişte bu tarihin satırı yoksa; her hat için boş satır ekle
            routes = df_work[["cikis_tm", "varis_tm"]].drop_duplicates()
            empty_rows = routes.copy()
            empty_rows["tarih"] = pred_date
            empty_rows["desi"]  = np.nan
            df_work = pd.concat([df_work, empty_rows], ignore_index=True)
            df_feat = build_features(df_work)
            df_today = df_feat[df_feat["tarih"] == pred_date].copy()

        # Bilinen takvim feature'larını geç (zaten build_features içinde hesaplanır)
        # Sadece lag feature'ları recursive olarak güncellendi; artık tahmin yap
        X_today = df_today[FEATURE_COLS]
        # Seed ensemble: modellerin tahminini ortalama al (tek model ise kendisi)
        preds   = np.mean([m.predict(X_today) for m in models], axis=0)

        # Zero-floor: negatif tahmini sıfıra çek
        preds = np.maximum(preds, 0.0)

        # Kanonik yuvarlama
        preds = np.round(preds, DECIMAL_PLACES)

        day_df = df_today[["tarih", "cikis_tm", "varis_tm"]].copy()
        day_df["tahmin"] = preds
        results.append(day_df)

        # Bu günün tahminini bir sonraki günün lag'i olarak history'ye ekle
        new_rows = day_df.rename(columns={"tahmin": "desi"})
        df_work = pd.concat([df_work, new_rows], ignore_index=True)

    return pd.concat(results, ignore_index=True)
