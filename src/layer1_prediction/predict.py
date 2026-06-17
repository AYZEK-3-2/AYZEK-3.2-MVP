# -*- coding: utf-8 -*-
"""
Recursive multi-step tahmin.

Lag tuzağı çözümü: tahmin ufkunda her gün üretilince,
o günün tahmini bir sonraki günün lag_1'i olarak beslenir.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import DECIMAL_PLACES, TAHMIN_BASLANGIC, TAHMIN_BITIS
from src.layer1_prediction.feature_engineering import (
    FEATURE_COLS, TARGET_COL, TATIL_GUNLERI, build_features
)


def recursive_predict(
    model: lgb.LGBMRegressor,
    df_history: pd.DataFrame,
    predict_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Parametre
    ---------
    model         : eğitilmiş LGBMRegressor
    df_history    : geçmiş desi verisi (tarih ≤ son eğitim günü)
    predict_dates : tahmin yapılacak günler (None → config'den alır)

    Döndürür
    --------
    DataFrame — kolonlar: tarih, cikis_tm, varis_tm, tahmin
    """
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
        preds   = model.predict(X_today)

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
