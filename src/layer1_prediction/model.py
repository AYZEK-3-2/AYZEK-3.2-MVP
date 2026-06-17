# -*- coding: utf-8 -*-
"""
LightGBM modelini eğitir ve time-series walk-forward CV ile değerlendirir.

Walk-forward CV: her katlama için geçmiş → gelecek sıralaması korunur;
gelecek veri eğitimde görünmez (veri sızıntısı yoktur).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from src.config import LGBM_PARAMS, CV_N_SPLITS, CV_GAP_DAYS, EGITIM_BITIS
from src.layer1_prediction.feature_engineering import FEATURE_COLS, TARGET_COL


def train(df_feat: pd.DataFrame) -> lgb.LGBMRegressor:
    """
    Tüm eğitim verisi üzerinde final modeli eğitir.

    Parametre
    ---------
    df_feat : build_features() çıktısı

    Döndürür
    --------
    model : eğitilmiş LGBMRegressor
    """
    mask_train = df_feat["tarih"] <= pd.Timestamp(EGITIM_BITIS)
    df_train = df_feat[mask_train]

    X = df_train[FEATURE_COLS]
    y = df_train[TARGET_COL]

    cat_cols = ["cikis_tm", "varis_tm"]
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y, categorical_feature=cat_cols)
    return model


def cross_validate(df_feat: pd.DataFrame) -> dict:
    """
    Walk-forward time-series CV ile MAE ve MAPE raporlar.
    CV sırasında recursive multi-step simülasyonu yapılır (lag tuzağı önlemi).

    Döndürür
    --------
    {"mae": float, "mape": float, "fold_scores": list}
    """
    from src.layer1_prediction.predict import recursive_predict

    mask_train = df_feat["tarih"] <= pd.Timestamp(EGITIM_BITIS)
    df_all = df_feat[mask_train].copy()

    unique_dates = np.sort(df_all["tarih"].unique())
    tss = TimeSeriesSplit(n_splits=CV_N_SPLITS, gap=CV_GAP_DAYS)

    fold_mae, fold_mape = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(tss.split(unique_dates)):
        train_dates = unique_dates[train_idx]
        test_dates  = unique_dates[test_idx]

        df_tr = df_all[df_all["tarih"].isin(train_dates)]
        df_te = df_all[df_all["tarih"].isin(test_dates)]

        X_tr = df_tr[FEATURE_COLS]
        y_tr = df_tr[TARGET_COL]

        cat_cols = ["cikis_tm", "varis_tm"]
        fold_model = lgb.LGBMRegressor(**LGBM_PARAMS)
        fold_model.fit(X_tr, y_tr, categorical_feature=cat_cols)

        # Gerçek koşulu simüle etmek için recursive tahmin
        df_raw_fold = df_all[df_all["tarih"].isin(train_dates)]
        preds = recursive_predict(
            model=fold_model,
            df_history=df_raw_fold,
            predict_dates=pd.to_datetime(test_dates),
        )

        y_true = df_te.set_index(["tarih", "cikis_tm", "varis_tm"])[TARGET_COL]
        y_pred = preds.set_index(["tarih", "cikis_tm", "varis_tm"])["tahmin"]
        aligned = y_true.align(y_pred, join="inner")

        mae  = mean_absolute_error(aligned[0], aligned[1])
        mape = (np.abs(aligned[0] - aligned[1]) / (aligned[0].clip(lower=1))).mean()
        fold_mae.append(mae)
        fold_mape.append(mape)

    return {
        "mae":         float(np.mean(fold_mae)),
        "mape":        float(np.mean(fold_mape)),
        "fold_scores": list(zip(fold_mae, fold_mape)),
    }
