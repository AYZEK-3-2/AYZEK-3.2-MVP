# -*- coding: utf-8 -*-
"""
LightGBM modelini eğitir ve time-series walk-forward CV ile değerlendirir.

Walk-forward CV: her katlama için geçmiş → gelecek sıralaması korunur;
gelecek veri eğitimde görünmez (veri sızıntısı yoktur).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from src.config import LGBM_PARAMS, CV_N_SPLITS, EGITIM_BITIS, SEED_LIST
from src.layer1_prediction.feature_engineering import (
    FEATURE_COLS, TARGET_COL, build_features,
    build_direct_features, DIRECT_FEATURE_COLS,
)


def _fit_ensemble(X, y, seeds=SEED_LIST) -> list:
    """X, y üzerinde her tohum için bir LGBMRegressor eğitir (seed ensemble)."""
    cat_cols = ["cikis_tm", "varis_tm"]
    models = []
    for seed in seeds:
        m = lgb.LGBMRegressor(**{**LGBM_PARAMS, "random_state": seed})
        m.fit(X, y, categorical_feature=cat_cols)
        models.append(m)
    return models


def train(df_feat: pd.DataFrame, seeds=SEED_LIST) -> list:
    """
    Tüm eğitim verisi üzerinde final seed ensemble'ını eğitir.

    Parametre
    ---------
    df_feat : build_features() çıktısı
    seeds   : ensemble tohumları (config.SEED_LIST)

    Döndürür
    --------
    models : eğitilmiş LGBMRegressor listesi. recursive_predict bu listeyi
             alıp tahminleri ortalar (tek model de geriye-uyumlu kabul edilir).
    """
    mask_train = df_feat["tarih"] <= pd.Timestamp(EGITIM_BITIS)
    df_train = df_feat[mask_train]
    return _fit_ensemble(df_train[FEATURE_COLS], df_train[TARGET_COL], seeds)


def train_direct(df_talep: pd.DataFrame, seeds=SEED_LIST) -> list:
    """
    BİRİNCİL yöntem: direct multi-step seed ensemble'ını eğitir.

    Parametre
    ---------
    df_talep : ham desi verisi (cikis_tm, varis_tm, tarih, desi)
    seeds    : ensemble tohumları

    Döndürür
    --------
    models : LGBMRegressor listesi. direct_predict bu listeyi alıp tahminleri ortalar.
    """
    df = df_talep[["cikis_tm", "varis_tm", "tarih", "desi"]].copy()
    df["tarih"] = pd.to_datetime(df["tarih"])
    df = df[df["tarih"] <= pd.Timestamp(EGITIM_BITIS)]

    feat = build_direct_features(df).dropna(subset=["v0", "vo7"])
    return _fit_ensemble(feat[DIRECT_FEATURE_COLS], feat[TARGET_COL], seeds)


def cross_validate(df_talep: pd.DataFrame, n_folds: int = CV_N_SPLITS,
                   horizon: int = 7, method: str = "direct") -> dict:
    """
    Genişleyen pencere (expanding-window) walk-forward CV.

    Her fold, geçmiş veriyle eğitilen ensemble ile ardışık `horizon` günlük bir
    bloğu üretir — tıpkı gerçek 11–17 Mayıs teslimi gibi. Böylece raporlanan
    metrikler üretim koşulunu birebir yansıtır (tek-adım CV'nin iyimser hatası
    yaşanmaz; bkz. doküman 4.3/2).

    Ham `df_talep` alır ve her foldun geçmişiyle feature'ları sıfırdan kurar —
    tarihçe eksiksizdir (lag/rolling doğru) ve test bloğunun hemen öncesinden
    başlar (gap sorunu yoktur).

    Parametre
    ---------
    df_talep : ham desi verisi (kolonlar: cikis_tm, varis_tm, tarih, desi)
    n_folds  : geriye doğru kaç adet horizon-günlük blok değerlendirilsin
    horizon  : her foldun tahmin ufku (gün) — gerçek görevle aynı: 7
    method   : "direct" (birincil) veya "recursive" (baseline karşılaştırma)

    Döndürür
    --------
    {"mae": float, "wape": float, "mape": float,
     "fold_scores": [(mae, wape, mape), ...]}

    Not: Bu veride bazı hat-günlerin gerçek desi'si çok küçük (5–50). Klasik MAPE
    bu satırlarda patlar (yanıltıcı yüzde). Bu yüzden baş metrik MAE + WAPE'dir
    (WAPE = Σ|hata| / Σgerçek); MAPE referans olarak da raporlanır.
    """
    from src.layer1_prediction.predict import recursive_predict, direct_predict

    df = df_talep[["cikis_tm", "varis_tm", "tarih", "desi"]].copy()
    df["tarih"] = pd.to_datetime(df["tarih"])
    df = df[df["tarih"] <= pd.Timestamp(EGITIM_BITIS)]
    son_gun = df["tarih"].max()

    cat_cols = ["cikis_tm", "varis_tm"]
    fold_mae, fold_wape, fold_mape = [], [], []

    for k in range(n_folds):
        test_bitis     = son_gun - pd.Timedelta(days=k * horizon)
        test_baslangic = test_bitis - pd.Timedelta(days=horizon - 1)
        test_gunleri   = pd.date_range(test_baslangic, test_bitis)

        df_history = df[df["tarih"] < test_baslangic]
        df_test    = df[df["tarih"].isin(test_gunleri)].copy()
        if df_history.empty or df_test.empty:
            continue

        # Foldun geçmişiyle ensemble'ı eğit + bloğu üret (seçilen yönteme göre)
        if method == "direct":
            feat_fold = build_direct_features(df_history).dropna(subset=["v0", "vo7"])
            fold_models = _fit_ensemble(feat_fold[DIRECT_FEATURE_COLS],
                                        feat_fold[TARGET_COL])
            preds = direct_predict(fold_models, df_history, test_gunleri).copy()
        else:
            feat_fold = build_features(df_history)
            fold_models = _fit_ensemble(feat_fold[FEATURE_COLS],
                                        feat_fold[TARGET_COL])
            preds = recursive_predict(fold_models, df_history, test_gunleri).copy()

        # Kategorik/obje dtype uyuşmazlığını önlemek için hat anahtarlarını str'e çevir
        for col in cat_cols:
            df_test[col] = df_test[col].astype(str)
            preds[col]   = preds[col].astype(str)

        y_true = df_test.set_index(["tarih", "cikis_tm", "varis_tm"])["desi"]
        y_pred = preds.set_index(["tarih", "cikis_tm", "varis_tm"])["tahmin"]
        aligned = y_true.align(y_pred, join="inner")
        if len(aligned[0]) == 0:
            continue

        err  = np.abs(aligned[0] - aligned[1])
        mae  = mean_absolute_error(aligned[0], aligned[1])
        wape = err.sum() / aligned[0].sum()                       # robust headline
        mape = (err / aligned[0].clip(lower=1)).mean()            # referans (kararsız)
        fold_mae.append(mae)
        fold_wape.append(wape)
        fold_mape.append(mape)

    return {
        "mae":         float(np.mean(fold_mae))  if fold_mae  else float("nan"),
        "wape":        float(np.mean(fold_wape)) if fold_wape else float("nan"),
        "mape":        float(np.mean(fold_mape)) if fold_mape else float("nan"),
        "fold_scores": [(float(a), float(w), float(p))
                        for a, w, p in zip(fold_mae, fold_wape, fold_mape)],
    }
