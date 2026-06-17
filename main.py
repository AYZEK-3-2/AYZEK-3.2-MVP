# -*- coding: utf-8 -*-
"""
Uçtan uca pipeline çalıştırıcı.

Çalıştırma: python main.py
Adımlar (MVP Mimarisi §4.1):
  1. Ham veri yükleme
  2. Haversine mesafe matrisi
  3. Feature mühendisliği + LightGBM eğitimi
  4. Recursive 7-günlük tahmin (zero-floor + kanonik yuvarlama)
  5. Kiralık araç yerleştirme
  6. Uğrama değerlendirmesi
  7. Greedy+ spot atama
  8. İnvariant kontrolü
  9. Çıktı üretimi (xlsx)
 10. ILP doğrulama (opsiyonel, örnek gün)
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import DATA_PROC, OUTPUT_FILES, DECIMAL_PLACES
from src.data_loader import load_all
from src.layer1_prediction.feature_engineering import build_features
from src.layer1_prediction.model import train, cross_validate
from src.layer1_prediction.predict import recursive_predict
from src.layer2_assignment.distance import build_distance_matrix, save_distance_matrix, load_distance_matrix
from src.layer2_assignment.rental_placer import place_rental_vehicles
from src.layer2_assignment.waypoint import find_waypoint_candidates
from src.layer2_assignment.spot_assigner import assign_spot
from src.layer3_validation.ilp_validator import validate_day


def run_pipeline(run_cv: bool = False, run_ilp: bool = False):
    print("=== AYZEK MVP Pipeline ===\n")

    # ------------------------------------------------------------------
    # 1. Ham veri
    # ------------------------------------------------------------------
    print("[1/9] Veri yükleniyor...")
    data = load_all()
    df_talep   = data["desi_talep"]
    df_koord   = data["koordinatlar"]
    df_maliyet = data["arac_maliyet"]
    df_kiralik = data["kiralik_araclar"]
    print(f"  Desi talep: {df_talep.shape}, Koordinatlar: {df_koord.shape}")

    # ------------------------------------------------------------------
    # 2. Mesafe matrisi
    # ------------------------------------------------------------------
    print("[2/9] Mesafe matrisi hesaplanıyor...")
    dist_path = DATA_PROC / "distance_matrix.csv"
    if dist_path.exists():
        dist_df = load_distance_matrix()
        print("  (önbellekten yüklendi)")
    else:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        dist_df = build_distance_matrix(df_koord)
        save_distance_matrix(dist_df)
    print(f"  Matris boyutu: {dist_df.shape}")

    # ------------------------------------------------------------------
    # 3. Feature mühendisliği + model eğitimi
    # ------------------------------------------------------------------
    print("[3/9] Feature mühendisliği...")
    df_feat = build_features(df_talep)
    print(f"  Feature boyutu: {df_feat.shape}")

    if run_cv:
        print("  Cross-validation çalışıyor (bu birkaç dakika sürebilir)...")
        cv_sonuc = cross_validate(df_feat)
        print(f"  CV MAE: {cv_sonuc['mae']:.1f} | MAPE: {cv_sonuc['mape']:.3f}")

    print("[3/9] Model eğitiliyor...")
    model = train(df_feat)
    print("  Model eğitimi tamamlandı.")

    # ------------------------------------------------------------------
    # 4. Recursive tahmin
    # ------------------------------------------------------------------
    print("[4/9] 7 günlük tahmin üretiliyor (recursive)...")
    df_tahmin = recursive_predict(model=model, df_history=df_talep)
    print(f"  Tahmin satırları: {len(df_tahmin)}")

    # ------------------------------------------------------------------
    # 5. Kiralık araç yerleştirme
    # ------------------------------------------------------------------
    print("[5/9] Kiralık araçlar yerleştiriliyor...")
    kiralik_plan, kalan_df = place_rental_vehicles(df_kiralik, df_tahmin, dist_df)
    print(f"  Kiralık plan satırları: {len(kiralik_plan)}")

    # ------------------------------------------------------------------
    # 6. Uğrama değerlendirmesi
    # ------------------------------------------------------------------
    print("[6/9] Uğrama adayları değerlendiriliyor...")
    all_tms = df_koord["tm"].tolist()
    wp_candidates = find_waypoint_candidates(kalan_df, dist_df, all_tms)
    print(f"  Uğrama adayı sayısı: {len(wp_candidates)}")

    # Seçilen uğramalı rotalar için kalan_df'i güncelle ve plan satırı üret
    ugrama_plan_rows = []
    ugrama_kullanilanlar = set()   # (tarih, hat1, hat2) — çifte atamayı önler

    for cand in wp_candidates:
        gun  = cand["tarih"]
        hat1 = cand["hat1"]
        hat2 = cand["hat2"]
        key  = (gun, hat1, hat2)
        if key in ugrama_kullanilanlar:
            continue

        # kalan_df'de hat1 ve hat2 yüklerini düş
        def reduce_kalan(cikis, varis, azalt):
            mask = (
                (kalan_df["tarih"] == gun) &
                (kalan_df["cikis_tm"] == cikis) &
                (kalan_df["varis_tm"] == varis)
            )
            kalan_df.loc[mask, "kalan"] = (kalan_df.loc[mask, "kalan"] - azalt).clip(lower=0)

        reduce_kalan(hat1[0], hat1[1], cand["load1"])
        reduce_kalan(hat2[0], hat2[1], cand["load2"])

        guzergah = "→".join(cand["stops"])
        ugrama_plan_rows.append({
            "Tarih":       gun,
            "Araç Tipi":   f"Spot {cand['tip']} (Uğramalı)",
            "Çıkış TM":    cand["stops"][0],
            "Varış TM":    cand["stops"][-1],
            "Güzergah":    guzergah,
            "Atanan Desi": round(cand["toplam_desi"], 2),
            "Maliyet":     cand["maliyet"],
        })
        ugrama_kullanilanlar.add(key)

    ugrama_plan = pd.DataFrame(ugrama_plan_rows)
    print(f"  Seçilen uğramalı rota: {len(ugrama_plan)}")

    # ------------------------------------------------------------------
    # 7. Greedy+ spot atama
    # ------------------------------------------------------------------
    print("[7/9] Spot araçlar atanıyor (Greedy+)...")
    spot_plan, unfulfilled = assign_spot(kalan_df, dist_df)
    print(f"  Spot plan satırları: {len(spot_plan)}")
    print(f"  Karşılanamayan artık yük satırları: {len(unfulfilled)}")

    # ------------------------------------------------------------------
    # 8. İnvariant kontrolü
    # ------------------------------------------------------------------
    print("[8/9] İnvariant kontrolü...")
    tam_plan = pd.concat([kiralik_plan, ugrama_plan, spot_plan], ignore_index=True)

    atanan_toplam = (
        tam_plan.groupby(["Tarih", "Çıkış TM", "Varış TM"])["Atanan Desi"].sum()
        .reset_index()
        .rename(columns={"Çıkış TM": "cikis_tm", "Varış TM": "varis_tm", "Tarih": "tarih", "Atanan Desi": "atanan"})
    )

    if not unfulfilled.empty:
        unful_grp = (
            unfulfilled.groupby(["Tarih", "Çıkış TM", "Varış TM"])["Karşılanamayan Desi"].sum()
            .reset_index()
            .rename(columns={"Çıkış TM": "cikis_tm", "Varış TM": "varis_tm", "Tarih": "tarih", "Karşılanamayan Desi": "karsılanamayan"})
        )
    else:
        unful_grp = pd.DataFrame(columns=["tarih", "cikis_tm", "varis_tm", "karsılanamayan"])

    check = df_tahmin.merge(atanan_toplam, on=["tarih", "cikis_tm", "varis_tm"], how="left")
    check = check.merge(unful_grp, on=["tarih", "cikis_tm", "varis_tm"], how="left")
    check["atanan"]       = check["atanan"].fillna(0)
    check["karsılanamayan"] = check["karsılanamayan"].fillna(0)
    check["delta"] = (check["tahmin"] - check["atanan"] - check["karsılanamayan"]).abs()

    violations = check[check["delta"] > 0.1]
    if violations.empty:
        print("  İnvariant: OK — atanan + karşılanamayan = tahmin")
    else:
        print(f"  UYARI: {len(violations)} hat-günde invariant ihlali! İlk 5:")
        print(violations[["tarih", "cikis_tm", "varis_tm", "tahmin", "atanan", "karsılanamayan", "delta"]].head())

    # ------------------------------------------------------------------
    # 9. Çıktı üretimi
    # ------------------------------------------------------------------
    print("[9/9] Çıktı dosyaları yazılıyor...")
    OUTPUT_FILES["tahmin"].parent.mkdir(parents=True, exist_ok=True)

    df_tahmin_out = df_tahmin.rename(columns={
        "tarih": "Tarih", "cikis_tm": "Çıkış TM",
        "varis_tm": "Varış TM", "tahmin": "Tahmin Edilen Desi"
    })
    df_tahmin_out.to_excel(OUTPUT_FILES["tahmin"], index=False)

    tam_plan.to_excel(OUTPUT_FILES["plan"], index=False)

    if not unfulfilled.empty:
        unfulfilled.to_excel(OUTPUT_FILES["karsılanamaz"], index=False)

    toplam_maliyet = tam_plan["Maliyet"].sum()
    print(f"\n  Toplam operasyonel maliyet: {toplam_maliyet:,.0f} TL")
    print(f"  Tahmin dosyası    : {OUTPUT_FILES['tahmin']}")
    print(f"  Araç plan dosyası : {OUTPUT_FILES['plan']}")

    # ------------------------------------------------------------------
    # 10. ILP doğrulama (opsiyonel)
    # ------------------------------------------------------------------
    if run_ilp:
        print("\n[10] ILP doğrulama valfi (ilk gün örnek)...")
        ornek_gun = df_tahmin["tarih"].min()
        gun_greedy_cost = tam_plan[tam_plan["Tarih"] == ornek_gun]["Maliyet"].sum()
        ilp_result = validate_day(ornek_gun, kalan_df, dist_df, gun_greedy_cost)
        print(f"  Gün: {ornek_gun.date()} | Greedy+: {ilp_result.get('greedy_maliyet')} TL"
              f" | ILP: {ilp_result.get('ilp_maliyet')} TL"
              f" | Fark: %{ilp_result.get('fark_pct', 'N/A')}")

    print("\n=== Pipeline tamamlandı ===")
    return tam_plan, df_tahmin_out, unfulfilled


if __name__ == "__main__":
    run_pipeline(run_cv=False, run_ilp=False)
