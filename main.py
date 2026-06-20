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
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Windows konsolu (cp1254) '≥ → ≠ • ⚠' gibi karakterlerde çöker; stdout'u UTF-8'e
# sabitle ki tüm modüllerin print'leri her makinede sorunsuz çalışsın.
for _akis in (sys.stdout, sys.stderr):
    try:
        _akis.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from src.config import DATA_PROC, OUTPUT_FILES, DECIMAL_PLACES
from src.data_loader import load_all, sanity_check
from src.output_writer import write_all
from src.layer1_prediction.model import train_direct, cross_validate
from src.layer1_prediction.predict import direct_predict
from src.layer2_assignment.distance import build_distance_matrix, save_distance_matrix, load_distance_matrix, validate_distance_matrix
from src.layer2_assignment.rental_placer import place_rental_vehicles
from src.layer2_assignment.waypoint import find_waypoint_candidates
from src.layer2_assignment.spot_assigner import assign_spot
from src.layer3_validation.ilp_validator import validate_day
from src.layer3_validation.invariant import check_invariant, validate_unfulfilled_schema
from src.layer3_validation.quality_gate import run_quality_gate


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
    sanity_check(data)

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
    validate_distance_matrix(dist_df)

    # ------------------------------------------------------------------
    # 3. Model eğitimi (direct multi-step seed ensemble)
    # ------------------------------------------------------------------
    if run_cv:
        print("  Cross-validation çalışıyor (bu birkaç dakika sürebilir)...")
        cv_sonuc = cross_validate(df_talep)
        print(f"  CV MAE: {cv_sonuc['mae']:.1f} | WAPE: {cv_sonuc['wape']:.3f}"
              f" (%{100*cv_sonuc['wape']:.1f}) | MAPE(ref): {cv_sonuc['mape']:.2f}")

    print("[3/9] Model eğitiliyor (direct multi-step, 5-seed ensemble)...")
    model = train_direct(df_talep)
    print("  Model eğitimi tamamlandı.")

    # ------------------------------------------------------------------
    # 4. Direct tahmin
    # ------------------------------------------------------------------
    print("[4/9] 7 günlük tahmin üretiliyor (direct)...")
    df_tahmin = direct_predict(model=model, df_history=df_talep)
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

    # Seçilen uğramalı rotalar için kalan_df'i güncelle ve plan satırı üret.
    # Her hat (tarih, cikis, varis) yalnızca BİR ugrama adayında kullanılabilir;
    # en yüksek tasarruflu aday önce geldiği için greedy seçim optimale yakındır.
    ugrama_plan_rows = []
    ugrama_hats = set()   # (tarih, cikis, varis) — bu hat zaten ugrama'da kullanıldı

    for cand in wp_candidates:
        gun  = cand["tarih"]
        hat1 = cand["hat1"]
        hat2 = cand["hat2"]

        key1 = (gun, hat1[0], hat1[1])
        key2 = (gun, hat2[0], hat2[1])

        # Her iki hat da daha önce kullanılmamış olmalı
        if key1 in ugrama_hats or key2 in ugrama_hats:
            continue

        # kalan_df'den güncel yükleri al (önceki adaylar düşmüş olabilir)
        mask1 = ((kalan_df["tarih"] == gun) &
                 (kalan_df["cikis_tm"] == hat1[0]) &
                 (kalan_df["varis_tm"] == hat1[1]))
        mask2 = ((kalan_df["tarih"] == gun) &
                 (kalan_df["cikis_tm"] == hat2[0]) &
                 (kalan_df["varis_tm"] == hat2[1]))

        actual1 = float(kalan_df.loc[mask1, "kalan"].iloc[0]) if mask1.any() else 0.0
        actual2 = float(kalan_df.loc[mask2, "kalan"].iloc[0]) if mask2.any() else 0.0

        if actual1 <= 0 or actual2 <= 0:
            continue

        # kalan_df'yi güncelle
        kalan_df.loc[mask1, "kalan"] = (kalan_df.loc[mask1, "kalan"] - actual1).clip(lower=0)
        kalan_df.loc[mask2, "kalan"] = (kalan_df.loc[mask2, "kalan"] - actual2).clip(lower=0)

        guzergah = "→".join(cand["stops"])
        # Her hat ayrı satıra yazılır — invariant her O→D çiftini ayrı kontrol eder.
        # Maliyet yalnızca hat1 satırında; toplam maliyet değişmez.
        ugrama_plan_rows.append({
            "Tarih":       gun,
            "Araç Tipi":   f"Spot {cand['tip']} (Uğramalı)",
            "Çıkış TM":    hat1[0],
            "Varış TM":    hat1[1],
            "Güzergah":    guzergah,
            "Atanan Desi": round(actual1, 2),
            "Maliyet":     cand["maliyet"],
        })
        ugrama_plan_rows.append({
            "Tarih":       gun,
            "Araç Tipi":   f"Spot {cand['tip']} (Uğramalı)",
            "Çıkış TM":    hat2[0],
            "Varış TM":    hat2[1],
            "Güzergah":    guzergah,
            "Atanan Desi": round(actual2, 2),
            "Maliyet":     0.0,
        })

        ugrama_hats.add(key1)
        ugrama_hats.add(key2)

    ugrama_plan = pd.DataFrame(ugrama_plan_rows)
    print(f"  Seçilen uğramalı rota: {len(ugrama_plan)}")

    # ------------------------------------------------------------------
    # 7. Greedy+ spot atama
    # ------------------------------------------------------------------
    print("[7/9] Spot araçlar atanıyor (Greedy+)...")
    spot_plan, unfulfilled = assign_spot(kalan_df, dist_df)
    print(f"  Spot plan satırları: {len(spot_plan)}")
    print(f"  Karşılanamayan artık yük satırları: {len(unfulfilled)}")
    if not unfulfilled.empty:
        validate_unfulfilled_schema(unfulfilled)

    # ------------------------------------------------------------------
    # 8. İnvariant kontrolü
    # ------------------------------------------------------------------
    print("[8/9] İnvariant kontrolü...")
    tam_plan = pd.concat([kiralik_plan, ugrama_plan, spot_plan], ignore_index=True)
    check_invariant(df_tahmin, tam_plan, unfulfilled)

    # ------------------------------------------------------------------
    # 9. Çıktı üretimi
    # ------------------------------------------------------------------
    print("[9/9] Çıktı dosyaları yazılıyor...")
    df_tahmin_out = df_tahmin.rename(columns={
        "tarih": "Tarih", "cikis_tm": "Çıkış TM",
        "varis_tm": "Varış TM", "tahmin": "Tahmin Edilen Desi"
    })
    toplam_maliyet = write_all(df_tahmin_out, tam_plan, unfulfilled, OUTPUT_FILES)

    # ------------------------------------------------------------------
    # Kalite kapısı — teslim öncesi çapraz kontrol
    # ------------------------------------------------------------------
    print("\n[Kalite Kapısı] Süleyman + Hacer çıktıları kontrol ediliyor...")
    run_quality_gate(df_tahmin_out, tam_plan, unfulfilled)

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
