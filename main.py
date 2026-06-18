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
from src.layer1_prediction.model import train_direct, cross_validate
from src.layer1_prediction.predict import direct_predict
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

    # Seçilen uğramalı rotalar için kalan_df'i güncelle ve plan satırı üret
    ugrama_plan_rows = []
    ugrama_atama_rows = []   # invariant için: her hat-gün ayrı satır
    kullanulan_hatlar = set()  # (tarih, cikis, varis) — bir hat-gün sadece bir kez uğramaya girer

    def get_kalan(tarih, cikis, varis):
        mask = (
            (kalan_df["tarih"] == tarih) &
            (kalan_df["cikis_tm"] == cikis) &
            (kalan_df["varis_tm"] == varis)
        )
        rows = kalan_df.loc[mask, "kalan"]
        return float(rows.iloc[0]) if not rows.empty else 0.0

    def reduce_kalan(tarih, cikis, varis, azalt):
        mask = (
            (kalan_df["tarih"] == tarih) &
            (kalan_df["cikis_tm"] == cikis) &
            (kalan_df["varis_tm"] == varis)
        )
        kalan_df.loc[mask, "kalan"] = (kalan_df.loc[mask, "kalan"] - azalt).clip(lower=0)

    for cand in wp_candidates:
        gun  = cand["tarih"]
        hat1 = cand["hat1"]   # (cikis1, varis1)
        hat2 = cand["hat2"]   # (cikis2, varis2)

        key1 = (gun, hat1[0], hat1[1])
        key2 = (gun, hat2[0], hat2[1])

        # Her hat-gün sadece bir uğramaya katılabilir
        if key1 in kullanulan_hatlar or key2 in kullanulan_hatlar:
            continue

        # Kalan yük hâlâ mevcut mu? (daha önce azaltılmış olabilir)
        kalan1 = get_kalan(gun, hat1[0], hat1[1])
        kalan2 = get_kalan(gun, hat2[0], hat2[1])
        if kalan1 <= 0 or kalan2 <= 0:
            continue

        # Gerçek yükleri güncelle (candidate üretildiğinden bu yana azalmış olabilir)
        load1 = min(cand["load1"], kalan1)
        load2 = min(cand["load2"], kalan2)

        reduce_kalan(gun, hat1[0], hat1[1], load1)
        reduce_kalan(gun, hat2[0], hat2[1], load2)

        kullanulan_hatlar.add(key1)
        kullanulan_hatlar.add(key2)

        guzergah = "->".join(cand["stops"])
        ugrama_plan_rows.append({
            "Tarih":       gun,
            "Araç Tipi":   f"Spot {cand['tip']} (Ugramalı)",
            "Çıkış TM":    cand["stops"][0],
            "Varış TM":    cand["stops"][-1],
            "Güzergah":    guzergah,
            "Atanan Desi": round(load1 + load2, 2),
            "Maliyet":     cand["maliyet"],
        })
        # Invariant için her hat-günün atamasını ayrı kaydet
        for (c, v, ld) in [(hat1[0], hat1[1], load1), (hat2[0], hat2[1], load2)]:
            ugrama_atama_rows.append({
                "Tarih": gun, "cikis_tm": c, "varis_tm": v, "atanan": round(ld, 2)
            })

    ugrama_plan = pd.DataFrame(ugrama_plan_rows)
    print(f"  Secilen ugramalı rota: {len(ugrama_plan)}")

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

    # Kiralık + spot atamaları (kolon bazlı)
    kiralik_spot = pd.concat([kiralik_plan, spot_plan], ignore_index=True)
    atanan_toplam = (
        kiralik_spot.groupby(["Tarih", "Çıkış TM", "Varış TM"])["Atanan Desi"].sum()
        .reset_index()
        .rename(columns={"Çıkış TM": "cikis_tm", "Varış TM": "varis_tm",
                         "Tarih": "tarih", "Atanan Desi": "atanan"})
    )
    # Uğrama atamaları (hat bazlı ayrı kayıttan)
    if ugrama_atama_rows:
        ugr_atama = (
            pd.DataFrame(ugrama_atama_rows)
            .groupby(["Tarih", "cikis_tm", "varis_tm"])["atanan"].sum()
            .reset_index()
        )
        ugr_atama = ugr_atama.rename(columns={"Tarih": "tarih"})
        atanan_toplam = (
            pd.concat([atanan_toplam, ugr_atama], ignore_index=True)
            .groupby(["tarih", "cikis_tm", "varis_tm"])["atanan"].sum()
            .reset_index()
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
