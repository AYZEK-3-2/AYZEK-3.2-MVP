# -*- coding: utf-8 -*-
"""Ham veri dosyalarını data/raw/ altına kopyalar."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import shutil
import glob
from pathlib import Path

SRC = Path(r'C:\Users\pcc\Desktop\AYZEK 3.2\Erken teslim')
DST = Path(__file__).parent / 'data' / 'raw'
DST.mkdir(parents=True, exist_ok=True)

# Glob ile gerçek dosya adlarını bul
all_xlsx = list(SRC.glob('*.xlsx'))
print("Bulunan dosyalar:")
for f in all_xlsx:
    print(f"  {f.name!r}")

# Adına göre eşleştir
for src_path in all_xlsx:
    name = src_path.name
    if 'talep' in name.lower() or 'Talep' in name:
        dst = DST / 'Desi_talep.xlsx'
    elif 'oordinat' in name or 'koordinat' in name.lower():
        dst = DST / 'Koordinatlar_v2.xlsx'
    elif 'apasite' in name or 'Kapasite' in name:
        dst = DST / 'Araç_Kapasite_Maliyet.xlsx'
    elif 'iral' in name or 'kiral' in name.lower():
        dst = DST / 'Kiralık_Araçlar.xlsx'
    else:
        print(f"  Eşleşme yok: {name!r}")
        continue
    shutil.copy2(src_path, dst)
    print(f"  Kopyalandı: {name!r} → {dst.name}")

print("\ndata/raw/ içeriği:")
for f in DST.iterdir():
    print(f"  {f.name}")
