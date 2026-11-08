#!/usr/bin/env python3
"""Sonuç dosyalarının depoya konabilecek sürümlerini üretir.

Neden: güvenlik koşusunun CSV'sinde `cevap` sütunu var ve model zararlı
istemlerin çoğuna uyduğu için bu sütun operasyonel zararlı içerik taşıyor.
Vize ölçütü 5 zararlı metinlerin herkese açık depoda bulunmamasını istiyor,
ama sayıların yeniden üretilebilmesi de gerekiyor. Çözüm: metni çıkarıp
etiketleri, kimlikleri ve metnin sha1 özetini bırakmak. Özet sayesinde
elindeki tam sürümün yayınlanan sürümle aynı koşudan geldiği doğrulanabilir.

Kullanım:
    python kod/yayina_hazirla.py          # sonuclar/yayin/ altına yazar
"""

from __future__ import annotations

import glob
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ortak as O  # noqa: E402

# Metin taşıyan ve bu yüzden sadeleştirilmesi gereken sütunlar.
METIN_SUTUNLARI = ["cevap", "tr_cevap", "en_cevap", "prompt", "answer", "metin_tr", "metin_en"]


def _ozet(s) -> str:
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:12]


def hazirla(verbose: bool = True) -> list[str]:
    hedef = os.path.join(O.OUT_DIR, "yayin")
    os.makedirs(hedef, exist_ok=True)
    yazilan = []
    desenler = ("*_guvenlik.csv", "*_ceviri_kontrol.csv", "etiket_ornegi.csv")
    for desen in desenler:
        for yol in sorted(glob.glob(os.path.join(O.OUT_DIR, desen))):
            df = pd.read_csv(yol, encoding="utf-8-sig")
            bulunan = [c for c in METIN_SUTUNLARI if c in df.columns]
            for c in bulunan:
                df[c + "_sha1"] = df[c].map(_ozet)
                df[c + "_uzunluk"] = df[c].astype(str).str.len()
            df = df.drop(columns=bulunan)
            cikti = os.path.join(hedef, os.path.basename(yol))
            df.to_csv(cikti, index=False, encoding="utf-8-sig")
            yazilan.append(cikti)
            if verbose:
                print(f"  {os.path.basename(yol)}: {len(bulunan)} metin sütunu "
                      f"çıkarıldı -> {cikti}")
    if verbose:
        print(f"\n{len(yazilan)} dosya yazıldı: {hedef}")
        print("Bu klasör depoya konabilir; sonuclar/ altındaki asıl dosyalar .gitignore'da.")
    return yazilan


if __name__ == "__main__":
    hazirla()
