#!/usr/bin/env python3
"""Ortam ön kontrolü: uzun bir koşuya başlamadan önce neyin eksik olduğunu söyler.

Colab ve Kaggle oturumları sıfırdan başladığı için dosya ve paket eksikleri
koşunun ortasında ortaya çıkıyor. Bu betik saniyeler içinde hepsini kontrol eder.

Kullanım:
    python kod/kontrol.py
"""

from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ortak  # noqa: E402

GEREKLI_VERI = [
    ("yerellestirme_tablosu.csv", "veri_hatti.py (güvenlik seti yerelleştirme)"),
    ("tr_bbq_sablonlar.csv", "tr_bbq_uret.py (önyargı soruları)"),
    ("guvenlik_seti.csv", "Türkiye'ye özgü güvenlik alt kümesi"),
]
URETILEN_VERI = [
    ("tr_bbq_sorular.csv", "tr_bbq_uret.py ile üretilir"),
]
PAKETLER = [
    ("torch", "model yükleme", True),
    ("transformers", "model yükleme", True),
    ("accelerate", "device_map='auto'", True),
    ("bitsandbytes", "INT8 ve NF4", True),
    ("datasets", "TurkishMMLU ve kalibrasyon verisi", True),
    ("pandas", "tablolar", True),
    ("requests", "HarmBench/XSTest indirme", True),
    ("cryptography", "şifreli arşiv", False),
    ("scipy", "McNemar, z-testi", False),
    ("sklearn", "Cohen's kappa", False),
    ("gptqmodel", "GPTQ ve AWQ üretimi", False),
    ("sentencepiece", "çeviri modeli tokenizer'ı", False),
]

OK, EKSIK, UYARI = "  [+]", "  [!]", "  [~]"


def main() -> int:
    hata = 0
    print("=== Yollar ===")
    print(f"  ROOT (çıktılar)   : {ortak.ROOT}")
    print(f"  REPO_DIR (kaynak) : {ortak.REPO_DIR}")
    print(f"  veri/             : {ortak.DATA_DIR}")
    print(f"  sonuclar/         : {ortak.OUT_DIR}")

    print("\n=== Gereken veri dosyaları ===")
    for name, why in GEREKLI_VERI:
        try:
            print(f"{OK} {name}  ->  {ortak.find_file(name)}")
        except FileNotFoundError:
            print(f"{EKSIK} {name} YOK  ({why})")
            hata += 1

    print("\n=== Üretilen veri dosyaları ===")
    for name, how in URETILEN_VERI:
        try:
            print(f"{OK} {name}  ->  {ortak.find_file(name)}")
        except FileNotFoundError:
            print(f"{UYARI} {name} yok  ({how})")

    print("\n=== Paketler ===")
    for mod, why, required in PAKETLER:
        try:
            m = importlib.import_module(mod)
            print(f"{OK} {mod} {getattr(m, '__version__', '?')}  ({why})")
        except ImportError:
            if required:
                print(f"{EKSIK} {mod} YOK  ({why})")
                hata += 1
            else:
                print(f"{UYARI} {mod} yok  ({why}) - yalnızca bu adım için gerekli")

    print("\n=== GPU ===")
    gpus = ortak.gpu_names()
    if not gpus:
        print(f"{EKSIK} GPU yok. Colab: Runtime > Change runtime type > GPU")
        hata += 1
    else:
        import torch

        total = 0.0
        for i, name in enumerate(gpus):
            gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            total += gb
            print(f"{OK} cuda:{i} {name} {gb:.1f} GB")
        # FP16 8B modelin ölçülen ayak izi 14,96 GB; KV önbelleğiyle ~16 GB ister.
        if total < 17:
            print(f"{UYARI} Toplam {total:.1f} GB. FP16 referansı (~15 GB + KV önbellek) "
                  "bu belleğe sığmaz; FP16 koşusunu Kaggle'da 2xT4 ile yapın. "
                  "INT8, NF4, GPTQ ve AWQ seviyeleri burada çalışır.")
        else:
            print(f"{OK} Toplam {total:.1f} GB, bütün seviyeler için yeterli.")

    print("\n=== Kuantize ağırlıklar ===")
    for model_key in ortak.MODELS:
        for variant in sorted(ortak.CALIBRATED):
            path, remote = ortak.quant_path(model_key, variant)
            if remote:
                print(f"{OK} {model_key}/{variant}: uzak repo {path}")
            elif os.path.isdir(path) and os.listdir(path):
                print(f"{OK} {model_key}/{variant}: {path}")
            else:
                print(f"{UYARI} {model_key}/{variant} yok  "
                      f"(python kod/kuantize_gptq_awq.py --model {model_key} --yontem {variant})")

    print("\n=== Token ===")
    print(f"{OK} HF_TOKEN bulundu" if ortak.hf_login() else
          f"{UYARI} HF_TOKEN yok (erişim izni gereken model/veri seti için gerekir)")
    print(f"{OK} PROJE_SIFRE bulundu" if os.environ.get("PROJE_SIFRE") else
          f"{UYARI} PROJE_SIFRE yok (şifreli arşiv üretilemez, --sifreleme-yok gerekir)")

    print("\n" + ("=" * 50))
    if hata:
        print(f"{hata} zorunlu eksik var; yukarıdaki [!] satırlarını giderin.")
    else:
        print("Zorunlu kontrollerin hepsi geçti.")
    return 1 if hata else 0


if __name__ == "__main__":
    sys.exit(main())
