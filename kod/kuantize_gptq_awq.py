#!/usr/bin/env python3
"""GPTQ ve AWQ ağırlıklarını Türkçe kalibrasyon verisiyle üretir.

Neden ayrı bir betik: bitsandbytes seviyeleri (INT8, NF4) model yüklenirken
anında uygulanır, GPTQ ve AWQ ise kalibrasyon ister ve bir kez üretilip
kaydedilir. Bu betik o bir kerelik adımı yapar; değerlendirme notebook'ları
sonucu `ortak.resolve_source()` ile yükler.

Kalibrasyon dili kararı: GPTQModel örnekleri İngilizce C4 kullanır. Bu projede
kalibrasyon verisi **Türkçe** seçildi, çünkü hem değerlendirme dili Türkçe hem
de kalibrasyon dilinin çok dilli modellerde sonucu etkilediği biliniyor
(Marchisio vd., 2024). Bu bir yöntem kararıdır ve raporda tartışılır.
İngilizce ile karşılaştırma yapmak isterseniz: --kalibrasyon-dili en

Araç kararı: AutoAWQ Mayıs 2025'te resmen bakımdan düştü. GPTQModel hem
METHOD.GPTQ hem METHOD.AWQ destekliyor ve Marlin/JIT çekirdekleri Turing+
(sm_75+), yani T4 dahil çalışıyor. Tek kütüphane iki yöntemi veriyor.

Kurulum (Colab):
    pip install -v gptqmodel --no-build-isolation

Kullanım:
    python kuantize_gptq_awq.py --model trendyol --yontem gptq4
    python kuantize_gptq_awq.py --model trendyol --yontem awq4
    python kuantize_gptq_awq.py --model cosmos --yontem gptq4 --yukle kullanici/repo

Süre: 8B model + 512 kalibrasyon örneği ile T4'te yaklaşık 40-70 dakika.
Bellek: kuantizasyon sırasında katman katman çalışır, 16 GB yeterlidir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import MODELS, QUANT_DIR, SEED, hf_login, quant_path  # noqa: E402

# GPTQModel'in iş parçacığı havuzu, CPU çekirdek sayısının yarısı kadar işçi
# ayırıyor ((cpu_count + 1) // 2) ama model yükleyici için sabit 2 istiyor.
# Colab'ın ücretsiz katmanında 2 vCPU var, yani bütçe 1 çıkıyor ve kütüphane
# "model_loader:cpu requests 2 workers but parent cpu only has 1" diyerek
# model yüklenirken çöküyor. Bütçeyi açıkça veriyoruz; kullanıcı kendi değerini
# verdiyse ona dokunulmuyor.
os.environ.setdefault("GPTQMODEL_CPU_WORKERS", str(max(2, (os.cpu_count() or 2))))

# Kalibrasyon kaynakları. Türkçe için Vikipedi: ansiklopedik, temiz ve
# değerlendirme setlerinin (TurkishMMLU, TR-BBQ) diline yakın.
CALIB_SOURCES = {
    "tr": dict(path="wikimedia/wikipedia", name="20231101.tr", split="train", column="text"),
    "en": dict(path="wikimedia/wikipedia", name="20231101.en", split="train", column="text"),
}

# GPTQ ve AWQ'nun ortak ayarları: ikisi de 4-bit, group_size 128.
# Aynı tutulması şart, yoksa fark yöntemden değil ayardan gelir.
BITS = 4
GROUP_SIZE = 128
N_CALIB = 512
MIN_CHARS = 512      # çok kısa maddeler kalibrasyon için işe yaramaz
MAX_CHARS = 4096     # çok uzun maddeler belleği şişirir


def load_calibration(lang: str, n: int = N_CALIB, seed: int = SEED) -> list[str]:
    """Kalibrasyon metinlerini akış hâlinde okur (veri setinin tamamını indirmez)."""
    from datasets import load_dataset

    src = CALIB_SOURCES[lang]
    print(f"Kalibrasyon verisi: {src['path']} / {src['name']} ({lang})")
    ds = load_dataset(src["path"], src["name"], split=src["split"], streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    texts: list[str] = []
    for row in ds:
        t = (row.get(src["column"]) or "").strip()
        if MIN_CHARS <= len(t) <= MAX_CHARS:
            texts.append(t)
            if len(texts) >= n:
                break
    if len(texts) < n:
        print(f"Uyarı: yalnızca {len(texts)} örnek bulundu (istenen {n}).")
    print(f"{len(texts)} kalibrasyon örneği, ortalama {sum(map(len, texts)) // len(texts)} karakter")
    return texts


def build_config(method: str):
    """GPTQ ya da AWQ yapılandırması. Ortak ayarlar BITS/GROUP_SIZE'dan gelir."""
    from gptqmodel import AWQConfig, GPTQConfig

    if method == "gptq4":
        # desc_act=True niceleme hatasını azaltır ama bazı çekirdeklerde
        # yavaşlatır; kalite öncelikli olduğu için açık.
        return GPTQConfig(bits=BITS, group_size=GROUP_SIZE, desc_act=True)
    if method == "awq4":
        # sym=True, AutoAWQ'nun zero_point=False ayarının karşılığıdır.
        return AWQConfig(bits=BITS, group_size=GROUP_SIZE, sym=True)
    raise ValueError(f"Bilinmeyen yöntem: {method}")


def quantize(model_key: str, method: str, lang: str, n_calib: int,
             batch_size: int, out_dir: str | None) -> str:
    import torch
    from gptqmodel import GPTQModel

    model_id = MODELS[model_key]
    out = out_dir or quant_path(model_key, method)[0]
    if os.path.isdir(out) and os.listdir(out):
        print(f"Zaten var, atlanıyor: {out}\n(Yeniden üretmek için klasörü silin.)")
        return out

    hf_login()
    calib = load_calibration(lang, n_calib)
    config = build_config(method)

    # Çekirdek seçimi iki ayrı yerde gerekiyor ve ikisi aynı şey değil:
    #
    # 1) PAKETLEME (quantize sırasında): AWQ'nun yerel CUDA eklentisi
    #    (gptqmodel_awq_ops) sm_80 istiyor; gemm_cuda_gen.cu bf16 yolunda
    #    "requires CUDA_ARCH >= 800" diye static_assert atıyor,
    #    gemm_fast_cuda_entry.cu ise cp.async kullanıyor. T4 sm_75 olduğu için
    #    hiçbiri derlenmiyor ve hata quantize() çağrısında çıkıyor.
    #    gptqmodel'in kendi çözümü var: awq_torch çekirdeği aynı GEMM
    #    düzenini saf torch ile üretiyor (utils/backend.py:awq_gemm_pack_backend
    #    bunu ROCm/CPU için zaten böyle yapıyor). Biz de Turing'de onu
    #    kullanıyoruz; paketlenen ağırlıklar bit bit aynı.
    # 2) KOŞMA (yükleme sırasında): awq_gemm_triton Turing'de çalışıyor.
    pack_backend = os.environ.get("GPTQ_PACK_BACKEND") or (
        "awq_torch" if method == "awq4" else None)
    varsayilan_run = "awq_gemm_triton" if method == "awq4" else "gptq_exllama_v2"
    run_backend = os.environ.get("GPTQ_BACKEND") or varsayilan_run
    # GPTQ koşularından kalan GPTQ_BACKEND=gptq_* değeri AWQ ağırlıklarını
    # yükleyemez (ve tersi), bu yüzden yönteme uymayan değer yok sayılıyor.
    if method == "awq4" and not run_backend.startswith("awq"):
        print(f"UYARI: GPTQ_BACKEND={run_backend} awq4 için geçersiz, "
              f"{varsayilan_run} kullanılıyor.")
        run_backend = varsayilan_run
    if method == "gptq4" and not run_backend.startswith("gptq"):
        print(f"UYARI: GPTQ_BACKEND={run_backend} gptq4 için geçersiz, "
              f"{varsayilan_run} kullanılıyor.")
        run_backend = varsayilan_run
    print(f"Paketleme çekirdeği: {pack_backend or 'AUTO'}  |  "
          f"koşma çekirdeği: {run_backend}")

    print(f"\n{model_id} -> {method} (bits={BITS}, group_size={GROUP_SIZE}, kalibrasyon={lang})")
    t0 = time.perf_counter()
    # Kaynak model FP16 yükleniyor, burada kuantize çekirdeği devreye girmiyor.
    model = GPTQModel.load(model_id, config, dtype=torch.float16)
    kuantize_kw = {"backend": pack_backend} if pack_backend else {}
    model.quantize(calib, batch_size=batch_size, **kuantize_kw)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    model.save(out)
    elapsed = time.perf_counter() - t0
    print(f"\nTamamlandı: {out}  ({elapsed / 60:.1f} dakika)")

    # Rapora yazılacak üretim bilgisi; ağırlıkların yanında saklanır.
    meta = dict(model_id=model_id, model_key=model_key, yontem=method, bits=BITS,
                group_size=GROUP_SIZE, kalibrasyon_dili=lang, kalibrasyon_n=len(calib),
                kalibrasyon_kaynak=CALIB_SOURCES[lang], sure_dk=round(elapsed / 60, 1),
                seed=SEED, gpu=[torch.cuda.get_device_name(i)
                                for i in range(torch.cuda.device_count())])
    with open(os.path.join(out, "kuantizasyon_bilgisi.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Sağlık kontrolü: kuantize model yüklenip bir cevap üretebiliyor mu?
    print("\nSağlık kontrolü (kuantize ağırlıklarla üretim):")
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    check = GPTQModel.load(out, backend=run_backend)
    ids = check.generate("Türkiye'nin başkenti", max_new_tokens=16)[0]
    print("  ", check.tokenizer.decode(ids).strip()[:200])
    return out


def upload(path: str, repo: str, model_key: str, method: str):
    """Ağırlıkları Hugging Face'e özel repo olarak yükler (depoya ağırlık koymamak için)."""
    from huggingface_hub import HfApi

    hf_login()
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=path, repo_id=repo)
    print(f"Yüklendi (özel): https://huggingface.co/{repo}")
    print("Değerlendirmede kullanmak için:\n  export "
          f"{method.upper()}_{model_key.upper()}={repo}")


def main(argv=None):
    # allow_abbrev=False: argparse varsayılan olarak kısaltmayı kabul ediyor ve
    # örneğin --sifrele'yi sessizce --sifreleme-yok sanabiliyor. Yazım hatası
    # sessiz bir davranış değişikliği değil, hata vermeli.
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--yontem", choices=["gptq4", "awq4"], required=True)
    p.add_argument("--kalibrasyon-dili", choices=["tr", "en"], default="tr")
    p.add_argument("--n-kalibrasyon", type=int, default=N_CALIB)
    p.add_argument("--batch", type=int, default=1,
                   help="GPU belleği elverdiğince artırın; kuantizasyonu hızlandırır.")
    p.add_argument("--cikti", help=f"Varsayılan: {QUANT_DIR}/<model>-<yontem>")
    p.add_argument("--yukle", metavar="REPO",
                   help="Bitince Hugging Face'e özel repo olarak yükle.")
    a = p.parse_args(argv)

    out = quantize(a.model, a.yontem, a.kalibrasyon_dili, a.n_kalibrasyon, a.batch, a.cikti)
    if a.yukle:
        upload(out, a.yukle, a.model, a.yontem)


if __name__ == "__main__":
    main()
