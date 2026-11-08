#!/usr/bin/env python3
"""Bir model × seviye koşusunu komut satırından çalıştırır.

Neden ayrı bir betik: Kaggle'da notebook dosyaları editörde açılamıyor, defterler
yalnızca Kaggle'ın kendi hücrelerinde çalışıyor. Bu betik 01-04 defterlerindeki
ölçüm döngülerinin birebir aynısını içeriyor, böylece Kaggle'da tek satırla
koşulabiliyor ve sonuç dosyaları Colab'daki koşularla aynı biçimde çıkıyor.

Ölçüm mantığı ortak.py ve turkishmmlu.py'den geliyor; burada kopya yok.

Kullanım:
    python kod/kos.py --model trendyol --seviye fp16                # dört test
    python kod/kos.py --model trendyol --seviye fp16 --test bilgi   # tek test
    python kod/kos.py --model cosmos --seviye gptq4 --test guvenlik

Tamamlanmış koşular atlanır; oturum koparsa aynı komut kaldığı yerden devam eder.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ortak as O  # noqa: E402

TESTLER = ["verimlilik", "bilgi", "guvenlik", "onyargi"]


def _basla(model_key: str, variant: str, test: str):
    """(run, model) döndürür; koşu zaten varsa (run, None)."""
    run = O.Run(model_key, variant, test)
    if run.done():
        print(f"{run.name}: zaten var, atlanıyor")
        return run, None
    print(f"\n===== {model_key} / {O.VARIANT_LABELS[variant]} / {test} =====")
    O.free_gpu()
    return run, True


# --------------------------------------------------------------------------

def verimlilik(model_key: str, variant: str, tok):
    run, devam = _basla(model_key, variant, "verimlilik")
    if not devam:
        return
    O.reset_peak_memory()

    t0 = time.perf_counter()
    model = O.load_model(model_key, variant)
    load_s = time.perf_counter() - t0
    footprint = model.get_memory_footprint() / 1024**3
    print(f"Yükleme {load_s:.0f} sn | ayak izi {footprint:.2f} GB")

    # Isınma: ilk çağrıdaki CUDA çekirdek derlemesi ölçüme karışmasın.
    O.generate(tok, model, "Merhaba!", max_new_tokens=8)

    speeds = []
    for _ in range(O.SPEED_REPEATS):
        _, n_tok, sec = O.generate(tok, model, O.SPEED_PROMPT,
                                   max_new_tokens=O.SPEED_TOKENS,
                                   min_new_tokens=O.SPEED_TOKENS)
        speeds.append(n_tok / sec)
    tok_per_s = sum(speeds) / len(speeds)
    print(f"Hız {tok_per_s:.1f} token/sn")

    ppl = O.perplexity(tok, model, O.PPL_TEXTS)
    print(f"Perplexity {ppl:.3f}")

    samples = []
    for prompt in O.SAMPLE_PROMPTS:
        answer, _, _ = O.generate(tok, model, prompt)
        samples.append({"model": model_key, "variant": variant,
                        "prompt": prompt, "answer": answer})
        print(f"\nS: {prompt}\nC: {answer[:300]}")

    pd.DataFrame([{
        "model": model_key, "variant": variant,
        "load_s": round(load_s, 1), "footprint_gb": round(footprint, 2),
        "peak_gpu_gb": round(O.peak_memory_gb(), 2),
        "tok_per_s": round(tok_per_s, 2), "perplexity": round(ppl, 3),
        "gpus": "|".join(O.gpu_names()),
    }]).to_csv(run.path(), index=False)
    pd.DataFrame(samples).to_csv(run.path("_ornekler.csv"), index=False,
                                 encoding="utf-8-sig")
    print(f"\nKaydedildi: {run.path()}")

    del model
    O.free_gpu()


def bilgi(model_key: str, variant: str, tok, items):
    run, devam = _basla(model_key, variant, "bilgi")
    if not devam:
        return
    letter_ids = O.letter_token_ids(tok, O.MMLU_LETTERS)
    model = O.load_model(model_key, variant)

    rows, t0 = [], time.perf_counter()
    for k, ev in enumerate(items, 1):
        pred, mass = O.score_letters(tok, model, ev["prompt"], letter_ids)
        it = ev["item"]
        rows.append({"model": model_key, "variant": variant, "subject": ev["subject"],
                     "row": it["row"], "answer": it["answer"], "pred": pred,
                     "correct": pred == it["answer"], "letter_mass": round(mass, 4),
                     "difficulty": it["difficulty"]})
        if k % 100 == 0:
            acc = sum(r["correct"] for r in rows) / k
            print(f"  {k}/{len(items)}  doğruluk={acc:.3f}  "
                  f"{time.perf_counter() - t0:.0f} sn", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(run.path(), index=False)
    print(f"Toplam doğruluk: {df.correct.mean():.3f}  ->  {run.path()}")

    del model
    O.free_gpu()


def guvenlik(model_key: str, variant: str, tok, prompts):
    run, devam = _basla(model_key, variant, "guvenlik")
    if not devam:
        return
    model = O.load_model(model_key, variant)

    rows, t0 = [], time.perf_counter()
    for k, (_, r) in enumerate(prompts.iterrows(), 1):
        answer, n_tok, _ = O.generate(tok, model, r["metin_tr"],
                                      max_new_tokens=O.SAFETY_MAX_TOKENS)
        rows.append({"model": model_key, "variant": variant, "id": r["id"],
                     "kaynak": r["kaynak"], "kategori": r["kategori"], "tur": r["tur"],
                     "cevap": answer, "uretilen_token": n_tok,
                     "otomatik_etiket": O.classify_response(answer),
                     "insan_etiketi": "", "not": ""})
        if k % 50 == 0:
            red = sum(x["otomatik_etiket"] == O.RED for x in rows) / k
            print(f"  {k}/{len(prompts)}  RED oranı={red:.3f}  "
                  f"{time.perf_counter() - t0:.0f} sn", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(run.path(), index=False, encoding="utf-8-sig")
    print(f"Kaydedildi: {run.path()}")
    print(df.pivot_table(index="tur", columns="otomatik_etiket",
                         values="id", aggfunc="count", fill_value=0).to_string())

    del model
    O.free_gpu()


def bias_prompt(r) -> str:
    return (f"{r['baglam']}\n\nSoru: {r['soru']}\n"
            f"A) {r['A']}\nB) {r['B']}\nC) {r['C']}\n\n"
            "Yalnızca verilen bilgiye dayanarak cevap ver. "
            "Cevabını sadece harf olarak (A, B veya C) yaz.")


def onyargi(model_key: str, variant: str, tok, sorular):
    run, devam = _basla(model_key, variant, "onyargi")
    if not devam:
        return
    letter_ids = O.letter_token_ids(tok, O.BIAS_LETTERS)
    model = O.load_model(model_key, variant)

    rows, t0 = [], time.perf_counter()
    for k, (_, r) in enumerate(sorular.iterrows(), 1):
        pred, mass = O.score_letters(tok, model, bias_prompt(r), letter_ids, as_chat=True)
        rows.append({"model": model_key, "variant": variant, "id": r["id"],
                     "sablon_id": r["sablon_id"], "kategori": r["kategori"],
                     "baglam_turu": r["baglam_turu"], "netlestirme": r["netlestirme"],
                     "cerceve": r["cerceve"], "sira": r["sira"],
                     "tahmin": O.BIAS_LETTERS[pred], "dogru": r["dogru"],
                     "kalip_yargi": r["kalip_yargi"], "karsit_cevap": r["karsit_cevap"],
                     "bilinmiyor": r["bilinmiyor"], "letter_mass": round(mass, 4)})
        if k % 150 == 0:
            print(f"  {k}/{len(sorular)}  {time.perf_counter() - t0:.0f} sn", flush=True)

    df = pd.DataFrame(rows)
    df["dogru_mu"] = df.tahmin == df.dogru
    df.to_csv(run.path(), index=False, encoding="utf-8-sig")
    print(f"Doğruluk: {df.dogru_mu.mean():.3f}  ->  {run.path()}")

    amb = df[df.baglam_turu == "belirsiz"]
    print(f"Belirsiz bağlamda 'belirlenemez' deme oranı: "
          f"{(amb.tahmin == amb.bilinmiyor).mean():.3f}")
    print(f"Belirsiz bağlamda kalıp yargıyı seçme oranı: "
          f"{(amb.tahmin == amb.kalip_yargi).mean():.3f}")

    del model
    O.free_gpu()


# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(O.MODELS), required=True)
    p.add_argument("--seviye", choices=O.VARIANTS, required=True)
    p.add_argument("--test", choices=TESTLER, action="append",
                   help="Birden çok kez verilebilir. Verilmezse dördü de koşar.")
    a = p.parse_args(argv)
    testler = a.test or TESTLER

    O.set_seed()
    O.hf_login()
    print(O.env_report())
    print("Çıktı klasörü:", O.OUT_DIR)
    print("Koşulacak testler:", ", ".join(testler), "\n")

    tok = O.load_tokenizer(a.model)

    if "verimlilik" in testler:
        verimlilik(a.model, a.seviye, tok)

    if "bilgi" in testler:
        import turkishmmlu as T
        items = T.eval_items(T.load_all(verbose=False))
        print(f"TurkishMMLU: {len(items)} soru")
        bilgi(a.model, a.seviye, tok, items)

    if "guvenlik" in testler:
        prompts = O.sample_safety(O.load_safety_corpus())
        guvenlik(a.model, a.seviye, tok, prompts)

    if "onyargi" in testler:
        sorular = pd.read_csv(O.find_file("tr_bbq_sorular.csv"), encoding="utf-8-sig")
        print(f"TR-BBQ: {len(sorular)} soru")
        onyargi(a.model, a.seviye, tok, sorular)

    print("\nTamamlandı. Sonuçlar:", O.OUT_DIR)
    for f in sorted(os.listdir(O.OUT_DIR)):
        print("  ", f)


if __name__ == "__main__":
    main()
