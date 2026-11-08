#!/usr/bin/env python3
"""Güvenlik setlerini kaynağından indirir, Türkçeye çevirir, yerelleştirir ve şifreli kaydeder.

Tasarım kararı: zararlı istemler depoya elle yazılmaz. HarmBench ve XSTest
kamuya açık ve sürüm kontrollü kaynaklardır; bu betik onları çalışma anında
indirir, çevirir ve çıktıyı yalnızca `veri/gizli/` altına yazar. Depoya giren
tek şey istem metni içermeyen bir manifesto dosyasıdır (kimlik, kategori,
uzunluk, kontrol durumu). Böylece vize ölçütü 5'teki "zararlı istemler herkese
açık depoda yok" şartı sağlanır ve tekrar üretilebilirlik korunur: betiği
çalıştıran herkes aynı dosyayı yeniden üretir.

Kaynaklar:
  HarmBench  (MIT)        200 standart zararlı davranış, 6 kategori
  XSTest     (CC-BY-4.0)  250 güvenli + 200 zararlı istem, 18 tip

Kullanım:
    # 1) Önce kuru çalıştırma: indir, çevirmeden yapıyı gör
    python veri_hatti.py --cevirmen yok

    # 2) Gerçek koşu (GPU'lu ortamda, ~5 dk)
    python veri_hatti.py --cevirmen opus --ornek 40

    # 3) Kontrol örneğini doldurduktan sonra işaretle
    python veri_hatti.py --kontrol-isle veri/gizli/kontrol_ornegi.csv

    # 4) Düzeltmelerden sonra manifestoyu ve şifreli arşivi tazele
    python veri_hatti.py --sifrele

Şifreleme: PROJE_SIFRE ortam değişkeninden parola okunur.
    export PROJE_SIFRE="..."   # depoya yazılmaz, raporda yer almaz
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import random
import re
import sys
import zipfile

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import DATA_DIR, SECRET_DIR, SEED, find_file  # noqa: E402

HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"

HARMBENCH_TR = os.path.join(SECRET_DIR, "harmbench_tr.csv")
XSTEST_TR = os.path.join(SECRET_DIR, "xstest_tr.csv")
OZGUN_TR = os.path.join(SECRET_DIR, "guvenlik_seti.csv")  # Türkiye'ye özgü alt küme
MANIFEST = os.path.join(DATA_DIR, "guvenlik_manifesto.csv")
ARCHIVE = os.path.join(SECRET_DIR, "guvenlik_setleri.zip.enc")
SAMPLE = os.path.join(SECRET_DIR, "kontrol_ornegi.csv")

TRANSLATORS = {
    "opus": "Helsinki-NLP/opus-mt-tc-big-en-tr",
    "nllb": "facebook/nllb-200-distilled-1.3B",
}


# --------------------------------------------------------------------------
# İndirme
# --------------------------------------------------------------------------

def download(url: str) -> pd.DataFrame:
    print(f"İndiriliyor: {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    print(f"  {len(df)} satır, sütunlar: {list(df.columns)}")
    return df


def fetch_harmbench() -> pd.DataFrame:
    df = download(HARMBENCH_URL)
    # Yalnızca 'standard': copyright ve contextual davranışlar ek bağlam metni
    # ister, reddetme ölçümü için uygun değil.
    std = df[df["FunctionalCategory"] == "standard"].copy()
    return pd.DataFrame({
        "id": "HB_" + std["BehaviorID"].astype(str),
        "kaynak": "harmbench",
        "kategori": std["SemanticCategory"],
        "tur": "zararli",
        "metin_en": std["Behavior"].str.strip(),
    }).reset_index(drop=True)


def fetch_xstest() -> pd.DataFrame:
    df = download(XSTEST_URL)
    return pd.DataFrame({
        "id": "XS_" + df["id"].astype(str),
        "kaynak": "xstest",
        "kategori": df["type"],
        # safe = zararsız ama zararlı görünen kontrol istemi (aşırı reddetme ölçümü)
        "tur": df["label"].map({"safe": "zararsiz_kontrol", "unsafe": "zararli"}),
        "metin_en": df["prompt"].str.strip(),
    }).reset_index(drop=True)


def load_tr_kontrol() -> pd.DataFrame:
    """Türkçenin kendi çok anlamlılığına dayanan aşırı reddetme seti (AS3).

    XSTest'in "homonyms" kategorisi çeviriden sağ çıkmadığı için bu set onun
    yerini alıyor; ayrıntı veri/tr_kontrol_seti.csv başlığında.
    """
    try:
        path = find_file("tr_kontrol_seti.csv")
    except FileNotFoundError:
        print("Uyarı: tr_kontrol_seti.csv bulunamadı, Türkçe kontrol seti atlanıyor.")
        return pd.DataFrame(columns=["id", "kaynak", "kategori", "tur", "metin_en", "metin_tr"])
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"Türkçe kontrol seti: {len(df)} istem ({path})")
    return pd.DataFrame({
        "id": df["id"].astype(str),
        "kaynak": "tr_kontrol",
        "kategori": df["kategori"],
        "tur": "zararsiz_kontrol",
        "metin_en": "",
        "metin_tr": df["istem"].str.strip(),
    })


def load_ozgun() -> pd.DataFrame:
    """Daha önce yazılmış Türkiye'ye özgü alt küme (HarmBench'te karşılığı olmayan maddeler)."""
    candidates = [OZGUN_TR, os.path.join(DATA_DIR, "guvenlik_seti.csv")]
    try:
        candidates.append(find_file("guvenlik_seti.csv"))
    except FileNotFoundError:
        pass
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="utf-8-sig")
            out = pd.DataFrame({
                "id": "TR_" + df["id"].astype(str),
                "kaynak": "ozgun_tr",
                "kategori": df["kategori"],
                "tur": df["tur"],
                "metin_en": "",
                "metin_tr": df["istem"].str.strip(),
            })
            print(f"Türkiye'ye özgü alt küme: {len(out)} istem ({path})")
            return out
    print("Uyarı: Türkiye'ye özgü alt küme bulunamadı, atlanıyor.")
    return pd.DataFrame(columns=["id", "kaynak", "kategori", "tur", "metin_en", "metin_tr"])


# --------------------------------------------------------------------------
# Çeviri
# --------------------------------------------------------------------------

def translate_batch(texts: list[str], model_id: str, batch_size: int = 16) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Çeviri modeli yükleniyor: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.cuda().half()
    model.eval()

    gen_kwargs = {"max_new_tokens": 256, "num_beams": 4}
    # NLLB hedef dili bir token ile seçer.
    if "nllb" in model_id:
        tok.src_lang = "eng_Latn"
        tur = tok.convert_tokens_to_ids("tur_Latn")
        gen_kwargs["forced_bos_token_id"] = tur

    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  çevrildi {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return [t.strip() for t in out]


# --------------------------------------------------------------------------
# Yerelleştirme
# --------------------------------------------------------------------------

def load_localization_table(path: str | None = None) -> list[tuple[str, str, str]]:
    path = path or find_file("yerellestirme_tablosu.csv")
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("kaynak") and r.get("hedef"):
                rows.append((r["tur"], r["kaynak"].strip(), r["hedef"].strip()))
    # Uzun kaynakları önce uygula: "New York City" eşlemesi "New York"tan önce gelsin.
    rows.sort(key=lambda t: len(t[1]), reverse=True)
    print(f"Yerelleştirme tablosu: {len(rows)} kural ({path})")
    return rows


def localize(text: str, rules: list[tuple[str, str, str]]) -> tuple[str, list[str]]:
    """Kuralları uygular; (yeni metin, uygulanan kural listesi) döndürür.

    Kurallar hem İngilizce kaynak metne hem çeviri çıktısına uygulanabilir:
    çeviri modelleri kurum adlarını genellikle olduğu gibi bırakır.
    """
    applied = []
    for tur, src, dst in rules:
        pattern = re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(dst, text)
            applied.append(f"{tur}:{src}>{dst}")
    return text, applied


# --------------------------------------------------------------------------
# Şifreleme
# --------------------------------------------------------------------------

def _key(passphrase: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480_000)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))


def encrypt_files(paths: list[str], out_path: str, passphrase: str):
    """Dosyaları bir zip'e koyup şifreler. Çıktı: salt (16 bayt) + Fernet token."""
    from cryptography.fernet import Fernet

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if os.path.exists(p):
                z.write(p, os.path.basename(p))
    salt = os.urandom(16)
    token = Fernet(_key(passphrase, salt)).encrypt(buf.getvalue())
    with open(out_path, "wb") as f:
        f.write(salt + token)
    print(f"Şifreli arşiv: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")


def decrypt_archive(path: str, out_dir: str, passphrase: str):
    from cryptography.fernet import Fernet

    with open(path, "rb") as f:
        blob = f.read()
    salt, token = blob[:16], blob[16:]
    data = Fernet(_key(passphrase, salt)).decrypt(token)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(out_dir)
        names = z.namelist()
    print(f"Çözüldü -> {out_dir}: {names}")
    return names


def get_passphrase(required: bool = True) -> str | None:
    pw = os.environ.get("PROJE_SIFRE")
    if not pw and required:
        print("UYARI: PROJE_SIFRE tanımlı değil, şifreli arşiv üretilmedi.\n"
              "  export PROJE_SIFRE='...'  (depoya yazmayın)")
    return pw


# --------------------------------------------------------------------------
# Manifesto (depoya giren, metin içermeyen özet)
# --------------------------------------------------------------------------

def write_manifest(df: pd.DataFrame, path: str):
    man = pd.DataFrame({
        "id": df["id"],
        "kaynak": df["kaynak"],
        "kategori": df["kategori"],
        "tur": df["tur"],
        "karakter": df["metin_tr"].str.len(),
        "yerellestirme": df["yerellestirme"].fillna("").str.count(";") + (
            df["yerellestirme"].fillna("") != ""
        ).astype(int),
        "insan_kontrolu": (df["insan_kontrolu"] if "insan_kontrolu" in df.columns
                           else pd.Series("", index=df.index)),
        # Metnin kendisi değil, özeti: aynı dosyanın üretildiğini doğrulamaya yarar.
        "sha1_12": [hashlib.sha1(t.encode()).hexdigest()[:12] for t in df["metin_tr"]],
    })
    man.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Manifesto (istem metni içermez): {path} ({len(man)} satır)")


def write_gitignore():
    path = os.path.join(DATA_DIR, ".gitignore")
    lines = ["# Zararlı istem metinleri depoya girmez (vize ölçütü 5).",
             "gizli/", "*.enc.tmp", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Yazıldı: {path}")


# --------------------------------------------------------------------------
# Kontrol örneği
# --------------------------------------------------------------------------

def write_sample(df: pd.DataFrame, n: int, path: str):
    """İnsan doğrulaması için örneklem. Raporda hata oranı buradan verilir."""
    pool = df[df["metin_en"].str.len() > 0]
    n = min(n, len(pool))
    rng = random.Random(SEED)
    idx = rng.sample(list(pool.index), n)
    sample = df.loc[idx, ["id", "kaynak", "kategori", "tur", "metin_en", "metin_tr",
                          "yerellestirme"]].copy()
    # Kontrol eden kişinin dolduracağı sütunlar.
    sample["ceviri_dogru"] = ""      # E / H
    sample["yerellestirme_dogru"] = ""  # E / H
    sample["duzeltme"] = ""          # gerekiyorsa düzeltilmiş Türkçe
    sample["not"] = ""
    sample.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nKontrol örneği: {path} ({n} satır)")
    print("  'ceviri_dogru' ve 'yerellestirme_dogru' sütunlarını E/H olarak doldurun,")
    print("  hatalı satırlarda 'duzeltme' sütununa doğru Türkçeyi yazın.")


def _kontrol_sutunu(df: pd.DataFrame) -> pd.DataFrame:
    """insan_kontrolu sütununu metin tipine sabitler.

    Sütun tamamen boşsa pandas onu float64 (NaN) olarak okuyor; içine metin
    yazmak pandas 2.x'te FutureWarning veriyor, ileride TypeError verecek.
    """
    if "insan_kontrolu" not in df.columns:
        df["insan_kontrolu"] = ""
    df["insan_kontrolu"] = df["insan_kontrolu"].fillna("").astype(str).replace("nan", "")
    return df


def apply_sample(sample_path: str):
    """Doldurulmuş kontrol örneğini setlere işler ve hata oranını bildirir."""
    s = pd.read_csv(sample_path, encoding="utf-8-sig")
    filled = s[s["ceviri_dogru"].astype(str).str.strip().str.upper().isin(["E", "H"])]
    if filled.empty:
        print("Kontrol örneğinde doldurulmuş satır yok.")
        return
    err_ceviri = (filled["ceviri_dogru"].str.strip().str.upper() == "H").mean()
    yerel = filled[filled["yerellestirme_dogru"].astype(str).str.strip().str.upper().isin(["E", "H"])]
    err_yerel = ((yerel["yerellestirme_dogru"].str.strip().str.upper() == "H").mean()
                 if not yerel.empty else float("nan"))
    print(f"Doldurulmuş: {len(filled)} satır")
    print(f"  Çeviri hata oranı:        {err_ceviri:.1%}")
    print(f"  Yerelleştirme hata oranı: {err_yerel:.1%}")

    fixes = filled[filled["duzeltme"].notna() & (filled["duzeltme"].astype(str).str.strip() != "")]
    if fixes.empty:
        print("Düzeltme girilmemiş.")
        return
    for path in (HARMBENCH_TR, XSTEST_TR):
        if not os.path.exists(path):
            continue
        df = _kontrol_sutunu(pd.read_csv(path, encoding="utf-8-sig"))
        m = dict(zip(fixes["id"], fixes["duzeltme"].astype(str).str.strip()))
        hit = df["id"].isin(m)
        if hit.any():
            df.loc[hit, "metin_tr"] = df.loc[hit, "id"].map(m)
            df.loc[hit, "insan_kontrolu"] = "duzeltildi"
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  {hit.sum()} düzeltme işlendi -> {path}")
    # Kontrol edilip onaylananları da işaretle
    ok = filled[filled["ceviri_dogru"].str.strip().str.upper() == "E"]["id"]
    for path in (HARMBENCH_TR, XSTEST_TR):
        if not os.path.exists(path):
            continue
        df = _kontrol_sutunu(pd.read_csv(path, encoding="utf-8-sig"))
        hit = df["id"].isin(set(ok)) & (df["insan_kontrolu"] != "duzeltildi")
        if hit.any():
            df.loc[hit, "insan_kontrolu"] = "onaylandi"
            df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  {len(ok)} satır onaylandı olarak işaretlendi.")


# --------------------------------------------------------------------------
# Ana akış
# --------------------------------------------------------------------------

def insan_kontrollu_satirlar() -> int:
    """Mevcut setlerde insan kontrolünden geçmiş satır sayısı."""
    n = 0
    for path in (HARMBENCH_TR, XSTEST_TR, OZGUN_TR):
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if "insan_kontrolu" in df.columns:
            n += df["insan_kontrolu"].astype(str).isin(["duzeltildi", "onaylandi"]).sum()
    return int(n)


def build(translator: str, sample_n: int, no_encrypt: bool, table_path: str | None = None,
          force: bool = False):
    # Koruma: build() setleri sıfırdan yeniden çevirip üzerine yazıyor. Daha önce
    # --kontrol-isle ile işlenmiş insan düzeltmeleri varsa bu onları yok eder.
    # 2026-11-05'te tam olarak bu oldu (bir yazım hatası build'i tetikledi),
    # o yüzden artık kaza eseri çalıştırmak mümkün değil.
    kontrollu = insan_kontrollu_satirlar()
    if kontrollu and not force:
        sys.exit(
            f"DURDURULDU: mevcut setlerde insan kontrolünden geçmiş {kontrollu} satır var.\n"
            "build() setleri sıfırdan çevirip üzerine yazar ve bu düzeltmeleri siler.\n\n"
            "Ne yapmak istiyordunuz?\n"
            "  Arşiv/manifesto tazelemek  -> python veri_hatti.py --sifrele\n"
            "  Düzeltme işlemek           -> python veri_hatti.py --kontrol-isle <dosya>\n"
            "  Gerçekten sıfırdan kurmak  -> python veri_hatti.py --yeniden-kur\n"
            "    (önce veri/gizli/ klasörünü yedekleyin)"
        )

    rules = load_localization_table(table_path)

    hb = fetch_harmbench()
    xs = fetch_xstest()
    sets = {"harmbench": hb, "xstest": xs}

    for name, df in sets.items():
        print(f"\n=== {name}: {len(df)} istem ===")
        print(df["tur"].value_counts().to_dict())

        # 1) İngilizce metne yerelleştirme (kurum/şehir adları çeviriden sağ çıkar)
        pre = [localize(t, rules) for t in df["metin_en"]]
        df["metin_en_yerel"] = [p[0] for p in pre]
        applied_pre = [p[1] for p in pre]

        # 2) Çeviri
        if translator == "yok":
            print("Çeviri atlandı (--cevirmen yok): metin_tr = İngilizce metin.")
            df["metin_tr"] = df["metin_en_yerel"]
        else:
            df["metin_tr"] = translate_batch(list(df["metin_en_yerel"]), TRANSLATORS[translator])

        # 3) Çeviri çıktısına yerelleştirmeyi tekrar uygula
        post = [localize(t, rules) for t in df["metin_tr"]]
        df["metin_tr"] = [p[0] for p in post]
        df["yerellestirme"] = [";".join(a + b) for a, b in zip(applied_pre, [p[1] for p in post])]
        df["cevirmen"] = translator if translator != "yok" else ""
        df["insan_kontrolu"] = ""

        n_loc = (df["yerellestirme"] != "").sum()
        print(f"Yerelleştirme uygulanan istem: {n_loc}/{len(df)}")

    elle_yazilan = []
    for df_native, etiket in ((load_ozgun(), "ozgun"), (load_tr_kontrol(), "tr_kontrol")):
        if df_native.empty:
            continue
        df_native["metin_en_yerel"] = ""
        df_native["yerellestirme"] = ""
        df_native["cevirmen"] = "elle"
        df_native["insan_kontrolu"] = etiket
        elle_yazilan.append(df_native)

    cols = ["id", "kaynak", "kategori", "tur", "metin_en", "metin_tr",
            "yerellestirme", "cevirmen", "insan_kontrolu"]
    hb[cols].to_csv(HARMBENCH_TR, index=False, encoding="utf-8-sig")
    xs[cols].to_csv(XSTEST_TR, index=False, encoding="utf-8-sig")
    print(f"\nYazıldı: {HARMBENCH_TR}")
    print(f"Yazıldı: {XSTEST_TR}")
    for df_native in elle_yazilan:
        if df_native["kaynak"].iloc[0] == "ozgun_tr":
            df_native[cols].to_csv(OZGUN_TR, index=False, encoding="utf-8-sig")
            print(f"Yazıldı: {OZGUN_TR}")

    birlesik = pd.concat([hb[cols], xs[cols]] + [d[cols] for d in elle_yazilan],
                         ignore_index=True)
    print(f"\nToplam: {len(birlesik)} istem")
    print(birlesik.groupby(["kaynak", "tur"]).size().to_string())
    zararli = (birlesik["tur"] == "zararli").sum()
    kontrol = (birlesik["tur"] == "zararsiz_kontrol").sum()
    print(f"\nZararlı: {zararli}  |  Zararsız kontrol: {kontrol}")
    print(f"Ölçüt 3 (>=150 yerelleştirilmiş güvenlik istemi): "
          f"{'SAĞLANDI' if zararli >= 150 else 'EKSİK'}")

    write_manifest(birlesik, MANIFEST)
    write_gitignore()

    if sample_n:
        write_sample(birlesik, sample_n, SAMPLE)

    if not no_encrypt:
        pw = get_passphrase()
        if pw:
            encrypt_files([HARMBENCH_TR, XSTEST_TR, OZGUN_TR], ARCHIVE, pw)


def repack():
    """Düzeltilmiş setlerden manifestoyu ve şifreli arşivi yeniden üretir.

    --kontrol-isle setlerin metnini değiştiriyor; manifesto ise metnin sha1
    özetini ve insan_kontrolu sütununu taşıyor. Düzeltmeden sonra ikisi de
    eskiyor, bu yüzden indirme/çeviri adımını tekrarlamadan yalnızca paketleme
    adımı yeniden çalıştırılabilsin istendi (vize ölçütü 3 ve 5).
    """
    cols = ["id", "kaynak", "kategori", "tur", "metin_en", "metin_tr",
            "yerellestirme", "cevirmen", "insan_kontrolu"]
    parcalar = []
    for path in (HARMBENCH_TR, XSTEST_TR, OZGUN_TR):
        if not os.path.exists(path):
            print(f"Yok, atlanıyor: {path}")
            continue
        df = _kontrol_sutunu(pd.read_csv(path, encoding="utf-8-sig"))
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        parcalar.append(df[cols])
        print(f"Okundu: {path} ({len(df)} satır)")
    if not parcalar:
        sys.exit("Paketlenecek set yok. Önce --cevirmen ile veri hattını çalıştırın.")

    # Türkçe kontrol seti gizli/ altına yazılmıyor (zararsız, depoda açık duruyor)
    # ama koşu setinin parçası, o yüzden manifestoda yer almalı. build() onu
    # sayıyor; repack() de saymazsa iki manifesto birbirini tutmuyor.
    tr_kontrol = load_tr_kontrol()
    if len(tr_kontrol):
        tr_kontrol = tr_kontrol.copy()
        for c in cols:
            if c not in tr_kontrol.columns:
                tr_kontrol[c] = ""
        parcalar.append(tr_kontrol[cols])

    birlesik = pd.concat(parcalar, ignore_index=True)
    kontrol_edilen = (birlesik["insan_kontrolu"].isin(["duzeltildi", "onaylandi"])).sum()
    print(f"\nToplam {len(birlesik)} istem, {kontrol_edilen} tanesi insan kontrolünden geçmiş")

    write_manifest(birlesik, MANIFEST)
    write_gitignore()

    pw = get_passphrase()
    if pw:
        # Arşive yalnızca zararlı metin taşıyan setler girer; tr_kontrol_seti.csv
        # zararsız kontrollerden oluşuyor ve depoda açık durması gerekiyor.
        encrypt_files([HARMBENCH_TR, XSTEST_TR, OZGUN_TR], ARCHIVE, pw)
    else:
        print("Arşiv üretilmedi; PROJE_SIFRE tanımlayıp tekrar çalıştırın.")


def main(argv=None):
    # allow_abbrev=False: argparse varsayılan olarak kısaltmayı kabul ediyor ve
    # örneğin --sifrele'yi sessizce --sifreleme-yok sanabiliyor. Yazım hatası
    # sessiz bir davranış değişikliği değil, hata vermeli.
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cevirmen", choices=["opus", "nllb", "yok"], default="opus",
                   help="Çeviri modeli. 'yok': indir ve yapıyı göster, çevirme.")
    p.add_argument("--ornek", type=int, default=40,
                   help="İnsan kontrolü için örneklem büyüklüğü (0: üretme).")
    p.add_argument("--sifreleme-yok", action="store_true",
                   help="Şifreli arşiv üretme (yalnızca test için).")
    p.add_argument("--kontrol-isle", metavar="DOSYA",
                   help="Doldurulmuş kontrol örneğini setlere işle.")
    p.add_argument("--sifrele", action="store_true",
                   help="Düzeltilmiş setlerden manifestoyu ve şifreli arşivi yeniden üret "
                        "(indirme ve çeviri tekrarlanmaz).")
    p.add_argument("--coz", metavar="HEDEF_KLASOR",
                   help="Şifreli arşivi çöz (PROJE_SIFRE gerekir).")
    p.add_argument("--tablo", metavar="DOSYA",
                   help="yerellestirme_tablosu.csv yolu (veri/ klasörü ortamda yoksa).")
    p.add_argument("--yeniden-kur", action="store_true",
                   help="Setleri sıfırdan kur; mevcut insan düzeltmelerinin silinmesini "
                        "göze alır. Önce veri/gizli/ klasörünü yedekleyin.")
    a = p.parse_args(argv)

    if a.kontrol_isle:
        apply_sample(a.kontrol_isle)
        print("\nSetler değişti: manifesto ve şifreli arşiv artık eski.\n"
              "  export PROJE_SIFRE='...' && python kod/veri_hatti.py --sifrele")
        return
    if a.sifrele:
        repack()
        return
    if a.coz:
        pw = get_passphrase()
        if not pw:
            sys.exit("PROJE_SIFRE gerekli.")
        decrypt_archive(ARCHIVE, a.coz, pw)
        return

    build(a.cevirmen, a.ornek, a.sifreleme_yok, a.tablo, force=a.yeniden_kur)


if __name__ == "__main__":
    main()
