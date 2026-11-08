"""TurkishMMLU veri setini yükler, normalleştirir ve 5-shot istemleri kurar.

Ayrı bir modül olmasının sebebi: veri setinin sütun adları sürümler arasında
değişebiliyor, ayrıştırma mantığının tek yerde ve test edilebilir olması
gerekiyor. Notebook yalnızca çağırıyor.

Yüksel vd., TurkishMMLU: Measuring Massive Multitask Language Understanding in
Turkish, Findings of EMNLP 2024. 9 ders, 5 şık (A-E), ders başına 25 soruluk
dev bölümü, 9.807 test sorusu.
"""

from __future__ import annotations

import ast
import json
import random
import re

from ortak import MMLU_LETTERS, N_PER_SUBJECT, N_SHOT, SEED

DATASET_ID = "AYueksel/TurkishMMLU"

# Ders adları istemde Türkçe görünsün: model Türkçe bir sınav bağlamı görüyor.
SUBJECT_TR = {
    "Biology": "Biyoloji",
    "Chemistry": "Kimya",
    "Geography": "Coğrafya",
    "History": "Tarih",
    "Mathematics": "Matematik",
    "Philosophy": "Felsefe",
    "Physics": "Fizik",
    "Religion_and_Ethics": "Din Kültürü ve Ahlak Bilgisi",
    "Turkish_Language_and_Literature": "Türk Dili ve Edebiyatı",
}

QUESTION_KEYS = ["question", "soru", "Question"]
CHOICE_LIST_KEYS = ["choices", "options", "secenekler"]
ANSWER_KEYS = ["answer", "label", "correct_answer", "cevap", "Answer"]
DIFFICULTY_KEYS = ["difficulty", "correctness_ratio", "Correctness_Ratio"]

PREFIX_RE = re.compile(r"^\s*[A-Ea-e]\s*[\)\.:-]\s*")


# --------------------------------------------------------------------------
# Ayrıştırma
# --------------------------------------------------------------------------

def pick(row: dict, keys: list[str]):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    raise KeyError(f"Bu sütunlardan hiçbiri yok: {keys}. Mevcut sütunlar: {list(row)}")


def get_choices(row: dict) -> list[str]:
    """Şıkları liste, dize-liste ya da ayrı sütun biçimlerinden çıkarır."""
    choices = None
    for k in CHOICE_LIST_KEYS:
        if k not in row or row[k] is None:
            continue
        val = row[k]
        if isinstance(val, str) and val.strip().startswith("["):
            choices = ast.literal_eval(val)
        elif isinstance(val, (list, tuple)):
            choices = list(val)
        if choices is not None:
            break
    if choices is None:
        # Şıklar ayrı sütunlarda olabilir: A, B, ... ya da a, b, ...
        keys = [L for L in MMLU_LETTERS if L in row] or \
               [L.lower() for L in MMLU_LETTERS if L.lower() in row]
        if not keys:
            raise KeyError(f"Şıklar bulunamadı. Mevcut sütunlar: {list(row)}")
        choices = [row[k] for k in keys]
    return [PREFIX_RE.sub("", str(c)).strip() for c in choices]


def answer_index(value, choices: list[str], one_based: bool) -> int:
    """Cevabı harf, sayı ya da şık metni biçiminden indekse çevirir."""
    if isinstance(value, str):
        v = value.strip()
        if len(v) == 1 and v.upper() in MMLU_LETTERS:
            return MMLU_LETTERS.index(v.upper())
        if v.isdigit():
            value = int(v)
        elif v in choices:
            return choices.index(v)
        else:
            raise ValueError(f"Cevap anlaşılamadı: {value!r}")
    return int(value) - (1 if one_based else 0)


def normalize_split(split) -> list[dict]:
    """Bir bölümü {question, choices, answer, difficulty} biçimine getirir."""
    rows = list(split)
    raw = [pick(r, ANSWER_KEYS) for r in rows]
    ints = [int(a) for a in raw
            if isinstance(a, int) or (isinstance(a, str) and a.strip().isdigit())]
    # Tamsayı cevaplarda 0 hiç yoksa ve şık sayısı kadar büyük değer varsa,
    # cevaplar 1'den başlıyordur.
    one_based = bool(ints) and min(ints) >= 1 and max(ints) == len(MMLU_LETTERS)

    out = []
    for i, (r, a) in enumerate(zip(rows, raw)):
        choices = get_choices(r)
        idx = answer_index(a, choices, one_based)
        if not 0 <= idx < len(choices):
            raise ValueError(f"satır {i}: cevap indeksi aralık dışı ({a!r} -> {idx})")
        difficulty = None
        for k in DIFFICULTY_KEYS:
            if k in r and r[k] is not None:
                difficulty = r[k]
                break
        out.append({"row": i, "question": str(pick(r, QUESTION_KEYS)).strip(),
                    "choices": choices, "answer": idx, "difficulty": difficulty})
    return out


def load_subjects() -> list[str]:
    from datasets import get_dataset_config_names

    subjects = sorted(get_dataset_config_names(DATASET_ID))
    # Toplu "all"/"default" yapılandırmaları varsa ders bazlı olanlar tercih edilir.
    subjects = [s for s in subjects if s.lower() not in ("all", "default")] or subjects
    return subjects


def load_all(verbose: bool = True) -> dict[str, dict[str, list[dict]]]:
    from datasets import load_dataset

    data = {}
    for s in load_subjects():
        raw = load_dataset(DATASET_ID, s)
        data[s] = {"dev": normalize_split(raw["dev"]), "test": normalize_split(raw["test"])}
        if verbose:
            print(f"  {s:35s} dev={len(data[s]['dev']):3d} test={len(data[s]['test']):5d}")
    if verbose:
        first = next(iter(data))
        print("\nAyrıştırılmış örnek satır (doğruluğunu kontrol edin):")
        print(json.dumps(data[first]["test"][0], ensure_ascii=False, indent=1))
    return data


# --------------------------------------------------------------------------
# İstem kurma
# --------------------------------------------------------------------------

def format_question(item: dict, with_answer: bool) -> str:
    lines = [f"Soru: {item['question']}"]
    lines += [f"{MMLU_LETTERS[i]}) {c}" for i, c in enumerate(item["choices"])]
    lines.append("Cevap:" + (f" {MMLU_LETTERS[item['answer']]}" if with_answer else ""))
    return "\n".join(lines)


def build_prompt(subject: str, item: dict, shots: list[dict]) -> str:
    ders = SUBJECT_TR.get(subject, subject.replace("_", " "))
    header = f"Aşağıdaki çoktan seçmeli sorular {ders} dersine aittir.\n\n"
    parts = [format_question(s, True) for s in shots] + [format_question(item, False)]
    return header + "\n\n".join(parts)


def eval_items(data: dict, n_per_subject: int | None = N_PER_SUBJECT,
               n_shot: int = N_SHOT, seed: int = SEED) -> list[dict]:
    """Değerlendirilecek soruları sabit tohumla örnekler ve istemlerini kurar.

    Her model x seviye koşusu birebir aynı soruları görür.
    """
    rng = random.Random(seed)
    items = []
    for subject in sorted(data):
        test = data[subject]["test"]
        chosen = test if n_per_subject is None else \
            rng.sample(test, min(n_per_subject, len(test)))
        shots = data[subject]["dev"][:n_shot]
        if len(shots) < n_shot:
            print(f"Uyarı: {subject} dev bölümünde {len(shots)} örnek var ({n_shot} istendi).")
        for item in chosen:
            items.append({"subject": subject, "item": item,
                          "prompt": build_prompt(subject, item, shots)})
    return items
