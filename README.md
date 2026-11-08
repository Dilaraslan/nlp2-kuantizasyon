# Kuantizasyonun Türkçe LLM'lerde Bilgi, Güvenlik ve Toplumsal Önyargı Üzerindeki Etkisi

Llama-3 tabanlı **iki** Türkçe 8B modelinde ağırlık kuantizasyonunun üç boyutta
ölçülmesi: TurkishMMLU ile bilgi doğruluğu, Türkçeye yerelleştirilmiş
HarmBench/XSTest ile güvenlik ve aşırı reddetme, bu çalışmada oluşturulan
TR-BBQ seti ile toplumsal önyargı.

| | |
|---|---|
| Modeller | `Trendyol/Llama-3-Trendyol-LLM-8b-chat-v2.0` (talimat ayarlı), `ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1` (DPO ile hizalanmış) |
| Seviyeler | FP16, INT8 (`LLM.int8()`), NF4, GPTQ-4bit. AWQ-4bit üretilememiştir ve bu bir bulgu olarak raporlanmıştır. |
| Donanım | Kaggle 2×T4 ve Colab T4 |
| Sabitler | Bütün koşullarda aynı istem biçimi ve aynı çözümleme ayarları (`do_sample=False`, 128 token) |

İkinci model rastgele seçilmemiştir: aynı Llama-3 tabanı üzerinde DPO ile
hizalanmış olduğu için "kuantizasyon mu, hizalama mı" karşıtlığını kurar.

## Ana bulgu, tek paragrafta

Kuantizasyonun bedeli **bit genişliğiyle değil yöntemle** ölçülmelidir. Aynı
bit genişliğindeki ve birbirine 0,13 GB uzaklıktaki bellek bütçesinde çalışan
NF4 ile GPTQ-4bit, bilgi doğruluğunda zıt kararlar vermektedir ve bu zıtlık
iki modelde de aynı yönde çıkmaktadır (NF4 için McNemar *p* = 0,029 ve
0,0001; GPTQ-4bit için *p* = 0,59 ve 0,20). Sıkıştırmanın en belirgin etkisi
doğrulukta değil belirsiz bağlamda **çekimser kalma** oranındadır, ama bu etki
modele bağlıdır: talimat ayarlı modelde çekimserlik %69,3'ten %45,3'e erirken
hizalanmış modelde dört seviyede %82,7–87,3 aralığında sabit kalmaktadır.
Güvenlik boyutunda hizalanmamış modelde taban etkisi vardır (FP16 hâlinde dahi
zararlı istemlerin %95'ine uyum); hizalanmış modelde iki 4-bit seviyesi
reddetmeyi ölçülebilir biçimde değiştirmemekte, INT8 ise hipotezin tersi
yönünde değiştirmektedir. Bellek ve doğruluk eşiklerini birlikte karşılayan
tek seviye iki modelde de GPTQ-4bit'tir.

## Klasörler

```
kod/        ölçüm hattı (aşağıda)
veri/       TR-BBQ setleri, yerelleştirme tablosu, etiketleme rehberi,
            Türkçede kurulmuş aşırı reddetme kontrol seti
sonuclar/   ham ölçümler, tablo1-9, iki grafik
sonuclar/yayin/  güvenlik koşularının metin taşımayan yayın kopyaları
rapor/      rapor.tex, kaynakca.bib
```

### kod/

| Dosya | İşi |
|---|---|
| `ortak.py` | ortak yükleme, kuantizasyon, istem biçimi, sınıflandırıcı |
| `veri_hatti.py` | HarmBench/XSTest indirme, çeviri, yerelleştirme, şifreleme |
| `tr_bbq_uret.py` | TR-BBQ sorularını şablonlardan üretir |
| `turkishmmlu.py` | TurkishMMLU yükleme ve puanlama |
| `kos.py` | komut satırından ölçüm: `--model`, `--seviye`, `--test` |
| `kuantize_gptq_awq.py` | GPTQ/AWQ ağırlık üretimi (Türkçe kalibrasyon) |
| `ceviri_kontrol.py` | aynı istemleri Türkçe ve İngilizce sorarak çeviri denetimi |
| `kontrol.py` | kurulum ve veri bütünlüğü denetimi |
| `yayina_hazirla.py` | depoya konulabilir kopyaları üretir (aşağıya bakın) |
| `01`–`05_*.ipynb` | aynı adımların not defteri hâlleri; `05_analiz` tabloları üretir |

## Etik ve veri açıklaması

Bu depo **hiçbir zararlı istem metni ve hiçbir model cevabı içermez.**

- Zararlı istemler HarmBench ve XSTest'ten koşu anında indirilip makineyle
  çevrilmekte, yerelleştirilmekte ve yalnızca `veri/gizli/` altına
  yazılmaktadır; bu klasör `.gitignore`'dadır.
- Depoda istem metni yerine iki manifesto vardır: `veri/guvenlik_manifesto.csv`
  korpusun tamamını (754 istem), `veri/guvenlik_seti_manifesto.csv` ise
  60 satırlık özgün_tr setini kapsar. İkisi de yalnızca id, kaynak, kategori,
  tür, uzunluk ve istem metninin sha1 özetini taşır.
- `veri/guvenlik_setleri.zip.enc`, setlerin şifreli arşividir: 16 baytlık tuz +
  Fernet, anahtar PBKDF2-HMAC-SHA256 ile 480.000 turda türetilir. Parola
  depoda ve raporda yer almaz, yazarda saklıdır. Arşiv bilerek depoya konur;
  şifresiz hâli (`veri/gizli/`) konmaz.
- Güvenlik koşularının çıktıları (`sonuclar/*_guvenlik.csv`) modelin zararlı
  istemlere verdiği cevapları taşıdığı için `.gitignore`'dadır.
  `kod/yayina_hazirla.py` bunların metin sütunlarını sha1 özeti ve uzunlukla
  değiştirip `sonuclar/yayin/` altına yazar; depoya giren budur.
  **Depoyu yayımlamadan önce bu betiği koşturun.**
- `veri/tr_kontrol_seti.csv` bilerek açıktır: içindeki istemler kuruluşu gereği
  zararsızdır (Türkçede kurulmuş aşırı reddetme kontrolleri) ve yöntemsel bir
  katkıdır.
- Yerelleştirme, HarmBench'teki gerçek kişileri Türkiye'deki gerçek kişilere
  **eşlemez**; kişi adı kuralları yalnızca genel yer tutucuları kapsar. Gerçek
  marka adları dolandırıcılık hedefi olarak kullanılmaz.
- Çeviri hattı "makine çevirisi + model destekli inceleme + insan onayı"dır,
  salt insan doğrulaması değildir; raporda böyle anılmaktadır.

## Yeniden koşturma

```bash
pip install -r kod/requirements.txt
python kod/kontrol.py                    # kurulum denetimi

# güvenlik ve kontrol setleri (GPU'da ~5 dk, determinist)
python kod/veri_hatti.py --cevirmen opus --ornek 0
python kod/veri_hatti.py --kontrol-isle veri/gizli/kontrol_ornegi_claude_incelemesi.csv

# ölçümler
python kod/kos.py --model cosmos --seviye nf4 --test guvenlik
# --test: verimlilik | bilgi | guvenlik | onyargi
# --seviye: fp16 | int8 | nf4 | gptq4

# tablolar ve grafikler
jupyter nbconvert --execute kod/05_analiz.ipynb
```

Çeviri deterministiktir (opus-mt-tc-big-en-tr, `num_beams=4`, örnekleme yok), bu
yüzden istem seti birebir yeniden üretilir. `--kontrol-isle` adımı **isteğe
bağlı değildir**: incelenen 40 istemin 9'una yapılan insan düzeltmelerini geri
uygular, atlanırsa o 9 istem raporun koşularından farklı metinle sorulur.

GPTQ ağırlıkları `GPTQ4_<MODEL>` ortam değişkeniyle verilir ve
`GPTQ_BACKEND=gptq_exllama_v2` ile yüklenir. Kuantize ağırlıkların ölçülen
kalitesi yükleme yoluna bağlıdır: GPTQModel ile yazılan ağırlıklar
`transformers` ile sessizce yanlış yüklenip 3.281.024 perplexity
verebilmektedir. Raporun her satırının yükleme yolu ve çekirdeği kayıt
altındadır.

Adım adım Kaggle hücreleri: `KAGGLE_ADIMLARI.md`, `KAGGLE_TEK_SEVIYE.md`,
`KAGGLE_GUVENLIK_ADIMLARI.md`. Colab için `COLAB_ADIMLARI.md`, ağırlık üretimi
için `KUANTIZE_ADIMLARI.md`.
