# RL'de Train vs Eval Ortamları

Stable Baselines3 (`SB3`) kütüphanesinde `train_env` ve `eval_env` ayrımının nedenleri:

## 1. Bağımsızlık (Independence)
Eğitim ortamı (`train_env`), modelin sürekli aksiyon alıp hata yaparak öğrendiği alandır. Değerlendirme ortamı (`eval_env`) ise eğitimi belirli aralıklarla durdurur ve modeli "temiz" bir başlangıç noktasından test eder.

## 2. Özellik Farklılıkları
- **Train Env:** Verimlilik için gürültü (noise) veya ödül şekillendirme (reward shaping) içerebilir.
- **Eval Env:** Modelin "saf" halini test etmek için standart ve gürültüsüz tutulur.

## 3. En İyi Modeli Kaydetme
`EvalCallback`, `eval_env` üzerinde yaptığı testlerde en yüksek ödülü aldığı anı yakalar ve bu anki modeli `best_model.zip` olarak kaydeder. Bu, eğitim bittiğindeki son modelden daha başarılı olabilir.

## 4. Kaynak Yönetimi
Eğitim genellikle çok sayıda paralel ortamda yapılırken, değerlendirme tek ve kararlı bir ortamda yapılır.
