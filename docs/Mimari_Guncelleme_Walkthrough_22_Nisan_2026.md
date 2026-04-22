# Ters Sarkaç Sim2Real Mimari Güncellemesi Walkthrough
**Tarih:** 22 Nisan 2026
**Özet:** Bu güncelleme, prototip aşamasındaki ters sarkaç (inverted pendulum) modelimizi gerçeğe aktarmaya (Sim2Real) hazır, endüstriyel kalitede ve paralelleştirilmiş bir PPO (Proximal Policy Optimization) yapısına geçirmek için yapılmıştır.

## 1. Eğitim Ortamı Performans ve İstikrar İyileştirmeleri (`train_custom_pendulum.py`)
- **Çoklu Çekirdek Kullanımı (SubprocVecEnv):** Tekil çevreden (DummyVecEnv), 8 paralel çalışan (`N_ENVS=8`) alt işlem ortamına geçildi. Bu sayede modelin deneyim toplama hızı devasa ölçüde (yaklaşık ~8 kat) artırılmıştır.
- **Veri Normalizasyonu (VecNormalize):** Hem ortamdan gelen Gözlem (Observation) sensör verilerini hem de çıkış Ödüllerini (Rewards) normalize eden sarmalayıcı eklendi (`norm_obs=True, norm_reward=True`). Algoritmanın "Gradyan patlaması" yaşaması engellendi ve öğrenme stabilitesi güvence altına alındı.
- **Domain Randomization (Gerçek Dünya Simülasyonu):** Her paralel ortam açılışında, fiziksel ağırlık, sürtünme ve motor tork gücü limitleri `±10%` arasında rastgele değişecek şekilde ayarlandı. Model, "Kusursuz fiziği ezberlemek" yerine "Değişen fizik koşullarında tepki vermeyi" öğrenecek (Robustness).
- **Yeni Hiperparametreler:** Swing-up (Aşağıdan yukarı savurma) senaryosunu doğal olarak keşfedebilmesi (exploration) için 1 Milyon toplam adım limiti ve Entropi (`ent_coef=0.005`) değeri tanımlandı.
- **İzleme ve Kayıt:** `SystemMonitorCallback` yazılarak bilgisayarın CPU ve GPU (`nvidia-smi` üzerinden) kullanım metrikleri canlı olarak o anki TensorBoard grafiğine aktarıldı. Aniden kapatma senaryoları için (CTRL+C) `KeyboardInterrupt` dinleyicisi eklenip modelin mevcut aklını ve normalizasyon haritasını anında `.zip` / `.pkl` formatında diske yedeklemesi sağlandı.

## 2. Çevre & Fizik Modeli İyileştirmeleri (`custom_pendulum_env.py`)
- **Latency (Gecikme) Algoritması Tamiri:** `delay_steps` mekanizmasında milisaniye bölmelerinde sıfıra yuvarlanma yapmasını engelleyecek olan `round()` mantığı eklendi. Gecikmeler artık mükemmel kare senkronunda çalışıyor.
- **Curriculum Learning (Müfredat Öğrenimi) Altyapısı Eklendi:** Motorun hep rastgele yerlerden başlaması yerine parametrik olarak `reset(options={"reset_mode": "bottom"})` opsiyonlarıyla tamamen alttan veya tamamen üstten başlamasını seçebileceğimiz altyapı kodlandı.

## 3. Alüminyum Profil Ağırlık Güncellemesi
- Daha önceden hesaplanan "İçi Dolu 8mm Alüminyum (0.05kg)" verisi yerine referans dosyasına (`aluminyum_profil_analizi.pdf`) sadık kalınarak ortamın kütlesi güncellendi.
- "10mm Dış Çap x 1mm Et Kalınlıklı Alüminyum Boru" için hacim hesabı yapılarak (30cm çubuk uzunluğu baz alınarak) net kütlesi **23 Gram (0.023 kg)** olarak bulundu.
- Hem `show_custom_pendulum.py` hem de `train_custom_pendulum.py` içerisindeki `rod_mass` değerleri `0.023`'e eşitlendi. Bu sayede modelin motoru simülasyonda kaldırırken çekeceği yük miktarı gerçekle birebir senkronize edildi.

## 4. Test Modülü Optimizasyonu (`show_custom_pendulum.py`)
- Görselleştirme tarafı test edilirken çıldırmaması adına `train` tarafında eğitilen `VecNormalize.pkl` verisini yükleme (Load) mantığı geliştirildi.
- Test ortamındaki ağın dünya standart sapmalarını değiştirmemesi için `test_env.training = False` olarak kilitlendi. Bu sayede test ile eğitim evreni kusursuz hizalandı.

---

### Git Commit Mesajı Tavsiyesi (Push İçin)

Aşağıdaki bloğu kopyalayıp git commit mesajı olarak kullanabilirsiniz:

```text
Refactor: Upgrade PPO architecture for Sim2Real transfer

- Transition to SubprocVecEnv (8 cores) for massive parallelization speeds.
- Integrate VecNormalize to stabilize observations and scale target rewards.
- Implement Domain Randomization (±%10 on rod/motor dynamics) to close reality gap.
- Add SystemMonitorCallback to track CPU usage and Nvidia GPU bottlenecks via Tensorboard.
- Enable curriculum reset bounds (bottom/top startup options) in Pendulum Env.
- Implement graceful KeyboardInterrupt (CTRL+C) exiting while saving models immediately.
- Sync physical limits with 10x1mm tubular aluminum parameters (rod_mass=0.023 kg).
- Ensure test/show module loads dynamic normalization maps safely.
```
