# SDE Model Yükleme Hatası (Size Mismatch) ve Çözümü

## Sorun
Sıfırdan SDE (State-Dependent Exploration) olmadan eğitilmiş bir modeli (örn. `ppo_pendulum_mk1_for_2sde.zip`), daha sonra SDE ayarlarını açarak (`custom_objects={"use_sde": True}`) `PPO.load()` ile yüklemeye çalıştığımızda içsel bir çökmeyle (size mismatch) karşılaşıyoruz.

Stable Baselines3 `PPO.load()` metodunda `custom_objects` üzerinden parametreleri zorla değiştirseniz bile gürültü matrisi olan **`log_std`** gibi SDE'ye özgü parametrelerin boyutları eski modelle (SDE'siz) uyuşmadığı için ağ ağırlıkları yüklenemiyor.

## Çözüm: Ağırlık Ameliyatı (Weight Surgery)
Bu sorunu aşmak için `surgery.py` isminde bir kurtarma betiği ("Weight Surgery") kullanıldı. `surgery.py` şu adımları izliyor:

1. Eski SDE'siz modeli belleğe alıyoruz.
2. SDE aktif (`use_sde=True`) olacak şekilde sıfırdan **yeni bir iskelet PPO modeli** oluşturuyoruz.
3. Eski modeldeki ağırlıkları, yeni modelin ağırlıklarıyla karşılaştırıyoruz.
4. **Sadece boyutları birebir uyuşan katmanları kopyalıyoruz** (Örn: Policy ve Value ağları). Boyutu uyuşmayan SDE gürültü parametreleri atlanıyor ve yeni modelin rastgele oluşturduğu sağlıklı değerler korunuyor.
5. Aşılı yeni parametreleri `new_model.policy.load_state_dict()` ile yerleştiriyoruz.
6. `new_model.policy.reset_noise()` çağırarak SDE'nin gürültü matrislerini yeni koşullara göre baştan örneklemesini ve stabilitesini sağlıyoruz.
7. Modeli `ppo_pendulum_sde_surgeried.zip` adıyla kaydediyoruz.

Artık eğitime veya teste devam edilecek betiklerde (`pendulum2sde.py`), uyuşmazlığa düşen eski model yerine sadece bu **ameliyatlı yeni model** (`ppo_pendulum_sde_surgeried`) yükleniyor ve sorunsuz bir şekilde eğitime devam edilebiliyor.
