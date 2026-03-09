import torch
from stable_baselines3 import PPO
import gymnasium as gym

# 1. Eski (SDE'siz) modeli yükle
old_model = PPO.load("ppo_pendulum_mk1")
old_params = old_model.policy.state_dict()

# 2. Yeni (SDE'li) iskeleti oluştur
env = gym.make("Pendulum-v1")
new_model = PPO("MlpPolicy", env, use_sde=True, sde_sample_freq=4, device="cuda")
new_params = new_model.policy.state_dict()

# 3. AMELİYAT BAŞLIYOR: Ağırlıkları transfer et
updated_params = {}
for name, param in old_params.items():
    if name in new_params:
        if param.shape == new_params[name].shape:
            # Boyutlar tutuyorsa (Policy ve Value ağları gibi) kopyala
            updated_params[name] = param
        else:
            # Boyut uyuşmazlığı olan (log_std gibi) kopyalanmaz, 
            # yeni modelin kendi rastgele/sıfır değerleri kalır
            print(f"[!] Boyut uyuşmazlığı atlanıyor: {name} | Eski: {param.shape}, Yeni: {new_params[name].shape}")
            updated_params[name] = new_params[name]

# 4. Güncellenmiş parametreleri yeni modele enjekte et
new_model.policy.load_state_dict(updated_params)
new_model.policy.reset_noise()

print("\n[+] Ameliyat başarılı! Bilgi transfer edildi, gürültü matrisleri SDE'ye göre ayarlandı.")

# 5. Artık eğitime devam edebilirsin
new_model.learn(total_timesteps=50000)
new_model.save("ppo_pendulum_sde_surgeried")

# add değerlendirme, görselleştirme vs. from mk1