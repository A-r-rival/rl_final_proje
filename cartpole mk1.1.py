import gymnasium as gym
from stable_baselines3 import PPO
import os

# Dosya adı
model_path = "ppo_cartpole_model.zip"
env = gym.make("CartPole-v1", render_mode="human")
# print(env.metadata["render_modes"]) --->Seçenekleri görmek için

# 1. Kontrol: Eğer dosya varsa onu yükle, yoksa yeni oluştur
if os.path.exists(model_path):
    print("Mevcut model bulundu, yükleniyor...")
    # 'env=env' kısmını eklemeyi unutma, modelin hangi ortamda çalışacağını bilmesi lazım
    model = PPO.load(model_path, env=env)
else:
    print("Mevcut model bulunamadı, yeni model oluşturuluyor...")
    model = PPO("MlpPolicy", env, verbose=1)

# 2. Eğitime devam et (Kaldığı yerden üzerine ekleyerek öğrenir)
print("Eğitim devam ediyor...")
model.learn(total_timesteps=20000)

# 3. Güncellenmiş modeli tekrar kaydet (Üzerine yazar)
model.save(model_path)
# model.save("ppo_cartpole_model")

# 4. Test et
print("Test ediliyor...")
obs, info = env.reset()
for _ in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()