import gymnasium as gym
from stable_baselines3 import PPO
import os

# Ayarlar
MODEL_NAME = "ppo_cartpole_mk2"

# 1. EĞİTİM AŞAMASI (Görselleştirme KAPALI)
# Render_mode None (varsayılan) olduğu için çok hızlı çalışır
train_env = gym.make("CartPole-v1")

if os.path.exists(f"{MODEL_NAME}.zip"):
    print("Model yükleniyor...")
    model = PPO.load(MODEL_NAME, env=train_env, device="cuda") # Varsa GPU kullan
else:
    print("Yeni model oluşturuluyor...")
    model = PPO("MlpPolicy", train_env, verbose=1, device="cuda")

print("Eğitim başladı (Ekran açılmaz, çok daha hızlıdır)...")
model.learn(total_timesteps=50000)
model.save(MODEL_NAME)
train_env.close()

# ---

# 2. TEST AŞAMASI (Görselleştirme AÇIK)
print("Eğitim bitti. Sonuçlar görselleştiriliyor...")
test_env = gym.make("CartPole-v1", render_mode="human")
obs, info = test_env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    test_env.render()
    
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()