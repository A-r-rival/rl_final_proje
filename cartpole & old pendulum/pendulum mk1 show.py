import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os

# Ayarlar
#MODEL_NAME = "ppo_pendulum_mk1"
MODEL_NAME = "ppo_pendulum_sde_surgeried"

test_env = gym.make("Pendulum-v1", render_mode="human")

os.path.exists(f"{MODEL_NAME}.zip")
print("Model yükleniyor...")
model = PPO.load(MODEL_NAME, env=test_env, device="cuda") # Varsa GPU kullan


# 2. TEST AŞAMASI (Görselleştirme AÇIK)
print("Sonuçlar görselleştiriliyor...")

obs, info = test_env.reset()

for _ in range(700):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    test_env.render()
    
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()