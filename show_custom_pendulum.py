from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_pendulum_env import EncoderPendulumEnv
import numpy as np

# Ayarlar (Eğitimdeki ortamla aynı olmalı ki aynı fizikleri gözlemleyelim)
MODEL_NAME = "ppo_custom_pendulum"
CPR = 4096
MASS = 0.2
LENGTH = 0.4
TAU_MAX = 2.0
NOISE_STD = 0.001
DELAY_MS = 10
SEED = 42

print(f"[{MODEL_NAME}] modeli görselleştirilmek üzere yükleniyor...")
print(f"Fiziksel Parametreler -> Mass: {MASS}kg, Length: {LENGTH}m, Delay: {DELAY_MS}ms, CPR: {CPR}")

# Sadece render ("human") modu aktif olan, tekil bir fonksiyon oluşturuyoruz
def make_test_env():
    return EncoderPendulumEnv(
        mass=MASS, 
        length=LENGTH, 
        tau_max=TAU_MAX,
        cpr=CPR,
        encoder_noise_std=NOISE_STD,
        delay_ms=DELAY_MS,
        render_mode="human"
    )

test_env = DummyVecEnv([make_test_env])
test_env.seed(SEED)

# Modeli Disk'ten Yükle
try:
    model = PPO.load(MODEL_NAME, env=test_env, device="cpu") # Inference (test) CPU'da daha stabildir
except FileNotFoundError:
    print(f"\n[HATA] '{MODEL_NAME}.zip' dosyası bulunamadı. Lütfen önce `train_custom_pendulum.py` ile modeli eğitin.")
    exit(1)

# Simülasyon Döngüsü (Sonsuz test)
print("\nGörsel test başladı! Çıkış yapmak için terminalden Ctrl+C basabilir veya pencereyi kapatabilirsiniz.")
try:
    obs = test_env.reset()
    while True:
        # Deterministic=True, RL politikasının test sırasında en emin olduğu kararı almasını(mean) sağlar, noise eklemez.
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        
        # Eğer bölüm (episode) bittiyse ortamı sıfırla
        if done[0]:
            obs = test_env.reset()
except KeyboardInterrupt:
    print("\nTest sonlandırıldı.")
finally:
    test_env.close()
