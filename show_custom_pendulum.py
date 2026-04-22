import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from custom_pendulum_env import EncoderPendulumEnv
import numpy as np

# Ayarlar 
MODEL_NAME = "ppo_custom_pendulum_namiki120"
VECNORM_PATH = f"{MODEL_NAME}_vecnorm.pkl"

ENV_KWARGS = dict(
    mass=0.2, length=0.3, rod_mass=0.023,
    tau_max=1.41, cpr=640,
    encoder_noise_std=0.001, delay_ms=10,
)
SEED = 42

print(f"[{MODEL_NAME}] modeli görselleştirilmek üzere yükleniyor...")

# Sadece render ("human") modu aktif olan, tekil bir fonksiyon oluşturuyoruz
def make_test_env():
    return EncoderPendulumEnv(**ENV_KWARGS, render_mode="human")

# Test ortamını ayağa kaldır (DummyVecEnv görselleştirme için yeterlidir)
test_env_raw = DummyVecEnv([make_test_env])
test_env_raw.seed(SEED)

# Normalizasyon haritasını ve Modeli Disk'ten Yükle
if not os.path.exists(f"{MODEL_NAME}.zip"):
    print(f"\n[HATA] '{MODEL_NAME}.zip' dosyası bulunamadı. Lütfen önce `train_custom_pendulum.py` ile modeli eğitin.")
    exit(1)

if not os.path.exists(VECNORM_PATH):
    print(f"\n[HATA] '{VECNORM_PATH}' dosyası bulunamadı. Normalizasyon verisi olmadan model ortamı anlayamaz. Lütfen eğitimi baştan çalıştırın.")
    exit(1)

# Normalizasyonu yükle ve ayarlarını sabitle
# Teste özel: Ağın istatistik güncellemesini kapatır ve sadece gözlem(observation) normalizesi yapar.
test_env = VecNormalize.load(VECNORM_PATH, test_env_raw)
test_env.training = False
test_env.norm_reward = False

model = PPO.load(MODEL_NAME, env=test_env, device="cpu", custom_objects={"action_space": test_env.action_space})

# Simülasyon Döngüsü (Sonsuz test)
print("\nGörsel test başladı! Çıkış yapmak için terminalden Ctrl+C basabilir veya pencereyi kapatabilirsiniz.")
try:
    obs = test_env.reset()
    while True:
        # Deterministic=True, RL politikasının test sırasında en emin olduğu kararı almasını sağlar
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        
        # Eğer bölüm (episode) bittiyse ortamı sıfırla
        if done[0]:
            obs = test_env.reset()
except KeyboardInterrupt:
    print("\n[!] Test kullanıcı tarafından sonlandırıldı.")
finally:
    test_env.close()
