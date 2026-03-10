import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
import torch

# Ayarlar
MODEL_NAME = "ppo_pendulum_sde_surgeried"


# 2. Modeli yüklerken SDE ayarlarını ez
# custom_objects ile modelin içindeki değişkenleri güncelliyoruz
custom_objects = {
    "use_sde": True,
    "sde_sample_freq": 4,
    "learning_rate": 0.0001 # Genellikle SDE'ye geçerken hızı biraz düşürmek iyidir
}

# 1. EĞİTİM AŞAMASI (Görselleştirme KAPALI)
# Render_mode None (varsayılan) olduğu için çok hızlı çalışır
train_env = gym.make("Pendulum-v1")
eval_env = gym.make("Pendulum-v1")

if os.path.exists(f"{MODEL_NAME}.zip"):
    print("Model yükleniyor...")
    model = PPO.load(MODEL_NAME, env=train_env, device="cuda", custom_objects=custom_objects) # Varsa GPU kullan
#else:
    #print("Yeni SDE destekli model oluşturuluyor...")
    # DİKKAT: Yeni oluştururken parametreler direkt girilir
    #model = PPO("MlpPolicy", train_env, verbose=1, device="cuda", 
                #use_sde=True, 
                #sde_sample_freq=4, 
                #learning_rate=0.0001,
                #tensorboard_log="./tensorboard/") # Canlı takip için eklendi

# 3. ÖNEMLİ: SDE gürültü matrislerini yeniden oluşturmak için 
# modelin içindeki gürültü sistemini sıfırlamalıyız
model.policy.reset_noise()

# 3. Değerlendirme Mekanizmasını Kur
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)

eval_callback = EvalCallback(eval_env, 
                             callback_on_new_best=callback_on_best, 
                             verbose=1, 
                             eval_freq=4000,
                             best_model_save_path="./logs/")

print("Eğitim başladı (Ekran açılmaz, çok daha hızlıdır)...")
model.learn(total_timesteps=60000, callback=eval_callback)
model.save(MODEL_NAME)
train_env.close()
eval_env.close()

# ---

# 2. TEST AŞAMASI (Görselleştirme AÇIK)
print("Eğitim bitti. Sonuçlar görselleştiriliyor...")
test_env = gym.make("Pendulum-v1", render_mode="human")
obs, info = test_env.reset()

for _ in range(700):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    test_env.render()
    
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()