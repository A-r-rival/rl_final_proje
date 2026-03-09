import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from custom_pendulum_env import EncoderPendulumEnv

# Ayarlar
MODEL_NAME = "ppo_custom_pendulum"
CPR = 4096
MASS = 0.2
ROD_MASS = 0.05
LENGTH = 0.4
TAU_MAX = 2.0
NOISE_STD = 0.001
DELAY_MS = 10
SEED = 42

# Reproducibility için seed ayarla
set_random_seed(SEED)

# Ortamları Oluşturma Fonksiyonu
def make_env(render_mode=None):
    env = EncoderPendulumEnv(
        mass=MASS, 
        length=LENGTH,
        rod_mass=ROD_MASS, 
        tau_max=TAU_MAX,
        cpr=CPR,
        encoder_noise_std=NOISE_STD,
        delay_ms=DELAY_MS,
        render_mode=render_mode
    )
    env = Monitor(env) # Tensorboard ve Terminal çıktıları için ortamı Monitor'e sarmalıyoruz
    return env

# 1. EĞİTİM AŞAMASI
train_env = DummyVecEnv([lambda: make_env(render_mode=None)])
train_env.seed(SEED)
eval_env = DummyVecEnv([lambda: make_env(render_mode=None)])
eval_env.seed(SEED)

if os.path.exists(f"{MODEL_NAME}.zip"):
    print(f"{MODEL_NAME} model yükleniyor...")
    model = PPO.load(MODEL_NAME, env=train_env, device="cuda")
else:
    print("Yeni Özel Simülasyon modeli oluşturuluyor...")
    # ChatGPT tavsiyeli gürültülü ortam için sağlam hiperparametreler
    model = PPO("MlpPolicy", train_env, verbose=1, device="cuda",
                n_steps=4096, 
                batch_size=256, 
                learning_rate=3e-4,
                seed=SEED)

# 3. Değerlendirme Mekanizması
# NOT: Bizim Custom Env negatif "cost" döndürüyor, standart pendulum'daki -200 e benzer iyi bir skor yakalayabiliriz.
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-100, verbose=1)

eval_callback = EvalCallback(eval_env, 
                             callback_on_new_best=callback_on_best, 
                             verbose=1, 
                             eval_freq=4000,
                             best_model_save_path="./logs/")

# Checkpoint Mekanizması
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/checkpoints/',
                                         name_prefix='ppo_custom')

print(f"Eğitim başladı (Ekran açılmaz)... CPR: {CPR}, m: {MASS}kg, l: {LENGTH}m")
model.learn(total_timesteps=30000, callback=[eval_callback, checkpoint_callback], progress_bar=True)
model.save(MODEL_NAME)
train_env.close()
eval_env.close()

# ---

# 2. TEST AŞAMASI
print("Eğitim bitti. Sonuçlar görselleştiriliyor...")
test_env = DummyVecEnv([lambda: make_env(render_mode="human")])
obs = test_env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    test_env.render()
    
    # DummyVecEnv done state'ini liste olarak döndürür
    if done[0]:
        obs = test_env.reset()

test_env.close()
