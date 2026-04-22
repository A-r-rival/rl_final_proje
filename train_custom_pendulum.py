import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from custom_pendulum_env import EncoderPendulumEnv
import psutil
import subprocess

# ─── Ayarlar ───────────────────────────────────────────────────
MODEL_NAME   = "ppo_custom_pendulum_namiki120"
N_ENVS       = 8          # Paralel ortam sayısı
TOTAL_STEPS  = 1_000_000  # En az 1M (swing-up için 2M önerilir)
SEED         = 42

ENV_KWARGS = dict(
    mass=0.2, length=0.3, rod_mass=0.023,
    tau_max=1.41, cpr=640,
    encoder_noise_std=0.001, delay_ms=10,
)

# ─── Domain Randomization destekli ortam ──────────────────────
def make_env(rank: int, seed: int = 0, randomize: bool = True, render_mode=None):
    def _init():
        import numpy as np
        rng = np.random.default_rng(seed + rank)
        kwargs = ENV_KWARGS.copy()
        if randomize:
            kwargs["mass"]      = kwargs["mass"]      * float(rng.uniform(0.9, 1.1)) # ±%10
            kwargs["length"]    = kwargs["length"]    * float(rng.uniform(0.9, 1.1)) # ±%10
            kwargs["tau_max"]   = kwargs["tau_max"]   * float(rng.uniform(0.9, 1.1)) # ±%10
        env = EncoderPendulumEnv(**kwargs, render_mode=render_mode)
        env = Monitor(env)
        return env
    set_random_seed(seed + rank)
    return _init

# ─── Ortamları Oluştur ─────────────────────────────────────────
if __name__ == '__main__':
    train_env = SubprocVecEnv([make_env(i, SEED) for i in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env_raw = SubprocVecEnv([make_env(0, SEED + 100, randomize=False)])
    eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False, 
                            training=False, clip_obs=10.0)

    # ─── Özel CPU/GPU İzleme Callback'i ───────────────────────────
    class SystemMonitorCallback(BaseCallback):
        def __init__(self, log_freq: int = 1000, verbose: int = 0):
            super().__init__(verbose)
            self.log_freq = log_freq

        def _on_step(self) -> bool:
            # Her N adımda bir (GPU'yu çok yormamak için) ölçüm al
            if self.n_calls % self.log_freq == 0:
                # CPU ve RAM
                try:
                    cpu_usage = psutil.cpu_percent()
                    self.logger.record("system/cpu_usage_percent", cpu_usage)
                except Exception:
                    pass
                
                # GPU (nvidia-smi kullanarak)
                try:
                    smi_output = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        encoding="utf-8"
                    ).strip().split('\n')[0]
                    self.logger.record("system/gpu_usage_percent", float(smi_output))
                except Exception:
                    pass
            return True

    # ─── Model ────────────────────────────────────────────────────
    VECNORM_PATH = f"{MODEL_NAME}_vecnorm.pkl"

    if os.path.exists(f"{MODEL_NAME}.zip"):
        print(f"Model yükleniyor: {MODEL_NAME}")
        train_env = VecNormalize.load(VECNORM_PATH, train_env)
        model = PPO.load(MODEL_NAME, env=train_env, device="cuda",
                         custom_objects={"action_space": train_env.action_space})
    else:
        print("Yeni model oluşturuluyor...")
        model = PPO(
            "MlpPolicy", train_env,
            verbose=1, device="cuda",
            seed=SEED,
            # ── Rollout ──
            n_steps=2048,        # Her env başına adım (toplam: 2048*8=16384)
            batch_size=512,
            n_epochs=20,
            # ── Öğrenme ──
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            # ── Clipping & Entropy ──
            clip_range=0.2,
            ent_coef=0.005,      # Swing-up için exploration kritik
            vf_coef=0.5,
            max_grad_norm=0.5,
            # ── TensorBoard ──
            tensorboard_log="./tb_logs/",
        )

    # ─── Callback'ler ─────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/",
        eval_freq=max(10_000 // N_ENVS, 1),  # ~10k adımda bir değerlendir
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./logs/checkpoints/",
        name_prefix="ppo_custom",
        save_vecnormalize=True,   # VecNormalize istatistiklerini de kaydet!
    )

    # ─── Eğitim ───────────────────────────────────────────────────
    sys_monitor_callback = SystemMonitorCallback(log_freq=max(5000 // N_ENVS, 1))

    print(f"Eğitim başlıyor | {N_ENVS} paralel env | {TOTAL_STEPS:,} adım")
    print("Durdurmak isterseniz terminalde CTRL+C tuşlarına basabilirsiniz. Mevcut ilerleme güvenle kaydedilecektir.")
    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=[eval_callback, checkpoint_callback, sys_monitor_callback],
            progress_bar=True,
            reset_num_timesteps=False,  # Resume için False
        )
    except KeyboardInterrupt:
        print("\n[!] Kullanıcı tarafından eğitim elle durduruldu (CTRL+C). Eğitimin son durumu kaydediliyor, lütfen bekleyin...")

    model.save(MODEL_NAME)
    train_env.save(VECNORM_PATH)   # Normalizasyon istatistiklerini kaydet
    train_env.close()
    eval_env.close()

    # ─── Test ─────────────────────────────────────────────────────
    print("\nEğitim tamamlandı. Test başlıyor...")
    test_env_raw = SubprocVecEnv([make_env(0, SEED, randomize=False, render_mode="human")])
    test_env = VecNormalize.load(VECNORM_PATH, test_env_raw)
    test_env.training = False   # Test sırasında istatistik güncelleme

    obs = test_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        if done[0]:
            obs = test_env.reset()

    test_env.close()
