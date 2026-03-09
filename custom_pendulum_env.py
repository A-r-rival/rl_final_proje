import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

class EncoderPendulumEnv(gym.Env):
    """
    Özel (Custom) Ters Sarkaç Ortamı - Sim2Real için optimize edilmiştir.
    
    Özellikler:
    - Kütle (m) ve Uzunluk (l) ayarlanabilir. Atalet = m * l^2
    - Enkoder Sensörü (Quantization) + Jitter Gürültüsü
    - İletişim Gecikmesi (Latency)
    - Tork Limiti
    """
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 30
    }

    def __init__(self, 
                 mass=0.2,            # kg
                 length=0.4,          # metre
                 tau_max=2.0,         # Nm (Motor tork limiti)
                 cpr=4096,            # Encoder Çözünürlüğü
                 encoder_noise_std=0.001, 
                 delay_ms=10,         # Gecikme (milisaniye)
                 dt=0.05,             # Simülasyon adım süresi (saniye)
                 render_mode=None):
        
        super().__init__()
        
        self.m = mass
        self.l = length
        self.tau_max = tau_max
        self.dt = dt
        self.g = 9.81
        self.cpr = cpr
        self.angle_step = 2 * np.pi / cpr
        self.encoder_noise_std = encoder_noise_std
        
        # Atalet formülü (Ortadan asılı çubuk I=ml^2/3 yerine uçta ağırlık I=ml^2)
        self.inertia = self.m * (self.l ** 2)
        
        # Observation Buffer for Latency
        self.delay_steps = max(1, int(delay_ms / (self.dt * 1000)))  # dt=0.05s -> 50ms (Step). Demek ki delay_ms 10 ise 0 adım olur, max(1, ...) ile an az 1 frame (50ms) geciktiriyoruz.
        # Eğer özel hızlı dt istenirse dt=0.01s yapılıp delay_steps artırılabilir. Biz şimdilik Gymnasium standardı 0.05s de kalıyoruz.
        self.obs_buffer = deque(maxlen=self.delay_steps)
        
        # Max hız limiti
        self.max_speed = 8.0 
        
        # Action: Sadece Tork
        self.action_space = spaces.Box(
            low=-self.tau_max, 
            high=self.tau_max, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Observation: [cos(theta_q), sin(theta_q), theta_dot_estimated]
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, 
            high=high, 
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True
        self.window_width = 800
        self.window_height = 500

        self.state = None # [true_theta, true_theta_dot]
        self.prev_theta_q = None
        self.step_count = 0
        self.max_episode_steps = 200 # Standard Gymnasium limit

    def _get_obs(self):
        # 1. Gerçek state'den açıyı al
        theta, true_theta_dot = self.state
        
        # 2. Sensör gürültüsü ve Quantization (Encoder Mantığı)
        theta_noisy = theta + self.np_random.normal(0, self.encoder_noise_std)
        theta_q = np.round(theta_noisy / self.angle_step) * self.angle_step
        
        # 3. Hız Kestirimi (Velocity Estimation)
        if self.prev_theta_q is None:
            self.prev_theta_q = theta_q
            
        delta_theta_q = angle_normalize(theta_q - self.prev_theta_q)
        theta_dot_est = delta_theta_q / self.dt
        
        # Hız limitine kırp
        theta_dot_est = np.clip(theta_dot_est, -self.max_speed, self.max_speed)
        
        self.prev_theta_q = theta_q
        
        # 4. Gözlem vektörü
        return np.array([np.cos(theta_q), np.sin(theta_q), theta_dot_est], dtype=np.float32), theta_q, theta_dot_est

    def step(self, action):
        th, thdot = self.state  # Gerçek (gizli) state
        self.step_count += 1
        
        # Tork limiti uygula (Clip)
        u = np.clip(action, -self.tau_max, self.tau_max)[0]
        
        # Fizik motoru güncellemesi (Gerçek değerler üzerinden)
        # alpha = Acceleration (Açısal İvme)
        # tau = I * alpha  --> alpha = tau / I
        # Gravity Torque = m * g * l * sin(th)
        # Toplam Tork = u - m * g * l * sin(th) (Uçtan sarktığında teta 0 varsayımıyla Gym'in standardı)
        # Ama Gym standardında Teta=0 yukarı bakar. Gravity: m*g*l*sin(th) (Aşağı çeker, bu yüzden sin(th) pozitifse tork negatiftir vs.)
        # Standart Gym förmülü:
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        
        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt # Gym standardı.
        # Biz bunu I = ml^2'ye göre uyarlıyoruz:
        # Tork_g = m * g * l * sin(th) (Lokal ağırlık merkezi uca alındı)
        newthdot = thdot + ((m * g * l * np.sin(th) + u) / self.inertia) * dt
        
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot], dtype=np.float32)
        
        # Gözlem okuması (Kuantize)
        obs, theta_q, theta_dot_est = self._get_obs()
        
        # Buffer'a ekle (Latency)
        self.obs_buffer.append(obs)
        delayed_obs = self.obs_buffer[0] # En eski gözlemi al
        
        # Ödül (Reward) fonksiyonunu Enkoder'ın GÖRDÜĞÜ (Quantize edilmiş) verilere göre hesaplıyoruz.
        # Bu SİM2REAL'de çok önemlidir çünkü gerçekte ajanın sadece enkoder bilgisi var.
        norm_theta_q = angle_normalize(theta_q)
        costs = norm_theta_q**2 + 0.1 * (theta_dot_est**2) + 0.001 * (u**2)
        reward = -costs

        truncated = self.step_count >= self.max_episode_steps
        terminated = False

        return delayed_obs, float(reward), terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        high = np.array([np.pi, 1]) # Başlangıç: rastgele açı, küçük hız
        self.state = self.np_random.uniform(low=-high, high=high)
        self.prev_theta_q = None
        self.step_count = 0
        
        obs, _, _ = self._get_obs()
        
        # Buffer'ı yeni başlangıç gözlemi ile doldur (sanki hep bu durumdaymışız gibi)
        for _ in range(self.delay_steps):
            self.obs_buffer.append(obs)
            
        delayed_obs = self.obs_buffer[0]
            
        return delayed_obs, {}

    def render(self):
        if self.render_mode is None:
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame`"
            )

        if self.screen is None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.window_width, self.window_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.window_width, self.window_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Çubuğu ve Merkezi Çiz
        # Gymnasium Pendulum-v1 tarzı render
        th, _ = self.state
        
        # Koordinat haritalaması (Ekranın ortasına monte et)
        offset_x = self.window_width // 2
        offset_y = self.window_height // 2
        # Çubuk uzunluğunu piksele map edelim. (maksimum yarı çapın (boyun) %40'ı)
        scale = (self.window_height // 2) * 0.8
        
        # Gerçek açıya göre (kırmızı çubuk)
        end_x = int(offset_x + scale * np.sin(th))
        end_y = int(offset_y - scale * np.cos(th))
        
        # Çubuğun kendisini çiz
        pygame.draw.line(self.screen, (200, 50, 50), (offset_x, offset_y), (end_x, end_y), 8)
        
        # Uca bir ağırlık (kütle) ekle (koyu gri bir daire)
        pygame.draw.circle(self.screen, (100, 100, 100), (end_x, end_y), 20)
        
        # Ortadaki merkez mil noktasını (mavi daire) en üste çiz (z-index gibi)
        pygame.draw.circle(self.screen, (50, 50, 200), (offset_x, offset_y), 10)
        
        # Enkoderin gördüğü uç noktayı da gösterelim (yeşil gölge, çok hafif sapmış olabilir)
        if self.prev_theta_q is not None:
             end_qx = int(offset_x + scale * np.sin(self.prev_theta_q))
             end_qy = int(offset_y - scale * np.cos(self.prev_theta_q))
             pygame.draw.line(self.screen, (50, 200, 50), (offset_x, offset_y), (end_qx, end_qy), 2)

        # Parametreleri (mass, length, tau, vs.) sol alta yazdır
        font = pygame.font.SysFont(None, 24)
        infos = [
            f"Mass: {self.m} kg",
            f"Length: {self.l} m",
            f"Max Torque: {self.tau_max} Nm",
            f"CPR: {self.cpr}",
            f"Noise Std: {self.encoder_noise_std}",
            f"Comm. Delay: {int(self.delay_steps * self.dt * 1000)} ms"
        ]
        
        # Sol alttan başlayıp her satırda yukarıya doğru diz (pencere yüksekliği baz alınarak)
        y_text = self.window_height - (len(infos) * 25) - 10 
        for info in infos:
            text_surface = font.render(info, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, y_text))
            y_text += 25

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
