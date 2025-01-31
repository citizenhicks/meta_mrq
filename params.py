
class Hyperparameters:
    def __init__(self):
        # Generic
        self.batch_size: int = 256
        self.buffer_size: int = int(1e6) 
        self.discount: float = 0.99
        self.target_update_freq: int = 2000

        # Exploration
        self.buffer_size_before_training: int = int(10e3) 
        self.exploration_noise: float = 0.1
        self.initial_random_exploration_steps: int = int(2000)

        # TD3
        self.target_policy_noise: float = 0.2
        self.noise_clip: float = 0.3

        # Encoder Loss
        self.dyn_weight: float = 1.0
        self.reward_weight: float = 0.1
        self.done_weight: float = 0.5

        # Replay Buffer (LAP - although LAP specific params not directly used in this basic implementation)
        self.prioritized: bool = True
        self.alpha: float = 0.4
        self.min_priority: float = 1.0
        self.enc_horizon: int = 3
        self.Q_horizon: int = 3

        # Encoder Model
        self.zs_dim: int = 512
        self.zsa_dim: int = 512
        self.za_dim: int = 256
        self.enc_hdim: int = 512
        self.enc_activ: str = 'elu' 
        self.enc_lr: float = 1e-4
        self.enc_wd: float = 1e-4
        self.pixel_augs: bool = False

        # Value Model
        self.value_hdim: int = 512
        self.value_activ: str = 'elu' # same as encoder_activ
        self.value_lr: float = 3e-4
        self.value_wd: float = 1e-4
        self.value_grad_clip: float = 20

        # Policy Model
        self.policy_hdim: int = 512
        self.policy_activ: str = 'relu' # same as encoder_activ
        self.policy_lr: float = 3e-4
        self.policy_wd: float = 1e-4
        self.gumbel_tau: float = 10
        self.pre_activ_weight: float = 1e-5

        # Reward model
        self.num_bins: int = 65
        self.lower: float = -10.0
        self.upper: float = 10.0
