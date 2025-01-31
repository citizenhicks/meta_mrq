import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

from params import Hyperparameters
from models import StateEncoder, StateActionEncoder, Value, PolicyNetwork
from utils import TwoHot, masked_mse, multi_step_reward, realign
from replaybuffer import PrioritizedReplayBuffer

hp = Hyperparameters()
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class MRQAgent:
    def __init__(self, env, discrete, history=1):
        self.env = env
        self.action_dim = env.action_space.n
        self.discrete = discrete
        self.history = history
        self.obs_shape = self.env.observation_space.shape


        # Exploration noise scaling for discrete actions
        self.exploration_noise = hp.exploration_noise
        self.noise_clip = hp.noise_clip
        self.target_policy_noise = hp.target_policy_noise
        if self.discrete:
            self.exploration_noise *= 0.5
            self.noise_clip *= 0.5
            self.target_policy_noise *= 0.5

        self.output_dim = hp.num_bins + hp.zs_dim + 1

        # Initialize Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
             obs_shape= self.obs_shape, action_dim=self.action_dim, max_action=1, horizon=hp.Q_horizon, device=DEVICE
        )
        self.state_shape = self.replay_buffer.state_shape
        
        # Initialize Encoder, Policy, Value Networks and move them to DEVICE
        self.f = StateEncoder(self.obs_shape[0] * self.history, hp.zs_dim, hp.enc_hdim).to(DEVICE)
        self.g = StateActionEncoder(
            self.action_dim, hp.za_dim, hp.zs_dim, hp.zsa_dim, hp.enc_hdim, self.output_dim
        ).to(DEVICE)
        self.q = Value(hp.zsa_dim, hp.value_hdim).to(DEVICE)
        self.pi = PolicyNetwork(hp.zs_dim, hp.policy_hdim, self.action_dim, self.discrete).to(DEVICE)

        # Initialize Target Networks on DEVICE
        self.f_target = copy.deepcopy(self.f).to(DEVICE)
        self.g_target = copy.deepcopy(self.g).to(DEVICE)
        self.q_target = copy.deepcopy(self.q).to(DEVICE)
        self.pi_target = copy.deepcopy(self.pi).to(DEVICE)

        # Optimizers (AdamW)
        self.encoder_optimizer = optim.AdamW(
            list(self.f.parameters()) + list(self.g.parameters()),
            lr=hp.enc_lr, weight_decay=hp.enc_wd
        )
        self.value_optimizer = optim.AdamW(
            self.q.parameters(),
            lr=hp.value_lr, weight_decay=hp.value_wd
        )
        self.policy_optimizer = optim.AdamW(
            self.pi.parameters(),
            lr=hp.policy_lr, weight_decay=hp.policy_wd
        )

        # Reward binning setup - TwoHot class
        self.two_hot = TwoHot(lower=hp.lower, upper=hp.upper, num_bins=hp.num_bins, device=DEVICE)

        self.gammas = torch.zeros((1, hp.Q_horizon, 1), device=DEVICE)
        discount = 1.0
        for t in range(hp.Q_horizon):
            self.gammas[:, t] = discount
            discount *= hp.discount
        self.discount_factor_n_step = discount  # Store n-step discount factor

        # Tracked values
        self.reward_scale = 1.0
        self.target_reward_scale = 0.0
        self.training_steps = 0
        self.episodes = 0

        # Initial target network sync
        self.sync_target_networks()

    def sync_target_networks(self):
        self.f_target.load_state_dict(self.f.state_dict())
        self.g_target.load_state_dict(self.g.state_dict())
        self.q_target.load_state_dict(self.q.state_dict())
        self.pi_target.load_state_dict(self.pi.state_dict())

    def select_action(self, state, explore=True):
        if self.replay_buffer.size < hp.buffer_size_before_training and explore:
            return self.env.action_space.sample()

        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            zs = self.f(state_t)
            action, _ = self.pi(zs)

        if explore:
            if self.discrete:
                action = action + torch.randn_like(action) * self.exploration_noise
                return int(action.argmax(dim=1))
            else:
                action = action + torch.randn_like(action) * self.exploration_noise
                action = torch.clamp(action, -1, 1)
                return action.squeeze(0).cpu().numpy()
        else:
            if self.discrete:
                return int(action.argmax(dim=1))
            else:
                return action.squeeze(0).cpu().numpy()

    def train(self):
        if self.replay_buffer.size <= hp.buffer_size_before_training: return

        self.training_steps += 1

        if (self.training_steps-1) % hp.target_update_freq == 0:
            self.pi_target.load_state_dict(self.pi.state_dict())
            self.q_target.load_state_dict(self.q.state_dict())
            self.f_target.load_state_dict(self.f.state_dict())
            self.target_reward_scale = self.reward_scale
            self.reward_scale = self.replay_buffer.reward_scale()

            for _ in range(hp.target_update_freq):
                state, action, next_state, reward, not_done = self.replay_buffer.sample()
                self.train_encoder(state, action, next_state, reward, not_done, self.replay_buffer.env_terminates)

        state, action, next_state, reward, not_done = self.replay_buffer.sample(horizon=hp.Q_horizon)
        reward = multi_step_reward(reward, self.gammas)

        Q, Q_target = self.train_rl(state, action, next_state, reward, not_done,
            self.reward_scale, self.target_reward_scale)

        if hp.prioritized:
            priority = (Q - Q_target.expand(-1,2)).abs().max(1).values
            priority = priority.clamp(min=hp.min_priority).pow(hp.alpha)
            self.replay_buffer.update_priority(priority)


    def train_encoder(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
        reward: torch.Tensor, not_done: torch.Tensor, env_terminates: bool):
        with torch.no_grad():
            encoder_target = self.f_target.forward(next_state.reshape(-1, *self.state_shape)).reshape(state.shape[0], -1, hp.zs_dim)

        pred_zs = self.f.forward(state[:,0])
        prev_not_done = 1
        encoder_loss = 0

        for i in range(hp.enc_horizon):
            zsa_out, zsa_logit = self.g.forward (pred_zs, action[:,i])

            pred_d, pred_zs, pred_r = zsa_out[:,0:1], zsa_out[:,1:hp.zs_dim+1], zsa_out[:,hp.zs_dim+1:] 

            dyn_loss = masked_mse(pred_zs, encoder_target[:,i], prev_not_done)
            reward_loss = (self.two_hot.cross_entropy_loss(pred_r, reward[:,i]) * prev_not_done).mean()
            done_loss = masked_mse(pred_d, 1. - not_done[:,i].reshape(-1,1), prev_not_done) if env_terminates else 0

            encoder_loss = encoder_loss + hp.dyn_weight * dyn_loss + hp.reward_weight * reward_loss + hp.done_weight * done_loss
            prev_not_done = not_done[:,i].reshape(-1,1) * prev_not_done 

        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_optimizer.step()


    def train_rl(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
        reward: torch.Tensor, not_done: torch.Tensor, reward_scale: float, target_reward_scale: float):
        with torch.no_grad():
            next_zs = self.f_target.forward(next_state)

            noise = (torch.randn_like(action) * hp.target_policy_noise).clamp(-hp.noise_clip, hp.noise_clip)
            policy_action, pre_activ = self.pi.forward(next_zs)
            next_action = realign(policy_action + noise, self.discrete)

            next_zsa, next_zsa_logits = self.g_target.forward(next_zs, next_action)            
            Q_target = self.q_target.forward(next_zsa_logits).min(1,keepdim=True).values
            Q_target = (reward + not_done * hp.discount * Q_target * target_reward_scale)/reward_scale

            zs = self.f.forward(state)
            zsa, zsa_logits = self.g.forward(zs, action)

        Q = self.q.forward(zsa_logits)
        value_loss = F.smooth_l1_loss(Q,Q_target.reshape(Q_target.shape[0], 1).repeat(1,2))

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), hp.value_grad_clip)
        self.value_optimizer.step()

        policy_action, pre_activ = self.pi.forward(zs)
        zsa, zsa_logits = self.g.forward(zs, policy_action)
        Q_policy = self.q.forward(zsa_logits)
        policy_loss = -Q_policy.mean() + hp.pre_activ_weight * pre_activ.pow(2).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        return Q, Q_target

    def run(self, max_episodes=1000, max_steps_per_episode=1000):
        episode_rewards = []
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            for steps in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.add(state, action,  next_state, reward, terminated, truncated)

                state = next_state
                episode_reward += reward
                self.train()

                if done:
                    break
            print(f"Episode: {episode+1}, Steps: {steps+1}, Reward: {episode_reward:.2f}")
            episode_rewards.append(episode_reward)
        return episode_rewards

if __name__ == '__main__':
    env = gym.make('CartPole-v1') 
    agent = MRQAgent(env, discrete=True)
    agent.run(max_episodes=2000) 
    env.close()