"""
rl_agent.py  —  PPO Agent for Adaptive EKF Control  (v4)
=========================================================
For: LSTM + RL-Adaptive EKF Localization Project

Change from v3
--------------
  v4-FIX — obs_dim default updated from 8 to 10.

  carla_rl_environment.py v4 produces a 10-dim observation:
    [0] innovation_x          (unchanged)
    [1] innovation_y          (unchanged)
    [2] position_uncertainty  (unchanged)
    [3] time_since_gps        (unchanged)
    [4] q_scale               (unchanged)
    [5] r_scale               (unchanged)
    [6] gps_denied            (unchanged)
    [7] vehicle_speed         (unchanged)
    [8] lstm_disagreement     NEW — |lstm_bias| / 5.0
    [9] lstm_ready            NEW — 1.0 if LSTM buffer full

  The two new dimensions let PPO learn:
    - When the LSTM is making a large correction (dim 8 high):
      the IMU has significant bias → agent may want lower Q (trust LSTM more)
    - When the LSTM buffer is not yet full (dim 9 = 0):
      LSTM is not yet active → treat as pure IMU → inflate Q appropriately

  IMPORTANT: The v3 model (obs_dim=8) CANNOT be loaded after this change.
  You must retrain PPO from scratch with the new obs_dim=10.
  Use: python rl_train.py  (after training LSTM v4 first)

All PPO logic (GAE, clipping, entropy bonus) is unchanged.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


# =============================================================================
# POLICY NETWORK  (shared trunk, unchanged except input size flexibility)
# =============================================================================
class PolicyNetwork(nn.Module):
    """
    Shared-trunk Actor-Critic for PPO.
    obs_dim default updated to 10 to match v4 environment.
    Architecture unchanged.
    """

    def __init__(self, obs_dim: int = 10, action_dim: int = 2):  # v4: default 10
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),      nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, action_dim), nn.Tanh(),
        )
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -1.0))
        self.critic = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean[-2].weight, gain=0.01)

    def forward(self, x: torch.Tensor):
        feat        = self.shared(x)
        action_mean = self.actor_mean(feat) * 0.5   # scale to [-0.5, 0.5]
        value       = self.critic(feat)
        return action_mean, self.actor_log_std, value


# =============================================================================
# PPO AGENT
# =============================================================================
class PPOAgent:
    """
    Proximal Policy Optimization agent.

    v4 change: obs_dim default is 10 (was 8).
    All hyperparameters and PPO logic unchanged.
    """

    GAMMA        = 0.99
    GAE_LAMBDA   = 0.95
    CLIP_EPS     = 0.2
    VALUE_COEF   = 0.5
    ENTROPY_COEF = 0.01
    PPO_EPOCHS   = 4
    LR           = 3e-4
    MAX_GRAD     = 0.5

    def __init__(self, obs_dim: int = 10, action_dim: int = 2,  # v4: default 10
                 lr: float = LR):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.policy     = PolicyNetwork(obs_dim, action_dim)
        self.optimizer  = optim.Adam(self.policy.parameters(), lr=lr)
        self._buf = {'obs':[], 'actions':[], 'rewards':[], 'values':[], 'log_probs':[], 'dones':[]}

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            mean, log_std, value = self.policy(obs_t)
            value_f = float(value.item())
            if deterministic:
                action_t = mean; log_prob = None
            else:
                std  = torch.exp(log_std).clamp(1e-4, 2.0)
                dist = torch.distributions.Normal(mean, std)
                action_t = torch.clamp(dist.sample(), -0.5, 0.5)
                log_prob = float(dist.log_prob(action_t).sum(-1).item())
        return action_t.cpu().numpy()[0], value_f, log_prob

    def store_transition(self, obs, action, reward, value, log_prob, done):
        self._buf['obs'].append(obs);          self._buf['actions'].append(action)
        self._buf['rewards'].append(reward);   self._buf['values'].append(value)
        self._buf['log_probs'].append(log_prob if log_prob is not None else 0.0)
        self._buf['dones'].append(done)

    def _clear_buffer(self):
        for k in self._buf: self._buf[k] = []

    def _compute_gae(self, next_value: float):
        rewards = np.array(self._buf['rewards'], dtype=np.float32)
        values  = np.array(self._buf['values'],  dtype=np.float32)
        dones   = np.array(self._buf['dones'],   dtype=np.float32)
        n = len(rewards); adv = np.zeros(n, dtype=np.float32); last_gae = 0.0
        for t in reversed(range(n)):
            next_v  = next_value if t == n-1 else values[t+1] * (1.0 - dones[t])
            delta   = rewards[t] + self.GAMMA * next_v - values[t]
            adv[t]  = last_gae = delta + self.GAMMA * self.GAE_LAMBDA * (1.0 - dones[t]) * last_gae
        return adv, adv + values

    def update(self, next_obs: np.ndarray) -> dict:
        if len(self._buf['obs']) == 0: return {}
        with torch.no_grad():
            _, _, next_val = self.policy(torch.FloatTensor(next_obs).unsqueeze(0))
            next_value = float(next_val.item())
        adv, returns = self._compute_gae(next_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs_t     = torch.FloatTensor(np.array(self._buf['obs']))
        actions_t = torch.FloatTensor(np.array(self._buf['actions']))
        old_lp_t  = torch.FloatTensor(np.array(self._buf['log_probs']))
        adv_t     = torch.FloatTensor(adv)
        returns_t = torch.FloatTensor(returns)
        p_losses = []; v_losses = []; entropies = []
        for _ in range(self.PPO_EPOCHS):
            mean, log_std, values = self.policy(obs_t)
            std  = torch.exp(log_std).clamp(1e-4, 2.0)
            dist = torch.distributions.Normal(mean, std)
            new_lp = dist.log_prob(actions_t).sum(-1)
            ratio  = torch.exp(new_lp - old_lp_t)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1-self.CLIP_EPS, 1+self.CLIP_EPS) * adv_t
            p_loss   = -torch.min(s1, s2).mean()
            v_loss   = 0.5 * (returns_t - values.squeeze()).pow(2).mean()
            entropy  = dist.entropy().mean()
            loss     = p_loss + self.VALUE_COEF * v_loss - self.ENTROPY_COEF * entropy
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.MAX_GRAD)
            self.optimizer.step()
            p_losses.append(p_loss.item()); v_losses.append(v_loss.item()); entropies.append(entropy.item())
        self._clear_buffer()
        return {'policy_loss': float(np.mean(p_losses)),
                'value_loss':  float(np.mean(v_losses)),
                'entropy':     float(np.mean(entropies))}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({'policy_state': self.policy.state_dict(),
                    'optim_state':  self.optimizer.state_dict(),
                    'obs_dim':      self.obs_dim,
                    'action_dim':   self.action_dim}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        # Guard: refuse to load v3 model (obs_dim=8) into v4 agent (obs_dim=10)
        saved_obs_dim = ckpt.get('obs_dim', 8)
        if saved_obs_dim != self.obs_dim:
            raise ValueError(
                f"Model obs_dim={saved_obs_dim} does not match agent obs_dim={self.obs_dim}.\n"
                f"You are trying to load a v3 PPO model into a v4 agent.\n"
                f"Retrain PPO with: python rl_train.py")
        self.policy.load_state_dict(ckpt['policy_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])


# =============================================================================
# SMOKE TEST
# =============================================================================
if __name__ == '__main__':
    print("PPOAgent v4 smoke test  (obs_dim=10)...")
    agent = PPOAgent(obs_dim=10, action_dim=2)
    n_params = sum(p.numel() for p in agent.policy.parameters())
    print(f"  Parameters: {n_params:,}")
    obs = np.random.randn(10).astype(np.float32)  # 10-dim obs
    action, value, log_prob = agent.select_action(obs)
    print(f"  select_action: action={action.round(3)}, value={value:.3f}")
    assert action.shape == (2,) and all(-0.5 <= a <= 0.5 for a in action), "FAIL"
    for _ in range(10):
        a, v, lp = agent.select_action(obs)
        agent.store_transition(obs, a, -1.0, v, lp, False)
    stats = agent.update(next_obs=obs)
    print(f"  update stats: {stats}")
    agent.save('/tmp/test_agent_v4.pth')
    agent.load('/tmp/test_agent_v4.pth')
    print("  save/load: OK")
    # Verify it refuses to load a v3 model
    torch.save({'policy_state': agent.policy.state_dict(),
                'optim_state': agent.optimizer.state_dict(),
                'obs_dim': 8, 'action_dim': 2}, '/tmp/fake_v3.pth')
    try:
        agent.load('/tmp/fake_v3.pth')
        print("  FAIL: should have rejected v3 model")
    except ValueError as e:
        print(f"  v3 model rejection: OK  ({str(e)[:60]}...)")
    print("All v4 smoke tests passed")
