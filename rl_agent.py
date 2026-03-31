"""
rl_agent.py  —  PPO Agent for Adaptive EKF Control
====================================================
For: LSTM + RL-Adaptive EKF Localization Project

The PPO agent learns to tune Q_scale and R_scale of the EKF
every 0.05s based on the current localization context.

Observation (8-dim, from carla_rl_environment.py):
    [innovation_x, innovation_y, position_uncertainty,
     time_since_gps, q_scale, r_scale, gps_denied, vehicle_speed]

Action (2-dim):
    [delta_Q_scale, delta_R_scale]  in [-0.5, 0.5]
    Applied as: Q_scale += delta_Q  (clamped to [0.1, 3.0])

API used by train_carla.py and evaluate_carla.py:
    agent = PPOAgent(obs_dim=8, action_dim=2)
    action, value, log_prob = agent.select_action(obs)
    agent.store_transition(obs, action, reward, value, log_prob, done)
    agent.update(next_obs=obs)   → stats dict
    agent.save(path)
    agent.load(path)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


# =============================================================================
# POLICY NETWORK  (Actor-Critic)
# =============================================================================

class PolicyNetwork(nn.Module):
    """
    Shared-trunk Actor-Critic for PPO.

    Actor  → outputs action mean in [-0.5, 0.5] (tanh + scale)
    Critic → outputs scalar value estimate
    """

    def __init__(self, obs_dim: int = 8, action_dim: int = 2):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Actor head
        self.actor_mean = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh(),   # output in [-1, 1], scaled to [-0.5, 0.5] in forward
        )
        self.actor_log_std = nn.Parameter(
            torch.full((action_dim,), -1.0))   # exp(-1) ≈ 0.37 initial std

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for actor output layer (more conservative actions)
        nn.init.orthogonal_(self.actor_mean[-2].weight, gain=0.01)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, obs_dim)
        Returns: action_mean (batch, action_dim),
                 log_std    (action_dim,),
                 value      (batch, 1)
        """
        feat        = self.shared(x)
        action_mean = self.actor_mean(feat) * 0.5   # scale [-1,1] → [-0.5,0.5]
        value       = self.critic(feat)
        return action_mean, self.actor_log_std, value


# =============================================================================
# PPO AGENT
# =============================================================================

class PPOAgent:
    """
    Proximal Policy Optimization agent.

    Hyperparameters tuned for EKF noise adaptation:
      - Small action range [-0.5, 0.5] → conservative Q/R changes per step
      - gamma=0.99 → values long-term localization accuracy
      - clip_eps=0.2 → standard PPO clip
    """

    GAMMA       = 0.99
    GAE_LAMBDA  = 0.95
    CLIP_EPS    = 0.2
    VALUE_COEF  = 0.5
    ENTROPY_COEF = 0.01
    PPO_EPOCHS  = 4
    LR          = 3e-4
    MAX_GRAD    = 0.5

    def __init__(self, obs_dim: int = 8, action_dim: int = 2,
                 lr: float = LR):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.policy     = PolicyNetwork(obs_dim, action_dim)
        self.optimizer  = optim.Adam(self.policy.parameters(), lr=lr)

        # Internal rollout buffer — cleared after each update()
        self._buf = {
            'obs':      [],
            'actions':  [],
            'rewards':  [],
            'values':   [],
            'log_probs':[],
            'dones':    [],
        }

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray,
                      deterministic: bool = False):
        """
        Select action from policy.

        Parameters
        ----------
        obs           : (obs_dim,) numpy array
        deterministic : if True, return mean action (no sampling)

        Returns
        -------
        action   : (action_dim,) numpy array  in [-0.5, 0.5]
        value    : float
        log_prob : float (None if deterministic)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            mean, log_std, value = self.policy(obs_t)
            value_f = float(value.item())

            if deterministic:
                action_t  = mean
                log_prob  = None
            else:
                std  = torch.exp(log_std).clamp(1e-4, 2.0)
                dist = torch.distributions.Normal(mean, std)
                action_t = dist.sample()
                action_t = torch.clamp(action_t, -0.5, 0.5)
                log_prob = float(dist.log_prob(action_t).sum(-1).item())

        action = action_t.cpu().numpy()[0]
        return action, value_f, log_prob

    # ── Buffer management ─────────────────────────────────────────────────────

    def store_transition(self, obs, action, reward, value, log_prob, done):
        """Store one (s, a, r, v, logp, done) tuple in the rollout buffer."""
        self._buf['obs'].append(obs)
        self._buf['actions'].append(action)
        self._buf['rewards'].append(reward)
        self._buf['values'].append(value)
        self._buf['log_probs'].append(log_prob if log_prob is not None else 0.0)
        self._buf['dones'].append(done)

    def _clear_buffer(self):
        for k in self._buf:
            self._buf[k] = []

    # ── GAE ───────────────────────────────────────────────────────────────────

    def _compute_gae(self, next_value: float):
        """Generalised Advantage Estimation."""
        rewards  = np.array(self._buf['rewards'],  dtype=np.float32)
        values   = np.array(self._buf['values'],   dtype=np.float32)
        dones    = np.array(self._buf['dones'],     dtype=np.float32)
        n        = len(rewards)
        adv      = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_v   = next_value if t == n - 1 else values[t + 1] * (1.0 - dones[t])
            delta    = rewards[t] + self.GAMMA * next_v - values[t]
            adv[t]   = last_gae = (delta
                                   + self.GAMMA * self.GAE_LAMBDA
                                   * (1.0 - dones[t]) * last_gae)

        returns = adv + values
        return adv, returns

    # ── PPO Update ────────────────────────────────────────────────────────────

    def update(self, next_obs: np.ndarray) -> dict:
        """
        Run PPO update on the stored rollout buffer.

        Parameters
        ----------
        next_obs : last observation for value bootstrapping

        Returns
        -------
        dict with policy_loss, value_loss, entropy keys
        """
        if len(self._buf['obs']) == 0:
            return {}

        # Bootstrap value for last step
        with torch.no_grad():
            next_t          = torch.FloatTensor(next_obs).unsqueeze(0)
            _, _, next_val  = self.policy(next_t)
            next_value      = float(next_val.item())

        adv, returns = self._compute_gae(next_value)

        # Normalise advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Convert buffer to tensors
        obs_t       = torch.FloatTensor(np.array(self._buf['obs']))
        actions_t   = torch.FloatTensor(np.array(self._buf['actions']))
        old_lp_t    = torch.FloatTensor(np.array(self._buf['log_probs']))
        adv_t       = torch.FloatTensor(adv)
        returns_t   = torch.FloatTensor(returns)

        p_losses, v_losses, entropies = [], [], []

        for _ in range(self.PPO_EPOCHS):
            mean, log_std, values = self.policy(obs_t)
            std  = torch.exp(log_std).clamp(1e-4, 2.0)
            dist = torch.distributions.Normal(mean, std)

            new_lp = dist.log_prob(actions_t).sum(-1)
            ratio  = torch.exp(new_lp - old_lp_t)

            # Clipped surrogate objective
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1 - self.CLIP_EPS,
                                    1 + self.CLIP_EPS) * adv_t
            p_loss = -torch.min(s1, s2).mean()

            # Value loss
            v_loss = 0.5 * (returns_t - values.squeeze()).pow(2).mean()

            # Entropy bonus
            entropy = dist.entropy().mean()

            loss = (p_loss
                    + self.VALUE_COEF  * v_loss
                    - self.ENTROPY_COEF * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.MAX_GRAD)
            self.optimizer.step()

            p_losses.append(p_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(entropy.item())

        self._clear_buffer()

        return {
            'policy_loss': float(np.mean(p_losses)),
            'value_loss':  float(np.mean(v_losses)),
            'entropy':     float(np.mean(entropies)),
        }

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'policy_state': self.policy.state_dict(),
            'optim_state':  self.optimizer.state_dict(),
            'obs_dim':      self.obs_dim,
            'action_dim':   self.action_dim,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(ckpt['policy_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])


# =============================================================================
# QUICK SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    print("PPOAgent smoke test...")
    agent = PPOAgent(obs_dim=8, action_dim=2)
    n_params = sum(p.numel() for p in agent.policy.parameters())
    print(f"  Parameters: {n_params:,}")

    obs = np.random.randn(8).astype(np.float32)
    action, value, log_prob = agent.select_action(obs)
    print(f"  select_action: action={action}, value={value:.3f}, log_prob={log_prob:.3f}")
    assert action.shape == (2,), "Wrong action shape"
    assert all(-0.5 <= a <= 0.5 for a in action), "Action out of range"

    for _ in range(10):
        a, v, lp = agent.select_action(obs)
        agent.store_transition(obs, a, -1.0, v, lp, False)
    stats = agent.update(next_obs=obs)
    print(f"  update stats: {stats}")

    agent.save('/tmp/test_agent.pth')
    agent.load('/tmp/test_agent.pth')
    print("  save/load: OK")

    print("✓ All smoke tests passed")
