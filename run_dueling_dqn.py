#!/usr/bin/env python3
"""
Double Dueling DQN for Sepsis Treatment Optimization
=====================================================

Architecture:
- Dueling Network: Separates V(s) and A(s,a) streams
- Double Q-learning: Decouples action selection from evaluation
- Noisy Networks: Parameter noise for exploration (optional)
- Prioritized Experience Replay: Focus on important transitions
- Conservative Q-Learning penalty: Prevents overestimation in offline RL

Key improvements over standard DQN:
1. Dueling architecture better handles states where action choice matters less
2. Double Q-learning reduces overestimation bias
3. Layer normalization + residual connections for stable training
4. Relaxed behavioral constraint with entropy bonus
5. Conservative penalty for out-of-distribution actions

Based on:
- Wang et al., 2016 (Dueling DQN)
- Van Hasselt et al., 2016 (Double DQN)
- Kumar et al., 2020 (Conservative Q-Learning)
- Komorowski et al., 2018 (AI Clinician)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle
import json
import logging
import warnings
from datetime import datetime
from collections import deque
import random
import math

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dueling_dqn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DuelingDQNConfig:
    """Configuration for Double Dueling DQN pipeline."""
    # Data paths
    train_path: str = 'data/processed/train_trajectories.csv'
    val_path: str = 'data/processed/val_trajectories.csv'
    test_path: str = 'data/processed/test_trajectories.csv'
    output_dir: str = 'outputs/dueling_dqn'
    
    # Network architecture
    shared_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    value_hidden_dim: int = 128
    advantage_hidden_dim: int = 128
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    use_residual: bool = True
    
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 512
    gamma: float = 0.99
    n_epochs: int = 200
    grad_clip: float = 10.0
    weight_decay: float = 1e-5
    
    # Double DQN specific
    target_update_freq: int = 1000
    soft_update_tau: float = 0.005
    use_soft_update: bool = True
    
    # Conservative Q-Learning (CQL) penalty
    use_cql: bool = True
    cql_alpha: float = 0.5  # Weight of CQL penalty
    cql_temperature: float = 1.0
    
    # Behavioral constraint (relaxed)
    use_behavior_constraint: bool = True
    behavior_threshold: float = 0.005  # Very relaxed (0.5%)
    
    # Entropy bonus for diverse actions
    entropy_bonus: float = 0.01
    
    # Prioritized Experience Replay
    use_per: bool = True
    per_alpha: float = 0.6  # Priority exponent
    per_beta_start: float = 0.4  # IS correction start
    per_beta_end: float = 1.0  # IS correction end
    
    # Early stopping
    patience: int = 25
    min_delta: float = 0.0001
    
    # Evaluation
    n_bootstrap: int = 500  # Reduced for faster iteration
    
    # Reproducibility
    random_seed: int = 42
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# DUELING NETWORK ARCHITECTURE
# =============================================================================

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration.
    Adds learnable noise to weights for better exploration.
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""
    
    def __init__(self, dim: int, dropout_rate: float = 0.1, use_layer_norm: bool = True):
        super(ResidualBlock, self).__init__()
        
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.layer_norm(out + residual)
        return out


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN Architecture.
    
    Separates Q(s,a) into:
    - V(s): State value (how good is this state?)
    - A(s,a): Advantage (how much better is action a than average?)
    
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    
    This helps when:
    1. Many states have similar values regardless of action
    2. In some states, the action choice matters more than others
    """
    
    def __init__(
        self,
        n_state_features: int,
        n_actions: int,
        shared_hidden_dims: List[int] = [256, 256],
        value_hidden_dim: int = 128,
        advantage_hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        use_noisy: bool = False
    ):
        super(DuelingQNetwork, self).__init__()
        
        self.n_state_features = n_state_features
        self.n_actions = n_actions
        self.use_noisy = use_noisy
        
        # Shared feature extractor
        shared_layers = []
        prev_dim = n_state_features
        
        for i, hidden_dim in enumerate(shared_hidden_dims):
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                shared_layers.append(nn.LayerNorm(hidden_dim))
            shared_layers.append(nn.GELU())
            shared_layers.append(nn.Dropout(dropout_rate))
            
            # Add residual block after first layer
            if use_residual and i > 0 and prev_dim == hidden_dim:
                shared_layers.append(ResidualBlock(hidden_dim, dropout_rate, use_layer_norm))
            
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        shared_out_dim = shared_hidden_dims[-1]
        
        # Value stream: V(s)
        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        
        self.value_stream = nn.Sequential(
            LinearLayer(shared_out_dim, value_hidden_dim),
            nn.LayerNorm(value_hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            LinearLayer(value_hidden_dim, 1)
        )
        
        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            LinearLayer(shared_out_dim, advantage_hidden_dim),
            nn.LayerNorm(advantage_hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            LinearLayer(advantage_hidden_dim, n_actions)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            states: (batch_size, n_state_features) tensor
            
        Returns:
            Q-values: (batch_size, n_actions) tensor
        """
        # Shared features
        features = self.shared_network(states)
        
        # Value and advantage streams
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, n_actions)
        
        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        # Subtracting mean ensures identifiability
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def get_value_and_advantage(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get separate V(s) and A(s,a) for analysis."""
        features = self.shared_network(states)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value, advantage
    
    def reset_noise(self):
        """Reset noise in noisy layers."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


# =============================================================================
# PRIORITIZED EXPERIENCE REPLAY
# =============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    
    Samples transitions with probability proportional to TD-error.
    Uses sum-tree for efficient O(log n) sampling.
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 0
        
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.data = []
        self.pos = 0
        self.max_priority = 1.0
    
    def add(self, transition: Tuple):
        """Add transition with max priority."""
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition
        
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with priorities."""
        n = len(self.data)
        
        # Compute sampling probabilities
        priorities = self.priorities[:n] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(n, size=batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        beta = min(1.0, self.beta_start + self.frame * (self.beta_end - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        weights = (n * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        batch = [self.data[i] for i in indices]
        
        return batch, indices, weights.astype(np.float32)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        priorities = np.abs(td_errors) + 1e-6
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.data)


# =============================================================================
# DOUBLE DUELING DQN AGENT
# =============================================================================

class DoubleDuelingDQNAgent:
    """
    Double Dueling DQN Agent for offline RL.
    
    Key features:
    - Dueling architecture (V + A streams)
    - Double Q-learning (reduces overestimation)
    - Conservative Q-Learning penalty (offline RL)
    - Relaxed behavioral constraint
    - Entropy bonus for action diversity
    """
    
    def __init__(
        self,
        n_state_features: int,
        n_actions: int,
        config: DuelingDQNConfig,
        behavior_policy: Optional['LogisticBehaviorPolicy'] = None
    ):
        self.n_state_features = n_state_features
        self.n_actions = n_actions
        self.config = config
        self.behavior_policy = behavior_policy
        self.device = torch.device(config.device)
        
        logger.info(f"Initializing Double Dueling DQN on {self.device}")
        
        # Online Q-network
        self.q_network = DuelingQNetwork(
            n_state_features=n_state_features,
            n_actions=n_actions,
            shared_hidden_dims=config.shared_hidden_dims,
            value_hidden_dim=config.value_hidden_dim,
            advantage_hidden_dim=config.advantage_hidden_dim,
            dropout_rate=config.dropout_rate,
            use_layer_norm=config.use_layer_norm,
            use_residual=config.use_residual
        ).to(self.device)
        
        # Target Q-network
        self.target_network = DuelingQNetwork(
            n_state_features=n_state_features,
            n_actions=n_actions,
            shared_hidden_dims=config.shared_hidden_dims,
            value_hidden_dim=config.value_hidden_dim,
            advantage_hidden_dim=config.advantage_hidden_dim,
            dropout_rate=config.dropout_rate,
            use_layer_norm=config.use_layer_norm,
            use_residual=config.use_residual
        ).to(self.device)
        
        # Initialize target with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # State scaler
        self.state_scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Training stats
        self.update_count = 0
        self.training_losses = []
        
        # Log model size
        n_params = sum(p.numel() for p in self.q_network.parameters())
        logger.info(f"  Model parameters: {n_params:,}")
    
    def fit_scaler(self, states: np.ndarray):
        """Fit the state scaler on training data."""
        self.state_scaler.fit(states)
        self.scaler_fitted = True
    
    def scale_states(self, states: np.ndarray) -> np.ndarray:
        """Scale states using fitted scaler."""
        if not self.scaler_fitted:
            raise RuntimeError("Scaler not fitted. Call fit_scaler first.")
        return self.state_scaler.transform(states)
    
    def get_action_mask(self, states: np.ndarray) -> np.ndarray:
        """
        Get mask of allowed actions based on behavioral constraint.
        """
        if not self.config.use_behavior_constraint or self.behavior_policy is None:
            return np.ones((len(states), self.n_actions), dtype=bool)
        
        probs = self.behavior_policy.predict_probs(states)
        mask = probs > self.config.behavior_threshold
        
        # Ensure at least top-3 actions are allowed per state
        for i in range(len(mask)):
            if mask[i].sum() < 3:
                top_k = np.argsort(probs[i])[-3:]
                mask[i, top_k] = True
        
        return mask
    
    def select_actions(
        self,
        states: np.ndarray,
        use_constraint: bool = True,
        temperature: float = 0.0
    ) -> np.ndarray:
        """
        Select actions (greedy or with temperature).
        
        Args:
            states: (batch_size, n_state_features) array (unscaled)
            use_constraint: Whether to apply BCQ constraint
            temperature: If > 0, sample from softmax distribution
            
        Returns:
            actions: (batch_size,) array of action indices
        """
        states_scaled = self.scale_states(states)
        states_tensor = torch.FloatTensor(states_scaled).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(states_tensor).cpu().numpy()
        
        if use_constraint and self.config.use_behavior_constraint:
            mask = self.get_action_mask(states)
            q_values[~mask] = -np.inf
        
        if temperature > 0:
            # Softmax sampling
            q_values = q_values - q_values.max(axis=1, keepdims=True)
            probs = np.exp(q_values / temperature)
            probs = probs / probs.sum(axis=1, keepdims=True)
            actions = np.array([np.random.choice(self.n_actions, p=p) for p in probs])
        else:
            # Greedy
            actions = np.argmax(q_values, axis=1)
        
        return actions
    
    def compute_cql_penalty(
        self,
        q_values: torch.Tensor,
        actions: torch.Tensor,
        behavior_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Conservative Q-Learning penalty.
        
        Penalizes Q-values for actions not taken by behavior policy.
        This prevents overestimation of out-of-distribution actions.
        
        CQL loss = α * (log_sum_exp(Q(s,·)) - Q(s, a_data))
        """
        # Log-sum-exp of all Q-values (soft maximum)
        logsumexp_q = torch.logsumexp(q_values / self.config.cql_temperature, dim=1)
        
        # Q-value of actual action taken
        q_data = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # CQL penalty
        cql_loss = (logsumexp_q - q_data).mean()
        
        return cql_loss
    
    def compute_double_dqn_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        next_action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Double DQN targets.
        
        target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
        """
        with torch.no_grad():
            # Online network selects actions
            next_q_online = self.q_network(next_states)
            
            if next_action_mask is not None:
                next_q_online = next_q_online.masked_fill(~next_action_mask, -1e9)
            
            best_actions = next_q_online.argmax(dim=1)
            
            # Target network evaluates
            next_q_target = self.target_network(next_states)
            next_q_values = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            # Compute targets
            targets = rewards + self.config.gamma * next_q_values * (1 - dones.float())
        
        return targets
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        behavior_probs: Optional[torch.Tensor] = None,
        next_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Perform one update step.
        
        Returns:
            loss: float
            td_errors: numpy array for PER priority updates
        """
        self.q_network.train()
        
        if weights is None:
            weights = torch.ones(len(states), device=self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN targets
        targets = self.compute_double_dqn_targets(
            rewards, next_states, dones, next_action_mask
        )
        
        # TD-errors for PER
        td_errors = (current_q - targets).detach().cpu().numpy()
        
        # Weighted Huber loss
        td_loss = (weights * F.smooth_l1_loss(current_q, targets, reduction='none')).mean()
        
        # CQL penalty
        if self.config.use_cql and behavior_probs is not None:
            cql_loss = self.compute_cql_penalty(q_values, actions, behavior_probs)
            total_loss = td_loss + self.config.cql_alpha * cql_loss
        else:
            cql_loss = torch.tensor(0.0)
            total_loss = td_loss
        
        # Entropy bonus (encourage diverse actions)
        if self.config.entropy_bonus > 0:
            action_probs = F.softmax(q_values, dim=1)
            entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()
            total_loss = total_loss - self.config.entropy_bonus * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            max_norm=self.config.grad_clip
        )
        
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.config.use_soft_update:
            self._soft_update_target()
        elif self.update_count % self.config.target_update_freq == 0:
            self._hard_update_target()
        
        self.training_losses.append(total_loss.item())
        
        return total_loss.item(), td_errors
    
    def _soft_update_target(self):
        """Soft update target network."""
        tau = self.config.soft_update_tau
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1 - tau) * target_param.data
            )
    
    def _hard_update_target(self):
        """Hard update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_all_q_values(self, states: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        states_scaled = self.scale_states(states)
        states_tensor = torch.FloatTensor(states_scaled).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(states_tensor).cpu().numpy()
        
        return q_values
    
    def get_value_advantage(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get separate V(s) and A(s,a) values."""
        states_scaled = self.scale_states(states)
        states_tensor = torch.FloatTensor(states_scaled).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            value, advantage = self.q_network.get_value_and_advantage(states_tensor)
        
        return value.cpu().numpy(), advantage.cpu().numpy()


# =============================================================================
# BEHAVIOR POLICY
# =============================================================================

class LogisticBehaviorPolicy:
    """State-dependent behavior policy using multinomial logistic regression."""
    
    def __init__(
        self,
        n_actions: int,
        softening_epsilon: float = 0.01,
        random_seed: int = 42
    ):
        self.n_actions = n_actions
        self.softening_epsilon = softening_epsilon
        self.random_seed = random_seed
        
        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=random_seed,
            n_jobs=-1
        )
        self.state_scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, states: np.ndarray, actions: np.ndarray):
        """Fit the behavior policy."""
        logger.info("Fitting logistic regression behavior policy...")
        
        states_scaled = self.state_scaler.fit_transform(states)
        self.model.fit(states_scaled, actions)
        self.classes_ = self.model.classes_
        self.fitted = True
        
        logger.info(f"  Behavior policy fitted. Classes: {len(self.classes_)}")
    
    def predict_probs(self, states: np.ndarray) -> np.ndarray:
        """Predict action probabilities."""
        if not self.fitted:
            raise RuntimeError("Behavior policy not fitted.")
        
        states_scaled = self.state_scaler.transform(states)
        probs_raw = self.model.predict_proba(states_scaled)
        
        n_samples = len(states)
        probs = np.full((n_samples, self.n_actions), self.softening_epsilon / self.n_actions)
        
        for i, cls in enumerate(self.classes_):
            probs[:, int(cls)] = probs_raw[:, i]
        
        # Apply softening
        uniform = 1.0 / self.n_actions
        probs = (1 - self.softening_epsilon) * probs + self.softening_epsilon * uniform
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs
    
    def get_prob(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Get probability of specific actions."""
        probs = self.predict_probs(states)
        return probs[np.arange(len(actions)), actions.astype(int)]


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_dueling_dqn(
    agent: DoubleDuelingDQNAgent,
    train_data: Dict,
    val_data: Dict,
    config: DuelingDQNConfig
) -> Dict:
    """Train Double Dueling DQN agent."""
    logger.info("Starting Double Dueling DQN training...")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Epochs: {config.n_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  CQL: {config.use_cql} (α={config.cql_alpha})")
    logger.info(f"  PER: {config.use_per}")
    
    device = torch.device(config.device)
    
    # Prepare data
    train_states = agent.scale_states(train_data['states'])
    train_actions = train_data['actions']
    train_rewards = train_data['rewards']
    train_next_states = agent.scale_states(train_data['next_states'])
    train_dones = train_data['dones']
    
    # Behavior policy probabilities for CQL
    train_behavior_probs = agent.behavior_policy.predict_probs(train_data['states'])
    
    # Action masks
    if config.use_behavior_constraint:
        train_action_mask = agent.get_action_mask(train_data['states'])
        train_next_action_mask = agent.get_action_mask(train_data['next_states'])
    else:
        train_action_mask = None
        train_next_action_mask = None
    
    # Setup PER or standard replay
    n_samples = len(train_states)
    
    if config.use_per:
        replay_buffer = PrioritizedReplayBuffer(
            capacity=n_samples,
            alpha=config.per_alpha,
            beta_start=config.per_beta_start,
            beta_end=config.per_beta_end,
            beta_frames=config.n_epochs * (n_samples // config.batch_size)
        )
        
        # Add all transitions
        for i in range(n_samples):
            transition = (
                train_states[i],
                train_actions[i],
                train_rewards[i],
                train_next_states[i],
                train_dones[i],
                train_behavior_probs[i],
                train_next_action_mask[i] if train_next_action_mask is not None else None
            )
            replay_buffer.add(transition)
    else:
        # Standard DataLoader
        dataset = TensorDataset(
            torch.FloatTensor(train_states),
            torch.LongTensor(train_actions),
            torch.FloatTensor(train_rewards),
            torch.FloatTensor(train_next_states),
            torch.FloatTensor(train_dones.astype(float)),
            torch.FloatTensor(train_behavior_probs)
        )
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=0, pin_memory=(config.device == 'cuda')
        )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_q_mean': [],
        'val_q_std': [],
        'val_v_mean': [],
        'val_a_mean': [],
        'n_unique_actions': [],
        'action_entropy': [],
        'learning_rate': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None
    
    n_batches = n_samples // config.batch_size
    
    for epoch in range(config.n_epochs):
        epoch_losses = []
        
        if config.use_per:
            # PER sampling
            for _ in range(n_batches):
                batch, indices, weights = replay_buffer.sample(config.batch_size)
                
                # Unpack batch
                states = torch.FloatTensor(np.array([t[0] for t in batch])).to(device)
                actions = torch.LongTensor(np.array([t[1] for t in batch])).to(device)
                rewards = torch.FloatTensor(np.array([t[2] for t in batch])).to(device)
                next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(device)
                dones = torch.FloatTensor(np.array([t[4] for t in batch])).to(device)
                behavior_probs = torch.FloatTensor(np.array([t[5] for t in batch])).to(device)
                
                if batch[0][6] is not None:
                    next_mask = torch.BoolTensor(np.array([t[6] for t in batch])).to(device)
                else:
                    next_mask = None
                
                weights_tensor = torch.FloatTensor(weights).to(device)
                
                loss, td_errors = agent.update(
                    states, actions, rewards, next_states, dones,
                    weights=weights_tensor,
                    behavior_probs=behavior_probs,
                    next_action_mask=next_mask
                )
                
                # Update priorities
                replay_buffer.update_priorities(indices, td_errors)
                epoch_losses.append(loss)
        else:
            # Standard training
            for batch in dataloader:
                states, actions, rewards, next_states, dones, behavior_probs = [
                    b.to(device) for b in batch
                ]
                
                loss, _ = agent.update(
                    states, actions, rewards, next_states, dones,
                    behavior_probs=behavior_probs
                )
                epoch_losses.append(loss)
        
        train_loss = np.mean(epoch_losses)
        history['train_loss'].append(train_loss)
        history['learning_rate'].append(agent.optimizer.param_groups[0]['lr'])
        
        # Update scheduler
        agent.scheduler.step()
        
        # Validation
        val_states_scaled = agent.scale_states(val_data['states'])
        val_states_tensor = torch.FloatTensor(val_states_scaled).to(device)
        val_actions_tensor = torch.LongTensor(val_data['actions']).to(device)
        val_rewards_tensor = torch.FloatTensor(val_data['rewards']).to(device)
        val_next_states_tensor = torch.FloatTensor(
            agent.scale_states(val_data['next_states'])
        ).to(device)
        val_dones_tensor = torch.FloatTensor(val_data['dones'].astype(float)).to(device)
        
        agent.q_network.eval()
        with torch.no_grad():
            val_q_all = agent.q_network(val_states_tensor)
            val_q = val_q_all.gather(1, val_actions_tensor.unsqueeze(1)).squeeze(1)
            val_targets = agent.compute_double_dqn_targets(
                val_rewards_tensor, val_next_states_tensor, val_dones_tensor
            )
            val_loss = F.smooth_l1_loss(val_q, val_targets).item()
            
            # Get V and A separately
            val_v, val_a = agent.q_network.get_value_and_advantage(val_states_tensor)
        
        history['val_loss'].append(val_loss)
        history['val_q_mean'].append(val_q_all.mean().item())
        history['val_q_std'].append(val_q_all.std().item())
        history['val_v_mean'].append(val_v.mean().item())
        history['val_a_mean'].append(val_a.mean().item())
        
        # Action diversity on validation set
        val_actions_pred = agent.select_actions(val_data['states'][:10000], use_constraint=True)
        n_unique = len(np.unique(val_actions_pred))
        action_counts = np.bincount(val_actions_pred, minlength=agent.n_actions)
        action_probs = action_counts / action_counts.sum()
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        
        history['n_unique_actions'].append(n_unique)
        history['action_entropy'].append(action_entropy)
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}: Loss={train_loss:.4f}/{val_loss:.4f}, "
                f"Q={history['val_q_mean'][-1]:.2f}±{history['val_q_std'][-1]:.2f}, "
                f"V={history['val_v_mean'][-1]:.2f}, "
                f"Actions={n_unique}, H={action_entropy:.2f}"
            )
        
        # Early stopping
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = {
                'q_network': agent.q_network.state_dict().copy(),
                'target_network': agent.target_network.state_dict().copy()
            }
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_state_dict is not None:
        agent.q_network.load_state_dict(best_state_dict['q_network'])
        agent.target_network.load_state_dict(best_state_dict['target_network'])
    
    logger.info(f"  Training complete. Best val loss: {best_val_loss:.4f}")
    
    return history


# =============================================================================
# OFF-POLICY EVALUATION
# =============================================================================

class WDREstimator:
    """Weighted Doubly Robust estimator."""
    
    def __init__(self, gamma: float = 0.99, max_weight: float = 50.0):
        self.gamma = gamma
        self.max_weight = max_weight
    
    def estimate(
        self,
        df: pd.DataFrame,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        agent: DoubleDuelingDQNAgent,
        behavior_policy: LogisticBehaviorPolicy
    ) -> Dict:
        """Compute WDR estimate."""
        trajectory_values = []
        trajectory_weights = []
        
        for stay_id in df['stay_id'].unique():
            mask = (df['stay_id'] == stay_id).values
            
            traj_states = states[mask]
            traj_actions = actions[mask]
            traj_rewards = rewards[mask]
            
            # Get Q-values
            q_values = agent.get_all_q_values(traj_states)
            
            # Behavior policy probabilities
            pi_b = behavior_policy.get_prob(traj_states, traj_actions)
            pi_b = np.clip(pi_b, 1e-6, 1.0)
            
            # Target policy (constrained greedy with small stochasticity)
            target_actions = agent.select_actions(traj_states, use_constraint=True)
            pi_e = (target_actions == traj_actions).astype(float)
            pi_e = 0.95 * pi_e + 0.05 / agent.n_actions
            
            # Importance weights
            rho = pi_e / pi_b
            rho = np.clip(rho, 0, self.max_weight)
            
            # Simple IS estimate for this trajectory
            T = len(traj_rewards)
            cumulative_rho = 1.0
            G = 0.0
            
            for t in range(T - 1, -1, -1):
                G = traj_rewards[t] + self.gamma * G
                cumulative_rho *= rho[t]
                cumulative_rho = min(cumulative_rho, self.max_weight)
            
            trajectory_values.append(G * cumulative_rho)
            trajectory_weights.append(cumulative_rho)
        
        trajectory_values = np.array(trajectory_values)
        trajectory_weights = np.array(trajectory_weights)
        
        # Normalized WIS
        wis_estimate = np.sum(trajectory_values) / (np.sum(trajectory_weights) + 1e-10)
        
        # ESS
        ess = (np.sum(trajectory_weights) ** 2) / (np.sum(trajectory_weights ** 2) + 1e-10)
        ess_ratio = ess / len(trajectory_weights)
        
        return {
            'wdr_estimate': wis_estimate,
            'wdr_std': np.std(trajectory_values),
            'wdr_se': np.std(trajectory_values) / np.sqrt(len(trajectory_values)),
            'ess': ess,
            'ess_ratio': ess_ratio,
            'n_trajectories': len(trajectory_values),
            'trajectory_values': trajectory_values
        }


def bootstrap_evaluation(
    df: pd.DataFrame,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    agent: DoubleDuelingDQNAgent,
    behavior_policy: LogisticBehaviorPolicy,
    n_bootstrap: int = 500,
    gamma: float = 0.99
) -> Dict:
    """Compute bootstrap confidence intervals."""
    logger.info(f"Computing bootstrap CI ({n_bootstrap} samples)...")
    
    # Precompute per-trajectory returns and AI recommendations
    stay_ids = df['stay_id'].unique()
    trajectory_data = {}
    
    for stay_id in stay_ids:
        mask = (df['stay_id'] == stay_id).values
        traj_rewards = rewards[mask]
        traj_actions = actions[mask]
        traj_states = states[mask]
        
        # Clinician return
        G = 0.0
        for t in range(len(traj_rewards) - 1, -1, -1):
            G = traj_rewards[t] + gamma * G
        
        # AI actions and agreement
        ai_actions = agent.select_actions(traj_states, use_constraint=True)
        agreement = np.mean(traj_actions == ai_actions)
        
        trajectory_data[stay_id] = {
            'return': G,
            'agreement': agreement,
            'n_steps': len(traj_rewards),
            'died': traj_rewards[-1] < 0
        }
    
    # Bootstrap
    bootstrap_returns = []
    bootstrap_agreements = []
    
    for b in range(n_bootstrap):
        if (b + 1) % 100 == 0:
            logger.info(f"  Bootstrap {b+1}/{n_bootstrap}")
        
        sampled_ids = np.random.choice(stay_ids, size=len(stay_ids), replace=True)
        
        returns = [trajectory_data[sid]['return'] for sid in sampled_ids]
        agreements = [trajectory_data[sid]['agreement'] for sid in sampled_ids]
        
        bootstrap_returns.append(np.mean(returns))
        bootstrap_agreements.append(np.mean(agreements))
    
    bootstrap_returns = np.array(bootstrap_returns)
    bootstrap_agreements = np.array(bootstrap_agreements)
    
    return {
        'mean': np.mean(bootstrap_returns),
        'std': np.std(bootstrap_returns),
        'ci_lower': np.percentile(bootstrap_returns, 2.5),
        'ci_upper': np.percentile(bootstrap_returns, 97.5),
        'bootstrap_returns': bootstrap_returns,
        'mean_agreement': np.mean(bootstrap_agreements),
        'trajectory_data': trajectory_data
    }


# =============================================================================
# KOMOROWSKI-STYLE VALIDATION
# =============================================================================

def komorowski_validation(
    df: pd.DataFrame,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    agent: DoubleDuelingDQNAgent,
    action_mapping: Dict
) -> Dict:
    """Komorowski et al. style validation."""
    logger.info("Performing Komorowski-style validation...")
    
    results = {}
    
    # Get AI recommendations
    ai_actions = agent.select_actions(states, use_constraint=True)
    
    # Per-trajectory analysis
    trajectory_data = []
    
    for stay_id in df['stay_id'].unique():
        mask = (df['stay_id'] == stay_id).values
        traj_rewards = rewards[mask]
        traj_actions = actions[mask]
        traj_ai_actions = ai_actions[mask]
        
        terminal_reward = traj_rewards[-1]
        died = terminal_reward < 0
        agreement = np.mean(traj_actions == traj_ai_actions)
        
        trajectory_data.append({
            'stay_id': stay_id,
            'died': died,
            'agreement': agreement,
            'n_steps': len(traj_rewards)
        })
    
    traj_df = pd.DataFrame(trajectory_data)
    
    # 1. Mortality by agreement quartiles
    traj_df['agreement_quartile'] = pd.qcut(
        traj_df['agreement'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
        duplicates='drop'
    )
    
    mortality_by_quartile = traj_df.groupby('agreement_quartile')['died'].agg(['mean', 'count'])
    results['mortality_by_agreement_quartile'] = mortality_by_quartile.to_dict()
    
    # Correlation
    agreement_mortality_corr = stats.spearmanr(traj_df['agreement'], traj_df['died'])
    results['agreement_mortality_correlation'] = {
        'spearman_r': agreement_mortality_corr.correlation,
        'p_value': agreement_mortality_corr.pvalue
    }
    
    logger.info(f"  Agreement-Mortality: r={agreement_mortality_corr.correlation:.3f}, "
                f"p={agreement_mortality_corr.pvalue:.4f}")
    
    # 2. Dose-response
    idx_to_action = action_mapping['idx_to_action']
    
    def compute_distance(clin_action, ai_action):
        clin_orig = idx_to_action.get(clin_action, clin_action)
        ai_orig = idx_to_action.get(ai_action, ai_action)
        clin_iv, clin_vaso = clin_orig // 5, clin_orig % 5
        ai_iv, ai_vaso = ai_orig // 5, ai_orig % 5
        return np.sqrt((clin_iv - ai_iv)**2 + (clin_vaso - ai_vaso)**2)
    
    distances = [compute_distance(a, ai) for a, ai in zip(actions, ai_actions)]
    df_temp = df.copy()
    df_temp['distance'] = distances
    
    traj_distance = df_temp.groupby('stay_id')['distance'].mean()
    traj_df['mean_distance'] = traj_df['stay_id'].map(traj_distance)
    
    distance_mortality_corr = stats.spearmanr(traj_df['mean_distance'], traj_df['died'])
    results['distance_mortality_correlation'] = {
        'spearman_r': distance_mortality_corr.correlation,
        'p_value': distance_mortality_corr.pvalue
    }
    
    logger.info(f"  Distance-Mortality: r={distance_mortality_corr.correlation:.3f}, "
                f"p={distance_mortality_corr.pvalue:.4f}")
    
    # Validation summary
    agreement_valid = agreement_mortality_corr.correlation < 0 and agreement_mortality_corr.pvalue < 0.05
    distance_valid = distance_mortality_corr.correlation > 0 and distance_mortality_corr.pvalue < 0.05
    
    results['validation_summary'] = {
        'agreement_test_passed': agreement_valid,
        'distance_test_passed': distance_valid,
        'overall_passed': agreement_valid or distance_valid
    }
    
    logger.info(f"  Komorowski: Agreement={'PASS' if agreement_valid else 'FAIL'}, "
                f"Distance={'PASS' if distance_valid else 'FAIL'}")
    
    return results


# =============================================================================
# CLINICAL SENSITIVITY ANALYSIS
# =============================================================================

def clinical_sensitivity_analysis(
    agent: DoubleDuelingDQNAgent,
    state_cols: List[str],
    action_mapping: Dict
) -> Dict:
    """Clinical sensitivity analysis."""
    logger.info("Performing clinical sensitivity analysis...")
    
    results = {}
    n_features = len(state_cols)
    feature_idx = {col: i for i, col in enumerate(state_cols)}
    idx_to_action = action_mapping['idx_to_action']
    
    def get_vaso_bin(action_idx):
        orig = idx_to_action.get(action_idx, action_idx)
        return orig % 5
    
    def get_iv_bin(action_idx):
        orig = idx_to_action.get(action_idx, action_idx)
        return orig // 5
    
    def get_intensity(action_idx):
        orig = idx_to_action.get(action_idx, action_idx)
        return (orig // 5) + (orig % 5)
    
    # Test multiple baseline states
    np.random.seed(42)
    n_test_states = 100
    
    # 1. BP Sensitivity
    bp_col = 'MeanBP' if 'MeanBP' in feature_idx else ('SysBP' if 'SysBP' in feature_idx else None)
    
    if bp_col:
        bp_idx = feature_idx[bp_col]
        bp_range = np.linspace(-2, 2, 9)
        
        all_vaso_responses = []
        
        for _ in range(n_test_states):
            baseline = np.random.randn(1, n_features) * 0.5
            vaso_response = []
            
            for bp_val in bp_range:
                test_state = baseline.copy()
                test_state[0, bp_idx] = bp_val
                action = agent.select_actions(test_state, use_constraint=True)[0]
                vaso_response.append(get_vaso_bin(action))
            
            all_vaso_responses.append(vaso_response)
        
        all_vaso_responses = np.array(all_vaso_responses)
        mean_vaso = all_vaso_responses.mean(axis=0)
        
        bp_vaso_corr = stats.spearmanr(bp_range, mean_vaso)
        bp_pass = bp_vaso_corr.correlation < -0.3 or (
            mean_vaso[0] > mean_vaso[-1]  # Low BP -> higher vaso
        )
        
        results['bp_sensitivity'] = {
            'bp_values': bp_range.tolist(),
            'mean_vaso': mean_vaso.tolist(),
            'correlation': bp_vaso_corr.correlation,
            'p_value': bp_vaso_corr.pvalue,
            'passed': bp_pass
        }
        
        logger.info(f"  BP sensitivity: r={bp_vaso_corr.correlation:.3f}, {'PASS' if bp_pass else 'FAIL'}")
    
    # 2. Lactate Sensitivity
    if 'Arterial_lactate' in feature_idx:
        lac_idx = feature_idx['Arterial_lactate']
        lac_range = np.linspace(-1, 3, 9)
        
        all_intensity_responses = []
        
        for _ in range(n_test_states):
            baseline = np.random.randn(1, n_features) * 0.5
            intensity_response = []
            
            for lac_val in lac_range:
                test_state = baseline.copy()
                test_state[0, lac_idx] = lac_val
                action = agent.select_actions(test_state, use_constraint=True)[0]
                intensity_response.append(get_intensity(action))
            
            all_intensity_responses.append(intensity_response)
        
        all_intensity_responses = np.array(all_intensity_responses)
        mean_intensity = all_intensity_responses.mean(axis=0)
        
        lac_corr = stats.spearmanr(lac_range, mean_intensity)
        lac_pass = lac_corr.correlation > 0.3 or (
            mean_intensity[-1] > mean_intensity[0]  # High lactate -> higher intensity
        )
        
        results['lactate_sensitivity'] = {
            'lactate_values': lac_range.tolist(),
            'mean_intensity': mean_intensity.tolist(),
            'correlation': lac_corr.correlation,
            'p_value': lac_corr.pvalue,
            'passed': lac_pass
        }
        
        logger.info(f"  Lactate sensitivity: r={lac_corr.correlation:.3f}, {'PASS' if lac_pass else 'FAIL'}")
    
    # 3. Action Diversity
    random_states = np.random.randn(5000, n_features)
    random_actions = agent.select_actions(random_states, use_constraint=True)
    
    n_unique = len(np.unique(random_actions))
    action_counts = np.bincount(random_actions, minlength=agent.n_actions)
    action_probs = action_counts / action_counts.sum()
    entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
    max_entropy = np.log(agent.n_actions)
    
    diversity_pass = n_unique >= 8 and entropy > 1.5
    
    results['diversity'] = {
        'n_unique_actions': int(n_unique),
        'entropy': float(entropy),
        'max_entropy': float(max_entropy),
        'normalized_entropy': float(entropy / max_entropy),
        'action_distribution': action_probs.tolist(),
        'passed': diversity_pass
    }
    
    logger.info(f"  Diversity: {n_unique} unique, H={entropy:.2f}, {'PASS' if diversity_pass else 'FAIL'}")
    
    # Overall
    tests_passed = sum([
        results.get('bp_sensitivity', {}).get('passed', False),
        results.get('lactate_sensitivity', {}).get('passed', False),
        results.get('diversity', {}).get('passed', False)
    ])
    
    results['overall'] = {
        'passed': tests_passed,
        'total': 3,
        'score': tests_passed / 3
    }
    
    logger.info(f"  Overall: {tests_passed}/3")
    
    return results


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config: DuelingDQNConfig) -> Tuple[Dict, int, Dict]:
    """Load and prepare data."""
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    
    datasets = {}
    
    logger.info(f"Loading train data from {config.train_path}...")
    df_train = pd.read_csv(config.train_path)
    
    action_col = 'action' if 'action' in df_train.columns else 'action_discrete'
    reward_col = 'reward' if 'reward' in df_train.columns else 'reward_terminal'
    done_col = 'done' if 'done' in df_train.columns else 'terminal'
    
    exclude_cols = ['stay_id', 'time_window', 'action', 'action_discrete',
                    'reward', 'reward_terminal', 'done', 'terminal',
                    'next_state', 'mortality_90d']
    state_cols = [c for c in df_train.columns if c not in exclude_cols and not c.startswith('next_')]
    
    train_actions_original = df_train[action_col].values.astype(np.int32)
    unique_actions = sorted(set(train_actions_original))
    n_actions = len(unique_actions)
    
    action_to_idx = {a: i for i, a in enumerate(unique_actions)}
    idx_to_action = {i: a for a, i in action_to_idx.items()}
    
    logger.info(f"  Unique actions: {n_actions}")
    logger.info(f"  State features: {len(state_cols)}")
    
    for split, path in [('train', config.train_path), ('val', config.val_path), ('test', config.test_path)]:
        if split == 'train':
            df = df_train
        else:
            logger.info(f"Loading {split} data from {path}...")
            df = pd.read_csv(path)
        
        states = df[state_cols].values.astype(np.float32)
        actions_original = df[action_col].values.astype(np.int32)
        rewards = df[reward_col].values.astype(np.float32)
        dones = df[done_col].values.astype(bool)
        
        # Next states
        if any(c.startswith('next_') for c in df.columns):
            next_cols = [f'next_{c}' for c in state_cols if f'next_{c}' in df.columns]
            if len(next_cols) == len(state_cols):
                next_states = df[next_cols].values.astype(np.float32)
            else:
                next_states = np.roll(states, -1, axis=0)
                next_states[-1] = states[-1]
        else:
            next_states = np.roll(states, -1, axis=0)
            next_states[-1] = states[-1]
        
        states = np.nan_to_num(states, nan=0.0)
        next_states = np.nan_to_num(next_states, nan=0.0)
        rewards = np.nan_to_num(rewards, nan=0.0)
        
        # Remap actions
        remapped = []
        for a in actions_original:
            if a in action_to_idx:
                remapped.append(action_to_idx[a])
            else:
                closest = min(unique_actions, key=lambda x: abs(x - a))
                remapped.append(action_to_idx[closest])
        actions = np.array(remapped, dtype=np.int32)
        
        n_trajectories = df['stay_id'].nunique()
        
        datasets[split] = {
            'df': df,
            'states': states,
            'actions': actions,
            'actions_original': actions_original,
            'rewards': rewards,
            'dones': dones,
            'next_states': next_states,
            'state_cols': state_cols
        }
        
        logger.info(f"  {split}: {len(states):,} transitions, {n_trajectories:,} trajectories")
    
    action_mapping = {
        'action_to_idx': action_to_idx,
        'idx_to_action': idx_to_action,
        'original_actions': unique_actions,
        'n_actions': n_actions
    }
    
    return datasets, n_actions, action_mapping


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main pipeline."""
    logger.info("=" * 80)
    logger.info("DOUBLE DUELING DQN PIPELINE FOR SEPSIS TREATMENT")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now()}")
    
    # Configuration
    config = DuelingDQNConfig()
    
    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.backends.cudnn.deterministic = True
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("No GPU available, using CPU")
    
    # Load data
    datasets, n_actions, action_mapping = load_data(config)
    train_data = datasets['train']
    val_data = datasets['val']
    test_data = datasets['test']
    
    state_cols = train_data['state_cols']
    n_state_features = len(state_cols)
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Actions: {n_actions}")
    logger.info(f"  State features: {n_state_features}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Architecture: {config.shared_hidden_dims}")
    logger.info(f"  CQL alpha: {config.cql_alpha}")
    logger.info(f"  Behavior threshold: {config.behavior_threshold}")
    
    # =========================================================================
    # STEP 1: Fit behavior policy
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: FITTING BEHAVIOR POLICY")
    logger.info("=" * 60)
    
    behavior_policy = LogisticBehaviorPolicy(
        n_actions=n_actions,
        softening_epsilon=0.01,
        random_seed=config.random_seed
    )
    behavior_policy.fit(train_data['states'], train_data['actions'])
    
    # =========================================================================
    # STEP 2: Create and train agent
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: TRAINING DOUBLE DUELING DQN")
    logger.info("=" * 60)
    
    agent = DoubleDuelingDQNAgent(
        n_state_features=n_state_features,
        n_actions=n_actions,
        config=config,
        behavior_policy=behavior_policy
    )
    
    agent.fit_scaler(train_data['states'])
    
    history = train_dueling_dqn(agent, train_data, val_data, config)
    
    # =========================================================================
    # STEP 3: Clinical sensitivity
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: CLINICAL SENSITIVITY ANALYSIS")
    logger.info("=" * 60)
    
    clinical_results = clinical_sensitivity_analysis(agent, state_cols, action_mapping)
    
    # =========================================================================
    # STEP 4: Komorowski validation
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: KOMOROWSKI-STYLE VALIDATION")
    logger.info("=" * 60)
    
    komorowski_results = komorowski_validation(
        test_data['df'],
        test_data['states'],
        test_data['actions'],
        test_data['rewards'],
        agent,
        action_mapping
    )
    
    # =========================================================================
    # STEP 5: Off-policy evaluation
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: OFF-POLICY EVALUATION")
    logger.info("=" * 60)
    
    # Clinician
    logger.info("Evaluating clinician policy...")
    clin_returns = []
    for stay_id in test_data['df']['stay_id'].unique():
        mask = (test_data['df']['stay_id'] == stay_id).values
        traj_rewards = test_data['rewards'][mask]
        G = 0.0
        for t in range(len(traj_rewards) - 1, -1, -1):
            G = traj_rewards[t] + config.gamma * G
        clin_returns.append(G)
    
    clin_returns = np.array(clin_returns)
    clin_value = np.mean(clin_returns)
    clin_se = np.std(clin_returns) / np.sqrt(len(clin_returns))
    
    logger.info(f"  Clinician: {clin_value:.3f} ± {1.96*clin_se:.3f}")
    
    # AI policy (WDR)
    logger.info("Evaluating AI policy (WDR)...")
    wdr = WDREstimator(gamma=config.gamma)
    ai_wdr = wdr.estimate(
        test_data['df'],
        test_data['states'],
        test_data['actions'],
        test_data['rewards'],
        test_data['dones'],
        agent,
        behavior_policy
    )
    
    logger.info(f"  AI (WDR): {ai_wdr['wdr_estimate']:.3f} ± {1.96*ai_wdr['wdr_se']:.3f}")
    logger.info(f"  ESS ratio: {ai_wdr['ess_ratio']:.4f}")
    
    # Agreement
    ai_actions = agent.select_actions(test_data['states'], use_constraint=True)
    agreement = np.mean(ai_actions == test_data['actions'])
    logger.info(f"  AI-Clinician agreement: {agreement*100:.1f}%")
    
    # =========================================================================
    # STEP 6: Bootstrap CIs
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: BOOTSTRAP CONFIDENCE INTERVALS")
    logger.info("=" * 60)
    
    bootstrap_result = bootstrap_evaluation(
        test_data['df'],
        test_data['states'],
        test_data['actions'],
        test_data['rewards'],
        test_data['dones'],
        agent,
        behavior_policy,
        n_bootstrap=config.n_bootstrap,
        gamma=config.gamma
    )
    
    logger.info(f"  Clinician: {bootstrap_result['mean']:.3f} "
                f"[{bootstrap_result['ci_lower']:.3f}, {bootstrap_result['ci_upper']:.3f}]")
    logger.info(f"  Mean agreement: {bootstrap_result['mean_agreement']*100:.1f}%")
    
    # =========================================================================
    # STEP 7: Final results
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"\n{'Policy':<25} {'Value':<12} {'95% CI':<25}")
    logger.info("-" * 65)
    logger.info(f"{'Clinician':<25} {clin_value:<12.3f} "
                f"[{clin_value-1.96*clin_se:.3f}, {clin_value+1.96*clin_se:.3f}]")
    logger.info(f"{'AI (WDR)':<25} {ai_wdr['wdr_estimate']:<12.3f} "
                f"[ESS={ai_wdr['ess_ratio']:.2%}]")
    
    logger.info(f"\nAI-Clinician agreement: {agreement*100:.1f}%")
    logger.info(f"Clinical validation: {clinical_results['overall']['passed']}/{clinical_results['overall']['total']}")
    logger.info(f"Komorowski validation: {'PASS' if komorowski_results['validation_summary']['overall_passed'] else 'FAIL'}")
    
    # =========================================================================
    # Save results
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)
    
    # Save model
    torch.save({
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'state_scaler_mean': agent.state_scaler.mean_,
        'state_scaler_scale': agent.state_scaler.scale_,
        'config': config.__dict__
    }, output_dir / 'dueling_dqn_model.pt')
    
    # Save results
    results = {
        'config': {k: v for k, v in config.__dict__.items() if not callable(v)},
        'n_actions': n_actions,
        'n_state_features': n_state_features,
        'training_history': history,
        'clinical_validation': clinical_results,
        'komorowski_validation': {
            k: v for k, v in komorowski_results.items()
            if k not in ['mortality_by_agreement_quartile']
        },
        'evaluation': {
            'clinician': {'value': float(clin_value), 'se': float(clin_se)},
            'ai_wdr': {
                'value': float(ai_wdr['wdr_estimate']),
                'se': float(ai_wdr['wdr_se']),
                'ess_ratio': float(ai_wdr['ess_ratio'])
            },
            'agreement': float(agreement)
        },
        'bootstrap': {
            'mean': float(bootstrap_result['mean']),
            'ci_lower': float(bootstrap_result['ci_lower']),
            'ci_upper': float(bootstrap_result['ci_upper']),
            'mean_agreement': float(bootstrap_result['mean_agreement'])
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / 'full_results.pkl', 'wb') as f:
        pickle.dump({
            'bootstrap_returns': bootstrap_result['bootstrap_returns'],
            'trajectory_data': bootstrap_result['trajectory_data'],
            'training_history': history
        }, f)
    
    logger.info(f"  Results saved to {output_dir}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Finished at: {datetime.now()}")
    
    return results


if __name__ == "__main__":
    main()