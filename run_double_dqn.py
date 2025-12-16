#!/usr/bin/env python3
"""
Double DQN for Sepsis Treatment Optimization
=============================================

This implementation includes:
1. Double DQN with neural network Q-function
2. Experience replay (offline - using full dataset)
3. Fixed target network updates
4. BCQ-style behavioral constraint for offline RL
5. WDR Off-Policy Evaluation with bootstrap CIs
6. Komorowski-style validation (mortality vs agreement, dose-response)
7. Clinical sensitivity analysis
8. Full statistical significance testing

Based on:
- Van Hasselt et al., 2016 (Double DQN)
- Komorowski et al., 2018 (AI Clinician)
- Course materials (DQN variants)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
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

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('double_dqn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DoubleDQNConfig:
    """Configuration for Double DQN pipeline."""
    # Data paths
    train_path: str = 'data/processed/train_trajectories.csv'
    val_path: str = 'data/processed/val_trajectories.csv'
    test_path: str = 'data/processed/test_trajectories.csv'
    output_dir: str = 'outputs/double_dqn'
    
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    dropout_rate: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 256
    gamma: float = 0.99
    n_epochs: int = 100
    
    # Double DQN specific
    target_update_freq: int = 500  # Update target network every N steps
    soft_update_tau: float = 0.005  # Soft update coefficient (if using soft updates)
    use_soft_update: bool = True
    
    # Behavioral constraint (BCQ-style)
    use_behavior_constraint: bool = True
    behavior_threshold: float = 0.01  # Only allow actions with P(a|s) > threshold
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Evaluation
    n_bootstrap: int = 1000
    
    # Hyperparameter search
    n_hyperparameter_trials: int = 20
    
    # Reproducibility
    random_seed: int = 42
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# NEURAL NETWORK Q-FUNCTION
# =============================================================================

class QNetwork(nn.Module):
    """
    Neural network for Q-value estimation.
    
    Architecture: State -> Hidden layers -> Q(s,a) for all actions
    """
    
    def __init__(
        self,
        n_state_features: int,
        n_actions: int,
        hidden_dims: List[int] = [128, 128],
        dropout_rate: float = 0.1
    ):
        super(QNetwork, self).__init__()
        
        self.n_state_features = n_state_features
        self.n_actions = n_actions
        
        # Build network layers
        layers = []
        prev_dim = n_state_features
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer: Q-values for all actions
        layers.append(nn.Linear(prev_dim, n_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            states: (batch_size, n_state_features) tensor
            
        Returns:
            Q-values: (batch_size, n_actions) tensor
        """
        return self.network(states)
    
    def get_q_values(self, states: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get Q-values for given states and optionally specific actions.
        
        Args:
            states: (batch_size, n_state_features) tensor
            actions: Optional (batch_size,) tensor of action indices
            
        Returns:
            If actions is None: (batch_size, n_actions) Q-values
            If actions provided: (batch_size,) Q-values for those actions
        """
        q_all = self.forward(states)
        
        if actions is None:
            return q_all
        else:
            return q_all.gather(1, actions.unsqueeze(1)).squeeze(1)


# =============================================================================
# DOUBLE DQN AGENT
# =============================================================================

class DoubleDQNAgent:
    """
    Double DQN Agent for offline RL.
    
    Key features:
    - Online network for action selection
    - Target network for value estimation (reduces overestimation)
    - BCQ-style behavioral constraint for offline RL
    """
    
    def __init__(
        self,
        n_state_features: int,
        n_actions: int,
        config: DoubleDQNConfig,
        behavior_policy: Optional['LogisticBehaviorPolicy'] = None
    ):
        self.n_state_features = n_state_features
        self.n_actions = n_actions
        self.config = config
        self.behavior_policy = behavior_policy
        self.device = torch.device(config.device)
        
        # Online Q-network
        self.q_network = QNetwork(
            n_state_features=n_state_features,
            n_actions=n_actions,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate
        ).to(self.device)
        
        # Target Q-network (copy of online network)
        self.target_network = QNetwork(
            n_state_features=n_state_features,
            n_actions=n_actions,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate
        ).to(self.device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
        # State scaler
        self.state_scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Training stats
        self.update_count = 0
        self.training_losses = []
    
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
        
        Args:
            states: (batch_size, n_state_features) array
            
        Returns:
            mask: (batch_size, n_actions) boolean array (True = allowed)
        """
        if not self.config.use_behavior_constraint or self.behavior_policy is None:
            return np.ones((len(states), self.n_actions), dtype=bool)
        
        # Get behavior policy probabilities
        probs = self.behavior_policy.predict_probs(states)
        
        # Allow actions with probability above threshold
        mask = probs > self.config.behavior_threshold
        
        # Ensure at least one action is allowed per state
        for i in range(len(mask)):
            if not mask[i].any():
                mask[i, np.argmax(probs[i])] = True
        
        return mask
    
    def select_actions(self, states: np.ndarray, use_constraint: bool = True) -> np.ndarray:
        """
        Select greedy actions (with optional behavioral constraint).
        
        Args:
            states: (batch_size, n_state_features) array (unscaled)
            use_constraint: Whether to apply BCQ constraint
            
        Returns:
            actions: (batch_size,) array of action indices
        """
        # Scale states
        states_scaled = self.scale_states(states)
        states_tensor = torch.FloatTensor(states_scaled).to(self.device)
        
        # Get Q-values
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(states_tensor).cpu().numpy()
        
        if use_constraint and self.config.use_behavior_constraint:
            # Apply behavioral constraint
            mask = self.get_action_mask(states)
            # Set Q-values of disallowed actions to -inf
            q_values[~mask] = -np.inf
        
        # Select greedy actions
        actions = np.argmax(q_values, axis=1)
        
        return actions
    
    def compute_double_dqn_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        next_action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Double DQN targets.
        
        Double DQN: Use online network for action selection,
                    target network for value estimation.
        
        target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
        
        Args:
            rewards: (batch_size,) tensor
            next_states: (batch_size, n_state_features) tensor
            dones: (batch_size,) tensor
            next_action_mask: Optional (batch_size, n_actions) mask tensor
            
        Returns:
            targets: (batch_size,) tensor
        """
        with torch.no_grad():
            # Step 1: Use ONLINE network to select best actions
            next_q_online = self.q_network(next_states)
            
            if next_action_mask is not None:
                # Mask disallowed actions
                next_q_online = next_q_online.masked_fill(~next_action_mask, -1e9)
            
            best_actions = next_q_online.argmax(dim=1)
            
            # Step 2: Use TARGET network to evaluate those actions
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
        action_mask: Optional[torch.Tensor] = None,
        next_action_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Perform one update step.
        
        Args:
            states: (batch_size, n_state_features) tensor
            actions: (batch_size,) tensor
            rewards: (batch_size,) tensor
            next_states: (batch_size, n_state_features) tensor
            dones: (batch_size,) tensor
            action_mask: Optional mask for current state actions
            next_action_mask: Optional mask for next state actions
            
        Returns:
            loss: float
        """
        self.q_network.train()
        
        # Compute current Q-values
        current_q = self.q_network.get_q_values(states, actions)
        
        # Compute Double DQN targets
        targets = self.compute_double_dqn_targets(
            rewards, next_states, dones, next_action_mask
        )
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.config.use_soft_update:
            self._soft_update_target()
        elif self.update_count % self.config.target_update_freq == 0:
            self._hard_update_target()
        
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value
    
    def _soft_update_target(self):
        """Soft update: θ_target = τ*θ_online + (1-τ)*θ_target"""
        tau = self.config.soft_update_tau
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1 - tau) * target_param.data
            )
    
    def _hard_update_target(self):
        """Hard update: copy online network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_all_q_values(self, states: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions.
        
        Args:
            states: (batch_size, n_state_features) array (unscaled)
            
        Returns:
            q_values: (batch_size, n_actions) array
        """
        states_scaled = self.scale_states(states)
        states_tensor = torch.FloatTensor(states_scaled).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(states_tensor).cpu().numpy()
        
        return q_values


# =============================================================================
# BEHAVIOR POLICY (LOGISTIC REGRESSION)
# =============================================================================

class LogisticBehaviorPolicy:
    """
    State-dependent behavior policy using multinomial logistic regression.
    """
    
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
        
        # Scale states
        states_scaled = self.state_scaler.fit_transform(states)
        
        # Fit logistic regression
        self.model.fit(states_scaled, actions)
        self.classes_ = self.model.classes_
        
        self.fitted = True
        logger.info(f"  Behavior policy fitted. Classes: {len(self.classes_)}")
    
    def predict_probs(self, states: np.ndarray) -> np.ndarray:
        """
        Predict action probabilities for given states.
        
        Returns probabilities for all n_actions, padding zeros for unseen classes.
        """
        if not self.fitted:
            raise RuntimeError("Behavior policy not fitted.")
        
        states_scaled = self.state_scaler.transform(states)
        probs_raw = self.model.predict_proba(states_scaled)
        
        # Handle case where not all actions were seen
        n_samples = len(states)
        probs = np.full((n_samples, self.n_actions), self.softening_epsilon / self.n_actions)
        
        for i, cls in enumerate(self.classes_):
            probs[:, int(cls)] = probs_raw[:, i]
        
        # Apply softening
        uniform = 1.0 / self.n_actions
        probs = (1 - self.softening_epsilon) * probs + self.softening_epsilon * uniform
        
        # Normalize
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs
    
    def get_prob(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Get probability of specific actions."""
        probs = self.predict_probs(states)
        return probs[np.arange(len(actions)), actions.astype(int)]


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_double_dqn(
    agent: DoubleDQNAgent,
    train_data: Dict,
    val_data: Dict,
    config: DoubleDQNConfig
) -> Dict:
    """
    Train Double DQN agent.
    
    Args:
        agent: DoubleDQNAgent instance
        train_data: Training data dictionary
        val_data: Validation data dictionary
        config: Configuration
        
    Returns:
        history: Training history
    """
    logger.info("Starting Double DQN training...")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Epochs: {config.n_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    
    device = torch.device(config.device)
    
    # Prepare data
    train_states = agent.scale_states(train_data['states'])
    train_actions = train_data['actions']
    train_rewards = train_data['rewards']
    train_next_states = agent.scale_states(train_data['next_states'])
    train_dones = train_data['dones']
    
    # Get action masks if using behavioral constraint
    if config.use_behavior_constraint and agent.behavior_policy is not None:
        train_action_mask = agent.get_action_mask(train_data['states'])
        train_next_action_mask = agent.get_action_mask(train_data['next_states'])
    else:
        train_action_mask = None
        train_next_action_mask = None
    
    # Create data loader
    dataset = TensorDataset(
        torch.FloatTensor(train_states),
        torch.LongTensor(train_actions),
        torch.FloatTensor(train_rewards),
        torch.FloatTensor(train_next_states),
        torch.FloatTensor(train_dones.astype(float))
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_q_mean': [],
        'val_q_std': [],
        'n_unique_actions': [],
        'action_entropy': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None
    
    for epoch in range(config.n_epochs):
        # Training
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            states, actions, rewards, next_states, dones = [b.to(device) for b in batch]
            
            # Get action masks for this batch
            if train_action_mask is not None:
                batch_start = batch_idx * config.batch_size
                batch_end = min(batch_start + config.batch_size, len(train_action_mask))
                action_mask = torch.BoolTensor(train_action_mask[batch_start:batch_end]).to(device)
                next_mask = torch.BoolTensor(train_next_action_mask[batch_start:batch_end]).to(device)
            else:
                action_mask = None
                next_mask = None
            
            loss = agent.update(
                states, actions, rewards, next_states, dones,
                action_mask, next_mask
            )
            epoch_losses.append(loss)
        
        train_loss = np.mean(epoch_losses)
        history['train_loss'].append(train_loss)
        
        # Validation
        val_states = agent.scale_states(val_data['states'])
        val_states_tensor = torch.FloatTensor(val_states).to(device)
        val_actions_tensor = torch.LongTensor(val_data['actions']).to(device)
        val_rewards_tensor = torch.FloatTensor(val_data['rewards']).to(device)
        val_next_states_tensor = torch.FloatTensor(agent.scale_states(val_data['next_states'])).to(device)
        val_dones_tensor = torch.FloatTensor(val_data['dones'].astype(float)).to(device)
        
        agent.q_network.eval()
        with torch.no_grad():
            val_q = agent.q_network.get_q_values(val_states_tensor, val_actions_tensor)
            val_targets = agent.compute_double_dqn_targets(
                val_rewards_tensor, val_next_states_tensor, val_dones_tensor
            )
            val_loss = F.smooth_l1_loss(val_q, val_targets).item()
            
            # Q-value statistics
            val_q_all = agent.q_network(val_states_tensor)
            q_mean = val_q_all.mean().item()
            q_std = val_q_all.std().item()
        
        history['val_loss'].append(val_loss)
        history['val_q_mean'].append(q_mean)
        history['val_q_std'].append(q_std)
        
        # Action diversity
        val_actions_pred = agent.select_actions(val_data['states'][:10000])
        n_unique = len(np.unique(val_actions_pred))
        action_counts = np.bincount(val_actions_pred, minlength=agent.n_actions)
        action_probs = action_counts / action_counts.sum()
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        
        history['n_unique_actions'].append(n_unique)
        history['action_entropy'].append(action_entropy)
        
        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, Q={q_mean:.2f}±{q_std:.2f}, "
                f"Actions={n_unique}, H={action_entropy:.2f}"
            )
        
        # Early stopping
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = agent.q_network.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_state_dict is not None:
        agent.q_network.load_state_dict(best_state_dict)
        # Also update target network
        agent.target_network.load_state_dict(best_state_dict)
    
    logger.info(f"  Training complete. Best val loss: {best_val_loss:.4f}")
    
    return history


# =============================================================================
# OFF-POLICY EVALUATION (WDR)
# =============================================================================

class WDREstimator:
    """Weighted Doubly Robust estimator for off-policy evaluation."""
    
    def __init__(self, gamma: float = 0.99, max_weight: float = 100.0):
        self.gamma = gamma
        self.max_weight = max_weight
    
    def estimate(
        self,
        df: pd.DataFrame,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        agent: DoubleDQNAgent,
        behavior_policy: LogisticBehaviorPolicy
    ) -> Dict:
        """
        Compute WDR estimate.
        
        Returns:
            Dictionary with WDR estimate, standard error, and ESS
        """
        trajectory_values = []
        trajectory_weights = []
        
        for stay_id in df['stay_id'].unique():
            mask = (df['stay_id'] == stay_id).values
            
            traj_states = states[mask]
            traj_actions = actions[mask]
            traj_rewards = rewards[mask]
            traj_dones = dones[mask]
            
            # Get Q-values
            q_values = agent.get_all_q_values(traj_states)
            
            # Get behavior policy probabilities
            pi_b = behavior_policy.get_prob(traj_states, traj_actions)
            pi_b = np.clip(pi_b, 1e-6, 1.0)
            
            # Get target policy (greedy with constraint)
            target_actions = agent.select_actions(traj_states, use_constraint=True)
            pi_e = (target_actions == traj_actions).astype(float)
            
            # Make target policy slightly stochastic for IS stability
            pi_e = 0.99 * pi_e + 0.01 / agent.n_actions
            
            # Importance weights
            rho = pi_e / pi_b
            rho = np.clip(rho, 0, self.max_weight)
            
            # WDR computation
            T = len(traj_rewards)
            v_wdr = 0.0
            cumulative_rho = 1.0
            
            for t in range(T):
                q_t = q_values[t, int(traj_actions[t])]
                v_t = np.sum(q_values[t] * (pi_e[t] if t < T else 0))  # Simplified
                
                cumulative_rho = np.clip(cumulative_rho * rho[t], 0, self.max_weight)
                
                if t == 0:
                    v_wdr = v_t
                
                advantage = traj_rewards[t] + (0 if traj_dones[t] else self.gamma * q_values[min(t+1, T-1)].max()) - q_t
                v_wdr += (self.gamma ** t) * cumulative_rho * advantage
            
            trajectory_values.append(v_wdr)
            trajectory_weights.append(cumulative_rho)
        
        trajectory_values = np.array(trajectory_values)
        trajectory_weights = np.array(trajectory_weights)
        
        # Compute ESS
        ess = (np.sum(trajectory_weights) ** 2) / (np.sum(trajectory_weights ** 2) + 1e-10)
        ess_ratio = ess / len(trajectory_weights)
        
        return {
            'wdr_estimate': np.mean(trajectory_values),
            'wdr_std': np.std(trajectory_values),
            'wdr_se': np.std(trajectory_values) / np.sqrt(len(trajectory_values)),
            'ess': ess,
            'ess_ratio': ess_ratio,
            'n_trajectories': len(trajectory_values),
            'trajectory_values': trajectory_values
        }


def bootstrap_ci(
    df: pd.DataFrame,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    agent: DoubleDQNAgent,
    behavior_policy: LogisticBehaviorPolicy,
    n_bootstrap: int = 1000,
    gamma: float = 0.99
) -> Dict:
    """Compute bootstrap confidence intervals."""
    logger.info(f"Computing bootstrap CI ({n_bootstrap} samples)...")
    
    wdr = WDREstimator(gamma=gamma)
    
    # Get unique trajectories
    stay_ids = df['stay_id'].unique()
    n_trajectories = len(stay_ids)
    
    bootstrap_estimates = []
    bootstrap_ess = []
    
    for b in range(n_bootstrap):
        if (b + 1) % 100 == 0:
            logger.info(f"  Bootstrap {b+1}/{n_bootstrap}")
        
        # Resample trajectories
        sampled_ids = np.random.choice(stay_ids, size=n_trajectories, replace=True)
        
        # Build resampled dataset
        masks = [df['stay_id'] == sid for sid in sampled_ids]
        combined_mask = np.zeros(len(df), dtype=bool)
        indices = []
        for sid in sampled_ids:
            idx = np.where(df['stay_id'] == sid)[0]
            indices.extend(idx.tolist())
        
        boot_states = states[indices]
        boot_actions = actions[indices]
        boot_rewards = rewards[indices]
        boot_dones = dones[indices]
        boot_df = df.iloc[indices].copy()
        boot_df['stay_id'] = np.repeat(np.arange(len(sampled_ids)), 
                                        [np.sum(df['stay_id'] == sid) for sid in sampled_ids])
        
        result = wdr.estimate(
            boot_df, boot_states, boot_actions, boot_rewards, boot_dones,
            agent, behavior_policy
        )
        
        bootstrap_estimates.append(result['wdr_estimate'])
        bootstrap_ess.append(result['ess_ratio'])
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    return {
        'mean': np.mean(bootstrap_estimates),
        'std': np.std(bootstrap_estimates),
        'ci_lower': np.percentile(bootstrap_estimates, 2.5),
        'ci_upper': np.percentile(bootstrap_estimates, 97.5),
        'bootstrap_estimates': bootstrap_estimates,
        'mean_ess': np.mean(bootstrap_ess)
    }


# =============================================================================
# KOMOROWSKI-STYLE VALIDATION
# =============================================================================

def komorowski_validation(
    df: pd.DataFrame,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    agent: DoubleDQNAgent,
    action_mapping: Dict
) -> Dict:
    """
    Komorowski et al. style validation:
    1. Mortality rate vs AI-clinician agreement
    2. Dose-response curves (mortality vs distance from AI recommendation)
    """
    logger.info("Performing Komorowski-style validation...")
    
    results = {}
    
    # Get AI recommendations
    ai_actions = agent.select_actions(states, use_constraint=True)
    
    # =========================================================================
    # 1. Mortality vs Agreement Analysis
    # =========================================================================
    logger.info("  Computing mortality vs agreement...")
    
    # Get terminal outcomes per trajectory
    trajectory_data = []
    
    for stay_id in df['stay_id'].unique():
        mask = (df['stay_id'] == stay_id).values
        traj_rewards = rewards[mask]
        traj_actions = actions[mask]
        traj_ai_actions = ai_actions[mask]
        
        # Terminal outcome
        terminal_reward = traj_rewards[-1]
        died = terminal_reward < 0
        
        # Agreement: fraction of timesteps where AI == clinician
        agreement = np.mean(traj_actions == traj_ai_actions)
        
        trajectory_data.append({
            'stay_id': stay_id,
            'died': died,
            'agreement': agreement,
            'n_steps': len(traj_rewards)
        })
    
    traj_df = pd.DataFrame(trajectory_data)
    
    # Bin by agreement level
    agreement_bins = [0, 0.25, 0.5, 0.75, 1.0]
    traj_df['agreement_bin'] = pd.cut(traj_df['agreement'], bins=agreement_bins)
    
    mortality_by_agreement = traj_df.groupby('agreement_bin')['died'].agg(['mean', 'count', 'std'])
    mortality_by_agreement.columns = ['mortality_rate', 'n_patients', 'std']
    mortality_by_agreement['se'] = mortality_by_agreement['std'] / np.sqrt(mortality_by_agreement['n_patients'])
    
    results['mortality_by_agreement'] = mortality_by_agreement.to_dict()
    
    # Test: Does higher agreement correlate with lower mortality?
    agreement_mortality_corr = stats.spearmanr(traj_df['agreement'], traj_df['died'])
    results['agreement_mortality_correlation'] = {
        'spearman_r': agreement_mortality_corr.correlation,
        'p_value': agreement_mortality_corr.pvalue
    }
    
    logger.info(f"    Agreement-Mortality correlation: r={agreement_mortality_corr.correlation:.3f}, p={agreement_mortality_corr.pvalue:.4f}")
    
    # =========================================================================
    # 2. Dose-Response Analysis
    # =========================================================================
    logger.info("  Computing dose-response curves...")
    
    # Compute "distance" between AI and clinician actions
    # Distance in IV/Vaso space
    idx_to_action = action_mapping['idx_to_action']
    
    def action_to_iv_vaso(action_idx):
        """Convert action index to (IV bin, Vaso bin)."""
        orig_action = idx_to_action.get(action_idx, action_idx)
        iv_bin = orig_action // 5
        vaso_bin = orig_action % 5
        return iv_bin, vaso_bin
    
    # Per-timestep analysis
    timestep_data = []
    
    for i in range(len(actions)):
        clin_iv, clin_vaso = action_to_iv_vaso(actions[i])
        ai_iv, ai_vaso = action_to_iv_vaso(ai_actions[i])
        
        # Euclidean distance in (IV, Vaso) space
        distance = np.sqrt((clin_iv - ai_iv)**2 + (clin_vaso - ai_vaso)**2)
        
        timestep_data.append({
            'distance': distance,
            'clin_iv': clin_iv,
            'clin_vaso': clin_vaso,
            'ai_iv': ai_iv,
            'ai_vaso': ai_vaso
        })
    
    timestep_df = pd.DataFrame(timestep_data)
    timestep_df['stay_id'] = df['stay_id'].values
    timestep_df['reward'] = rewards
    
    # Get terminal outcomes
    terminal_mask = df.groupby('stay_id').cumcount(ascending=False) == 0
    terminal_df = timestep_df[terminal_mask.values].copy()
    terminal_df['died'] = terminal_df['reward'] < 0
    
    # Aggregate distance per trajectory (mean distance from AI recommendation)
    traj_distance = timestep_df.groupby('stay_id')['distance'].mean().reset_index()
    traj_distance.columns = ['stay_id', 'mean_distance']
    
    traj_outcomes = terminal_df[['stay_id', 'died']].drop_duplicates()
    dose_response_df = traj_distance.merge(traj_outcomes, on='stay_id')
    
    # Bin by distance
    distance_bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
    dose_response_df['distance_bin'] = pd.cut(dose_response_df['mean_distance'], bins=distance_bins)
    
    mortality_by_distance = dose_response_df.groupby('distance_bin')['died'].agg(['mean', 'count'])
    mortality_by_distance.columns = ['mortality_rate', 'n_patients']
    
    results['mortality_by_distance'] = mortality_by_distance.to_dict()
    
    # Test: Does higher distance correlate with higher mortality?
    distance_mortality_corr = stats.spearmanr(dose_response_df['mean_distance'], dose_response_df['died'])
    results['distance_mortality_correlation'] = {
        'spearman_r': distance_mortality_corr.correlation,
        'p_value': distance_mortality_corr.pvalue
    }
    
    logger.info(f"    Distance-Mortality correlation: r={distance_mortality_corr.correlation:.3f}, p={distance_mortality_corr.pvalue:.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    # Expected: negative correlation for agreement (more agreement = less death)
    #           positive correlation for distance (more distance = more death)
    
    agreement_valid = agreement_mortality_corr.correlation < 0 and agreement_mortality_corr.pvalue < 0.05
    distance_valid = distance_mortality_corr.correlation > 0 and distance_mortality_corr.pvalue < 0.05
    
    results['validation_summary'] = {
        'agreement_test_passed': agreement_valid,
        'distance_test_passed': distance_valid,
        'overall_passed': agreement_valid or distance_valid
    }
    
    logger.info(f"  Komorowski validation: Agreement test={'PASS' if agreement_valid else 'FAIL'}, "
                f"Distance test={'PASS' if distance_valid else 'FAIL'}")
    
    return results


# =============================================================================
# CLINICAL SENSITIVITY ANALYSIS
# =============================================================================

def clinical_sensitivity_analysis(
    agent: DoubleDQNAgent,
    state_cols: List[str],
    action_mapping: Dict
) -> Dict:
    """
    Clinical sensitivity analysis: Does the policy respond appropriately
    to changes in clinical variables?
    """
    logger.info("Performing clinical sensitivity analysis...")
    
    results = {}
    n_features = len(state_cols)
    
    # Create baseline state (zeros = population mean after scaling)
    baseline_state = np.zeros((1, n_features))
    
    # Feature indices
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
    
    # =========================================================================
    # 1. BP Sensitivity: Low BP should -> higher vasopressor
    # =========================================================================
    if 'MeanBP' in feature_idx or 'SysBP' in feature_idx:
        bp_col = 'MeanBP' if 'MeanBP' in feature_idx else 'SysBP'
        bp_idx = feature_idx[bp_col]
        
        bp_range = np.linspace(-2, 2, 9)  # Standardized values
        vaso_recommendations = []
        
        for bp_val in bp_range:
            test_state = baseline_state.copy()
            test_state[0, bp_idx] = bp_val
            action = agent.select_actions(test_state, use_constraint=True)[0]
            vaso_recommendations.append(get_vaso_bin(action))
        
        # Check if vaso increases as BP decreases
        bp_vaso_corr = stats.spearmanr(bp_range, vaso_recommendations)
        bp_pass = bp_vaso_corr.correlation < -0.3  # Negative correlation expected
        
        results['bp_sensitivity'] = {
            'bp_values': bp_range.tolist(),
            'vaso_recommendations': vaso_recommendations,
            'correlation': bp_vaso_corr.correlation,
            'p_value': bp_vaso_corr.pvalue,
            'passed': bp_pass
        }
        
        logger.info(f"  BP sensitivity: r={bp_vaso_corr.correlation:.3f}, {'PASS' if bp_pass else 'FAIL'}")
    
    # =========================================================================
    # 2. Lactate Sensitivity: High lactate should -> more aggressive treatment
    # =========================================================================
    if 'Arterial_lactate' in feature_idx:
        lac_idx = feature_idx['Arterial_lactate']
        
        lac_range = np.linspace(-1, 3, 9)
        intensity_recommendations = []
        
        for lac_val in lac_range:
            test_state = baseline_state.copy()
            test_state[0, lac_idx] = lac_val
            action = agent.select_actions(test_state, use_constraint=True)[0]
            intensity_recommendations.append(get_intensity(action))
        
        lac_intensity_corr = stats.spearmanr(lac_range, intensity_recommendations)
        lac_pass = lac_intensity_corr.correlation > 0.3
        
        results['lactate_sensitivity'] = {
            'lactate_values': lac_range.tolist(),
            'intensity_recommendations': intensity_recommendations,
            'correlation': lac_intensity_corr.correlation,
            'p_value': lac_intensity_corr.pvalue,
            'passed': lac_pass
        }
        
        logger.info(f"  Lactate sensitivity: r={lac_intensity_corr.correlation:.3f}, {'PASS' if lac_pass else 'FAIL'}")
    
    # =========================================================================
    # 3. Action Diversity
    # =========================================================================
    # Sample random states and check action diversity
    random_states = np.random.randn(1000, n_features)
    random_actions = agent.select_actions(random_states, use_constraint=True)
    
    n_unique = len(np.unique(random_actions))
    action_counts = np.bincount(random_actions, minlength=agent.n_actions)
    action_probs = action_counts / action_counts.sum()
    entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
    max_entropy = np.log(agent.n_actions)
    
    diversity_pass = n_unique >= 5 and entropy > 1.0
    
    results['diversity'] = {
        'n_unique_actions': n_unique,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'normalized_entropy': entropy / max_entropy,
        'passed': diversity_pass
    }
    
    logger.info(f"  Action diversity: {n_unique} unique, H={entropy:.2f}, {'PASS' if diversity_pass else 'FAIL'}")
    
    # Overall score
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
    
    logger.info(f"  Overall clinical validation: {tests_passed}/3")
    
    return results


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config: DoubleDQNConfig) -> Tuple[Dict, int, Dict]:
    """Load and prepare data."""
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    
    datasets = {}
    
    # Load training data first to establish action mapping
    logger.info(f"Loading train data from {config.train_path}...")
    df_train = pd.read_csv(config.train_path)
    
    # Column names
    action_col = 'action' if 'action' in df_train.columns else 'action_discrete'
    reward_col = 'reward' if 'reward' in df_train.columns else 'reward_terminal'
    done_col = 'done' if 'done' in df_train.columns else 'terminal'
    
    # State columns
    exclude_cols = ['stay_id', 'time_window', 'action', 'action_discrete',
                    'reward', 'reward_terminal', 'done', 'terminal',
                    'next_state', 'mortality_90d']
    state_cols = [c for c in df_train.columns if c not in exclude_cols and not c.startswith('next_')]
    
    # Action mapping from training data
    train_actions_original = df_train[action_col].values.astype(np.int32)
    unique_actions = sorted(set(train_actions_original))
    n_actions = len(unique_actions)
    
    action_to_idx = {a: i for i, a in enumerate(unique_actions)}
    idx_to_action = {i: a for a, i in action_to_idx.items()}
    
    logger.info(f"  Unique actions in training: {n_actions}")
    logger.info(f"  Original action IDs: {unique_actions}")
    
    # Load all splits
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
        
        # Handle NaN
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
    logger.info("DOUBLE DQN PIPELINE FOR SEPSIS TREATMENT")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now()}")
    
    # Configuration
    config = DoubleDQNConfig()
    
    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    datasets, n_actions, action_mapping = load_data(config)
    train_data = datasets['train']
    val_data = datasets['val']
    test_data = datasets['test']
    
    state_cols = train_data['state_cols']
    n_state_features = len(state_cols)
    
    logger.info(f"\nUsing {n_actions} actions, {n_state_features} state features")
    logger.info(f"Device: {config.device}")
    
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
    # STEP 2: Create and train Double DQN agent
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: TRAINING DOUBLE DQN")
    logger.info("=" * 60)
    
    agent = DoubleDQNAgent(
        n_state_features=n_state_features,
        n_actions=n_actions,
        config=config,
        behavior_policy=behavior_policy
    )
    
    # Fit scaler
    agent.fit_scaler(train_data['states'])
    
    # Train
    history = train_double_dqn(agent, train_data, val_data, config)
    
    # =========================================================================
    # STEP 3: Clinical sensitivity analysis
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: CLINICAL SENSITIVITY ANALYSIS")
    logger.info("=" * 60)
    
    clinical_results = clinical_sensitivity_analysis(agent, state_cols, action_mapping)
    
    # =========================================================================
    # STEP 4: Komorowski-style validation
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
    
    # Clinician policy (on-policy)
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
    clin_std = np.std(clin_returns)
    clin_se = clin_std / np.sqrt(len(clin_returns))
    
    logger.info(f"  Clinician: {clin_value:.3f} ± {1.96*clin_se:.3f}")
    
    # AI policy (WDR)
    logger.info("Evaluating AI policy (WDR)...")
    wdr = WDREstimator(gamma=config.gamma)
    ai_wdr_result = wdr.estimate(
        test_data['df'],
        test_data['states'],
        test_data['actions'],
        test_data['rewards'],
        test_data['dones'],
        agent,
        behavior_policy
    )
    
    logger.info(f"  AI (WDR): {ai_wdr_result['wdr_estimate']:.3f} ± {1.96*ai_wdr_result['wdr_se']:.3f}")
    logger.info(f"  ESS ratio: {ai_wdr_result['ess_ratio']:.4f}")
    
    # Agreement
    ai_actions = agent.select_actions(test_data['states'], use_constraint=True)
    agreement = np.mean(ai_actions == test_data['actions'])
    logger.info(f"  AI-Clinician agreement: {agreement*100:.1f}%")
    
    # =========================================================================
    # STEP 6: Bootstrap confidence intervals
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: BOOTSTRAP CONFIDENCE INTERVALS")
    logger.info("=" * 60)
    
    bootstrap_result = bootstrap_ci(
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
    
    logger.info(f"  AI Policy: {bootstrap_result['mean']:.3f} [{bootstrap_result['ci_lower']:.3f}, {bootstrap_result['ci_upper']:.3f}]")
    
    # =========================================================================
    # STEP 7: Statistical significance
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: STATISTICAL SIGNIFICANCE")
    logger.info("=" * 60)
    
    # Compute difference distribution
    # Bootstrap for clinician
    clin_bootstrap = []
    stay_ids = test_data['df']['stay_id'].unique()
    for _ in range(config.n_bootstrap):
        sampled_ids = np.random.choice(stay_ids, size=len(stay_ids), replace=True)
        sampled_returns = [clin_returns[np.where(stay_ids == sid)[0][0]] for sid in sampled_ids]
        clin_bootstrap.append(np.mean(sampled_returns))
    
    clin_bootstrap = np.array(clin_bootstrap)
    
    # Difference distribution
    diff_distribution = bootstrap_result['bootstrap_estimates'] - clin_bootstrap
    diff_mean = np.mean(diff_distribution)
    diff_ci_lower = np.percentile(diff_distribution, 2.5)
    diff_ci_upper = np.percentile(diff_distribution, 97.5)
    
    # P-value (one-sided: AI better than clinician)
    p_value = np.mean(diff_distribution <= 0)
    
    significant = diff_ci_lower > 0
    
    logger.info(f"  AI - Clinician: {diff_mean:.3f} [{diff_ci_lower:.3f}, {diff_ci_upper:.3f}]")
    logger.info(f"  P-value: {p_value:.4f}")
    logger.info(f"  Significant improvement: {'YES' if significant else 'NO'}")
    
    # =========================================================================
    # STEP 8: Final results
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"\n{'Policy':<25} {'Value':<12} {'95% CI':<25} {'ESS':<10}")
    logger.info("-" * 75)
    logger.info(f"{'Clinician':<25} {clin_value:<12.3f} [{clin_value-1.96*clin_se:.3f}, {clin_value+1.96*clin_se:.3f}]")
    logger.info(f"{'AI (Double DQN)':<25} {bootstrap_result['mean']:<12.3f} [{bootstrap_result['ci_lower']:.3f}, {bootstrap_result['ci_upper']:.3f}]{ai_wdr_result['ess_ratio']:>10.4f}")
    logger.info("-" * 75)
    logger.info(f"{'Improvement':<25} {diff_mean:<12.3f} [{diff_ci_lower:.3f}, {diff_ci_upper:.3f}]  p={p_value:.4f}")
    
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
    }, output_dir / 'double_dqn_model.pt')
    
    # Save results
    results = {
        'config': config.__dict__,
        'n_actions': n_actions,
        'n_state_features': n_state_features,
        'action_mapping': {str(k): int(v) for k, v in action_mapping['action_to_idx'].items()},
        'training_history': history,
        'clinical_validation': clinical_results,
        'komorowski_validation': {
            k: v for k, v in komorowski_results.items() 
            if k != 'mortality_by_agreement' and k != 'mortality_by_distance'
        },
        'evaluation': {
            'clinician': {
                'value': float(clin_value),
                'std': float(clin_std),
                'se': float(clin_se)
            },
            'ai_wdr': {
                'value': float(ai_wdr_result['wdr_estimate']),
                'std': float(ai_wdr_result['wdr_std']),
                'se': float(ai_wdr_result['wdr_se']),
                'ess_ratio': float(ai_wdr_result['ess_ratio'])
            },
            'ai_bootstrap': {
                'mean': float(bootstrap_result['mean']),
                'ci_lower': float(bootstrap_result['ci_lower']),
                'ci_upper': float(bootstrap_result['ci_upper'])
            },
            'difference': {
                'mean': float(diff_mean),
                'ci_lower': float(diff_ci_lower),
                'ci_upper': float(diff_ci_upper),
                'p_value': float(p_value),
                'significant': bool(significant)
            },
            'agreement': float(agreement)
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save full results with numpy arrays
    full_results = {
        'bootstrap_estimates': bootstrap_result['bootstrap_estimates'],
        'clin_bootstrap': clin_bootstrap,
        'diff_distribution': diff_distribution,
        'trajectory_values': ai_wdr_result['trajectory_values'],
        'komorowski_results': komorowski_results
    }
    
    with open(output_dir / 'full_results.pkl', 'wb') as f:
        pickle.dump(full_results, f)
    
    logger.info(f"  Results saved to {output_dir}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Finished at: {datetime.now()}")
    
    return results


if __name__ == "__main__":
    main()