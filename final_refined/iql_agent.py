"""IQL agent with strengthened multi-task learning."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class MultiTaskNetwork(nn.Module):
    """Multi-task network with Q, V, BP, and Lactate heads."""

    def __init__(self, n_states: int, n_actions: int, hidden_dims: list, dropout_rate: float, use_layer_norm: bool):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        layers = []
        prev_dim = n_states
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)
        self.q_head = nn.Linear(prev_dim, n_actions)
        self.v_head = nn.Linear(prev_dim, 1)
        self.bp_head = nn.Linear(prev_dim, 1)
        self.lactate_head = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(states)
        q_values = self.q_head(features)
        v_value = self.v_head(features)
        bp_pred = self.bp_head(features)
        lactate_pred = self.lactate_head(features)
        return q_values, v_value.squeeze(-1), bp_pred.squeeze(-1), lactate_pred.squeeze(-1)


class IQLAgent:
    """IQL agent with strong auxiliary tasks."""

    def __init__(self, n_states: int, n_actions: int, config, device: str = 'cuda'):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.q_network = MultiTaskNetwork(
            n_states, n_actions,
            config.iql.hidden_dims,
            config.iql.dropout_rate,
            config.iql.use_layer_norm
        ).to(self.device)

        self.v_network = MultiTaskNetwork(
            n_states, n_actions,
            config.iql.hidden_dims,
            config.iql.dropout_rate,
            config.iql.use_layer_norm
        ).to(self.device)

        self.target_network = MultiTaskNetwork(
            n_states, n_actions,
            config.iql.hidden_dims,
            config.iql.dropout_rate,
            config.iql.use_layer_norm
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.AdamW(
            list(self.q_network.parameters()) + list(self.v_network.parameters()),
            lr=config.iql.learning_rate,
            weight_decay=config.iql.weight_decay
        )

        self.scaler = StandardScaler()
        self.scaler_fitted = False

        n_params = sum(p.numel() for p in self.q_network.parameters())
        logger.info(f"IQL initialized: {n_params:,} parameters")

    def fit_scaler(self, states: np.ndarray):
        """Fit state scaler."""
        self.scaler.fit(states)
        self.scaler_fitted = True

    def scale_states(self, states: np.ndarray) -> np.ndarray:
        """Scale states."""
        if not self.scaler_fitted:
            raise RuntimeError("Scaler not fitted")
        return self.scaler.transform(states)

    def expectile_loss(self, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        """Asymmetric L2 loss for expectile regression."""
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        next_bp: torch.Tensor,
        next_lactate: torch.Tensor
    ) -> dict:
        """Single update step."""
        self.q_network.train()
        self.v_network.train()

        q_pred, _, bp_pred, lactate_pred = self.q_network(states)
        q_pred_action = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_target, _, _, _ = self.target_network(next_states)
            q_target_max = q_target.max(dim=1)[0]
            td_target = rewards + self.config.iql.gamma * q_target_max * (1 - dones.float())

        td_diff = td_target - q_pred_action
        q_loss = self.expectile_loss(td_diff, self.config.iql.expectile).mean()

        _, v_pred, _, _ = self.v_network(states)
        with torch.no_grad():
            q_all, _, _, _ = self.q_network(states)
        v_diff = q_all.max(dim=1)[0] - v_pred
        v_loss = self.expectile_loss(v_diff, self.config.iql.expectile).mean()

        bp_loss = F.mse_loss(bp_pred, next_bp)
        lactate_loss = F.mse_loss(lactate_pred, next_lactate)

        action_probs = F.softmax(q_pred / self.config.iql.temperature, dim=1)
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()

        total_loss = (
            q_loss + v_loss +
            self.config.iql.bp_weight * bp_loss +
            self.config.iql.lactate_weight * lactate_loss -
            self.config.iql.entropy_weight * entropy
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.v_network.parameters()),
            self.config.iql.grad_clip
        )
        self.optimizer.step()

        self._update_target()

        return {
            'total_loss': total_loss.item(),
            'q_loss': q_loss.item(),
            'v_loss': v_loss.item(),
            'bp_loss': bp_loss.item(),
            'lactate_loss': lactate_loss.item(),
            'entropy': entropy.item()
        }

    def _update_target(self, tau: float = 0.005):
        """Soft update target network."""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_actions(self, states: np.ndarray, temperature: float = 0.0) -> np.ndarray:
        """Select actions."""
        states_scaled = self.scale_states(states)
        states_tensor = torch.FloatTensor(states_scaled).to(self.device)

        self.q_network.eval()
        with torch.no_grad():
            q_values, _, _, _ = self.q_network(states_tensor)
            q_values = q_values.cpu().numpy()

        if temperature > 0:
            probs = np.exp(q_values / temperature)
            probs = probs / probs.sum(axis=1, keepdims=True)
            actions = np.array([np.random.choice(self.n_actions, p=p) for p in probs])
        else:
            actions = np.argmax(q_values, axis=1)

        return actions

    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        states_scaled = self.scale_states(states)
        states_tensor = torch.FloatTensor(states_scaled).to(self.device)

        self.q_network.eval()
        with torch.no_grad():
            q_values, _, _, _ = self.q_network(states_tensor)
            return q_values.cpu().numpy()
