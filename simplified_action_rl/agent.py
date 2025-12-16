"""
Double DQN agent implementation.
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from models import QNetwork, BehaviorPolicy
from config import Config
import logging

logger = logging.getLogger(__name__)


class DoubleDQNAgent:
    """Double DQN with target network."""

    def __init__(self, n_states: int, n_actions: int, config: Config):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.device = torch.device(config.model.device)

        self.q_network = QNetwork(
            n_states, n_actions,
            config.model.hidden_dims,
            config.model.dropout_rate
        ).to(self.device)

        self.target_network = QNetwork(
            n_states, n_actions,
            config.model.hidden_dims,
            config.model.dropout_rate
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.model.learning_rate
        )

        self.scaler = StandardScaler()
        self.scaler_fitted = False
        self.update_count = 0

    def fit_scaler(self, states: np.ndarray):
        self.scaler.fit(states)
        self.scaler_fitted = True
        logger.info("State scaler fitted")

    def scale_states(self, states: np.ndarray) -> np.ndarray:
        if not self.scaler_fitted:
            raise RuntimeError("Scaler not fitted")
        return self.scaler.transform(states)

    def select_actions(self, states: np.ndarray) -> np.ndarray:
        states_scaled = self.scale_states(states)
        states_tensor = torch.FloatTensor(states_scaled).to(self.device)

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(states_tensor).cpu().numpy()

        actions = np.argmax(q_values, axis=1)
        return actions

    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        states_scaled = self.scale_states(states)
        states_tensor = torch.FloatTensor(states_scaled).to(self.device)

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(states_tensor).cpu().numpy()

        return q_values

    def compute_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            best_actions = next_q_online.argmax(dim=1)

            next_q_target = self.target_network(next_states)
            next_q_values = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            targets = rewards + self.config.model.gamma * next_q_values * (1 - dones.float())

        return targets

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        self.q_network.train()

        current_q = self.q_network.get_q_values(states, actions)
        targets = self.compute_targets(rewards, next_states, dones)

        loss = F.smooth_l1_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.config.model.gradient_clip
        )
        self.optimizer.step()

        self.update_count += 1
        if self.config.model.use_soft_update:
            self._soft_update_target()
        elif self.update_count % self.config.model.target_update_freq == 0:
            self._hard_update_target()

        return loss.item()

    def _soft_update_target(self):
        tau = self.config.model.soft_update_tau
        for target_p, online_p in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_p.data.copy_(tau * online_p.data + (1 - tau) * target_p.data)

    def _hard_update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
