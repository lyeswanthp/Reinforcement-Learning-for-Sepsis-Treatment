"""
Training loop for Double DQN.
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from agent import DoubleDQNAgent
from config import Config
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class Trainer:
    """Trains Double DQN agent."""

    def __init__(self, agent: DoubleDQNAgent, config: Config):
        self.agent = agent
        self.config = config
        self.device = torch.device(config.model.device)

    def train(self, train_data: Dict, val_data: Dict) -> Dict:
        logger.info("="*60)
        logger.info("TRAINING DOUBLE DQN")
        logger.info("="*60)
        logger.info(f"Device: {self.config.model.device}")
        logger.info(f"Epochs: {self.config.model.n_epochs}")
        logger.info(f"Batch size: {self.config.model.batch_size}")

        train_states = self.agent.scale_states(train_data['states'])
        train_next_states = self.agent.scale_states(train_data['next_states'])

        dataset = TensorDataset(
            torch.FloatTensor(train_states),
            torch.LongTensor(train_data['actions']),
            torch.FloatTensor(train_data['rewards']),
            torch.FloatTensor(train_next_states),
            torch.FloatTensor(train_data['dones'].astype(float))
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.model.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.config.model.device == 'cuda'
        )

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_q_mean': [],
            'n_unique_actions': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None

        for epoch in range(self.config.model.n_epochs):
            epoch_losses = []

            for batch in dataloader:
                states, actions, rewards, next_states, dones = [b.to(self.device) for b in batch]
                loss = self.agent.update(states, actions, rewards, next_states, dones)
                epoch_losses.append(loss)

            train_loss = np.mean(epoch_losses)
            history['train_loss'].append(train_loss)

            val_loss, val_q_mean, n_unique = self._validate(val_data)
            history['val_loss'].append(val_loss)
            history['val_q_mean'].append(val_q_mean)
            history['n_unique_actions'].append(n_unique)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}: "
                    f"Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, "
                    f"Val Q={val_q_mean:.2f}, "
                    f"Actions={n_unique}"
                )

            if val_loss < best_val_loss - self.config.model.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_state_dict = self.agent.q_network.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= self.config.model.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if best_state_dict is not None:
            self.agent.q_network.load_state_dict(best_state_dict)
            self.agent.target_network.load_state_dict(best_state_dict)

        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return history

    def _validate(self, val_data: Dict) -> tuple:
        val_states = self.agent.scale_states(val_data['states'])
        val_next_states = self.agent.scale_states(val_data['next_states'])

        val_states_t = torch.FloatTensor(val_states).to(self.device)
        val_actions_t = torch.LongTensor(val_data['actions']).to(self.device)
        val_rewards_t = torch.FloatTensor(val_data['rewards']).to(self.device)
        val_next_t = torch.FloatTensor(val_next_states).to(self.device)
        val_dones_t = torch.FloatTensor(val_data['dones'].astype(float)).to(self.device)

        self.agent.q_network.eval()
        with torch.no_grad():
            val_q = self.agent.q_network.get_q_values(val_states_t, val_actions_t)
            val_targets = self.agent.compute_targets(val_rewards_t, val_next_t, val_dones_t)
            val_loss = F.smooth_l1_loss(val_q, val_targets).item()

            val_q_all = self.agent.q_network(val_states_t[:10000])
            val_q_mean = val_q_all.mean().item()

        val_actions = self.agent.select_actions(val_data['states'][:10000])
        n_unique = len(np.unique(val_actions))

        return val_loss, val_q_mean, n_unique
