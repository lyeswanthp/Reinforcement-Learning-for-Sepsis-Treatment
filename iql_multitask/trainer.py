"""Training loop for IQL agent."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)


class IQLTrainer:
    """Trainer for IQL agent."""

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config

    def train(self, train_data: dict, val_data: dict) -> dict:
        """Train IQL agent."""
        logger.info("=" * 60)
        logger.info("TRAINING IQL")
        logger.info("=" * 60)
        logger.info(f"Learning rate: {self.config.iql.learning_rate}")
        logger.info(f"Expectile: {self.config.iql.expectile}")
        logger.info(f"Entropy weight: {self.config.iql.entropy_weight}")
        logger.info(f"BP weight: {self.config.iql.bp_weight}")
        logger.info(f"Lactate weight: {self.config.iql.lactate_weight}")

        train_states = self.agent.scale_states(train_data['states'])
        train_actions = train_data['actions']
        train_rewards = train_data['rewards']
        train_next_states = self.agent.scale_states(train_data['next_states'])
        train_dones = train_data['dones']
        train_next_bp = train_data['next_bp']
        train_next_lactate = train_data['next_lactate']

        dataset = TensorDataset(
            torch.FloatTensor(train_states),
            torch.LongTensor(train_actions),
            torch.FloatTensor(train_rewards),
            torch.FloatTensor(train_next_states),
            torch.FloatTensor(train_dones.astype(float)),
            torch.FloatTensor(train_next_bp),
            torch.FloatTensor(train_next_lactate)
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.iql.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.config.device == 'cuda')
        )

        history = {
            'train_loss': [],
            'val_loss': [],
            'q_mean': [],
            'q_std': [],
            'n_unique_actions': [],
            'entropy': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.iql.n_epochs):
            epoch_losses = []

            for batch in dataloader:
                states, actions, rewards, next_states, dones, next_bp, next_lactate = [
                    b.to(self.agent.device) for b in batch
                ]

                losses = self.agent.update(
                    states, actions, rewards, next_states, dones, next_bp, next_lactate
                )
                epoch_losses.append(losses['total_loss'])

            train_loss = np.mean(epoch_losses)
            history['train_loss'].append(train_loss)

            val_loss, val_stats = self._validate(val_data)
            history['val_loss'].append(val_loss)
            history['q_mean'].append(val_stats['q_mean'])
            history['q_std'].append(val_stats['q_std'])
            history['n_unique_actions'].append(val_stats['n_unique_actions'])
            history['entropy'].append(val_stats['entropy'])

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}: "
                    f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
                    f"Q={val_stats['q_mean']:.2f}±{val_stats['q_std']:.2f}, "
                    f"Actions={val_stats['n_unique_actions']}, H={val_stats['entropy']:.2f}"
                )

            if val_loss < best_val_loss - self.config.iql.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.iql.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return history

    def _validate(self, val_data: dict) -> tuple:
        """Validate agent."""
        val_states_scaled = self.agent.scale_states(val_data['states'])
        val_states_tensor = torch.FloatTensor(val_states_scaled).to(self.agent.device)
        val_actions_tensor = torch.LongTensor(val_data['actions']).to(self.agent.device)
        val_rewards_tensor = torch.FloatTensor(val_data['rewards']).to(self.agent.device)
        val_next_states_tensor = torch.FloatTensor(
            self.agent.scale_states(val_data['next_states'])
        ).to(self.agent.device)
        val_dones_tensor = torch.FloatTensor(val_data['dones'].astype(float)).to(self.agent.device)

        self.agent.q_network.eval()
        with torch.no_grad():
            q_pred, _, _, _ = self.agent.q_network(val_states_tensor)
            q_pred_actions = q_pred.gather(1, val_actions_tensor.unsqueeze(1)).squeeze(1)

            q_target, _, _, _ = self.agent.target_network(val_next_states_tensor)
            q_target_max = q_target.max(dim=1)[0]
            td_target = val_rewards_tensor + self.config.iql.gamma * q_target_max * (1 - val_dones_tensor)

            val_loss = torch.nn.functional.mse_loss(q_pred_actions, td_target).item()

        val_actions_pred = self.agent.select_actions(val_data['states'][:10000])
        n_unique = len(np.unique(val_actions_pred))
        action_counts = np.bincount(val_actions_pred, minlength=self.agent.n_actions)
        action_probs = action_counts / action_counts.sum()
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

        return val_loss, {
            'q_mean': q_pred.mean().item(),
            'q_std': q_pred.std().item(),
            'n_unique_actions': n_unique,
            'entropy': entropy
        }
