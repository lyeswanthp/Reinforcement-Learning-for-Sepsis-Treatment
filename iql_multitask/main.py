"""Main execution: Train IQL agent with multi-task learning."""

import numpy as np
import json
import torch
import logging
from pathlib import Path
from datetime import datetime

from config import Config
from data_loader import DataLoader
from behavior_policy import BehaviorPolicy
from iql_agent import IQLAgent
from trainer import IQLTrainer
from evaluator import WDREstimator, ClinicalValidator


def setup_logging(output_dir: Path):
    """Setup logging."""
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def compute_clinician_baseline(df, rewards, gamma):
    """Compute clinician policy returns."""
    returns = []
    for stay_id in df['stay_id'].unique():
        mask = (df['stay_id'] == stay_id).values
        traj_rewards = rewards[mask]
        G = 0.0
        for t in range(len(traj_rewards) - 1, -1, -1):
            G = traj_rewards[t] + gamma * G
        returns.append(G)
    return np.array(returns)


def main():
    config = Config()
    output_dir = Path(config.data.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("IQL WITH MULTI-TASK LEARNING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Action space: {config.action.n_actions} (3x3)")
    logger.info(f"Temporal split: {config.data.temporal_split}")
    logger.info(f"Multi-task: BP + Lactate prediction")

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

    data_loader = DataLoader(config)
    datasets = data_loader.load_all_splits()

    train_data = datasets['train']
    val_data = datasets['val']
    test_data = datasets['test']

    n_states = train_data['states'].shape[1]
    n_actions = config.action.n_actions

    logger.info(f"\nState features: {n_states}")
    logger.info(f"Actions: {n_actions}")

    logger.info("\n" + "=" * 60)
    logger.info("FITTING BEHAVIOR POLICY")
    logger.info("=" * 60)

    behavior_policy = BehaviorPolicy(
        n_actions=n_actions,
        softening=config.ope.behavior_softening,
        random_seed=config.random_seed
    )
    behavior_policy.fit(train_data['states'], train_data['actions'])

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING IQL AGENT")
    logger.info("=" * 80)

    agent = IQLAgent(n_states, n_actions, config)
    agent.fit_scaler(train_data['states'])

    trainer = IQLTrainer(agent, config)
    history = trainer.train(train_data, val_data)

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 80)

    clin_returns = compute_clinician_baseline(
        test_data['df'], test_data['rewards'], config.iql.gamma
    )
    clin_value = np.mean(clin_returns)
    clin_std = np.std(clin_returns)
    clin_se = clin_std / np.sqrt(len(clin_returns))

    logger.info(f"\nClinician policy: {clin_value:.3f} ± {1.96*clin_se:.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("IQL EVALUATION")
    logger.info("=" * 60)

    validator = ClinicalValidator(
        agent, data_loader.state_columns, data_loader.action_simplifier
    )
    clinical = validator.validate()

    wdr = WDREstimator(config.iql.gamma, config.ope.max_importance_weight)
    iql_wdr = wdr.estimate(
        test_data['df'], test_data['states'], test_data['actions'],
        test_data['rewards'], test_data['dones'],
        agent, behavior_policy
    )

    iql_actions = agent.select_actions(test_data['states'])
    iql_agreement = np.mean(iql_actions == test_data['actions'])

    logger.info(f"\nIQL Agent:")
    logger.info(f"  WDR: {iql_wdr['wdr_estimate']:.3f} ± {1.96*iql_wdr['wdr_se']:.3f}")
    logger.info(f"  Agreement: {iql_agreement*100:.1f}%")
    logger.info(f"  Clinical: {clinical['overall']['passed']}/3")

    action_counts = np.bincount(iql_actions, minlength=n_actions)
    action_probs = action_counts / action_counts.sum()
    entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
    logger.info(f"  Action diversity: {len(np.unique(iql_actions))}/{n_actions}, H={entropy:.2f}")

    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)

    logger.info(f"\n{'Model':<25} {'WDR Value':<15} {'Agreement':<12} {'Clinical'}")
    logger.info("-" * 70)
    logger.info(f"{'Clinician':<25} {clin_value:<15.3f} {'100.0%':<12} {'N/A'}")
    logger.info(f"{'IQL':<25} {iql_wdr['wdr_estimate']:<15.3f} {iql_agreement*100:<12.1f}% {clinical['overall']['passed']}/3")

    results = {
        'config': {
            'n_actions': n_actions,
            'n_states': n_states,
            'action_space': '3x3',
            'temporal_split': config.data.temporal_split,
            'expectile': config.iql.expectile,
            'entropy_weight': config.iql.entropy_weight,
            'bp_weight': config.iql.bp_weight,
            'lactate_weight': config.iql.lactate_weight
        },
        'clinician': {
            'value': float(clin_value),
            'std': float(clin_std)
        },
        'iql': {
            'wdr': float(iql_wdr['wdr_estimate']),
            'wdr_se': float(iql_wdr['wdr_se']),
            'agreement': float(iql_agreement),
            'clinical_passed': clinical['overall']['passed'],
            'action_diversity': int(len(np.unique(iql_actions))),
            'entropy': float(entropy)
        },
        'training_history': {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    torch.save({
        'q_network': agent.q_network.state_dict(),
        'v_network': agent.v_network.state_dict(),
        'target_network': agent.target_network.state_dict(),
        'scaler_mean': agent.scaler.mean_,
        'scaler_scale': agent.scaler.scale_
    }, output_dir / 'model.pt')

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Finished at: {datetime.now()}")

    return results


if __name__ == "__main__":
    main()
