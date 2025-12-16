"""
Main execution: Train linear and GBM agents, compare performance.
"""
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

from config import Config
from data_loader import DataLoader
from behavior_policy import BehaviorPolicy
from linear_q_learning import LinearQAgent
from gbm_q_learning import GBMQAgent
from evaluator import WDREstimator, ClinicalValidator


def setup_logging(output_dir: Path):
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
    logger.info("="*80)
    logger.info("HYBRID LINEAR RL PIPELINE (OPTION 5)")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Action space: {config.action.n_actions} (3x3)")
    logger.info(f"Models: Linear Q-learning + Gradient Boosting Q-learning")

    np.random.seed(config.random_seed)

    data_loader = DataLoader(config)
    datasets = data_loader.load_all_splits()

    train_data = datasets['train']
    val_data = datasets['val']
    test_data = datasets['test']

    n_states = train_data['states'].shape[1]
    n_actions = config.action.n_actions

    logger.info(f"\nState features: {n_states}")
    logger.info(f"Actions: {n_actions}")

    logger.info("\n" + "="*60)
    logger.info("FITTING BEHAVIOR POLICY")
    logger.info("="*60)

    behavior_policy = BehaviorPolicy(
        n_actions=n_actions,
        softening=config.ope.behavior_softening,
        random_seed=config.random_seed
    )
    behavior_policy.fit(train_data['states'], train_data['actions'])

    logger.info("\n" + "="*80)
    logger.info("MODEL 1: LINEAR Q-LEARNING")
    logger.info("="*80)

    linear_agent = LinearQAgent(n_states, n_actions, config)
    linear_agent.fit_scaler(train_data['states'])
    linear_history = linear_agent.train(train_data, val_data)

    logger.info("\n" + "="*80)
    logger.info("MODEL 2: GRADIENT BOOSTING Q-LEARNING")
    logger.info("="*80)

    gbm_agent = GBMQAgent(n_states, n_actions, config)
    gbm_agent.fit_scaler(train_data['states'])
    gbm_history = gbm_agent.train(train_data, val_data)

    logger.info("\n" + "="*80)
    logger.info("EVALUATION ON TEST SET")
    logger.info("="*80)

    clin_returns = compute_clinician_baseline(
        test_data['df'], test_data['rewards'], config.linear.gamma
    )
    clin_value = np.mean(clin_returns)
    clin_std = np.std(clin_returns)
    clin_se = clin_std / np.sqrt(len(clin_returns))

    logger.info(f"\nClinician policy: {clin_value:.3f} ± {1.96*clin_se:.3f}")

    wdr = WDREstimator(config.linear.gamma, config.ope.max_importance_weight)

    logger.info("\n" + "="*60)
    logger.info("LINEAR Q-LEARNING EVALUATION")
    logger.info("="*60)

    linear_validator = ClinicalValidator(
        linear_agent, data_loader.state_columns, data_loader.action_simplifier
    )
    linear_clinical = linear_validator.validate()

    linear_wdr = wdr.estimate(
        test_data['df'], test_data['states'], test_data['actions'],
        test_data['rewards'], test_data['dones'],
        linear_agent, behavior_policy
    )

    linear_actions = linear_agent.select_actions(test_data['states'])
    linear_agreement = np.mean(linear_actions == test_data['actions'])

    logger.info(f"\nLinear Q-learning:")
    logger.info(f"  WDR: {linear_wdr['wdr_estimate']:.3f} ± {1.96*linear_wdr['wdr_se']:.3f}")
    logger.info(f"  Agreement: {linear_agreement*100:.1f}%")
    logger.info(f"  Clinical: {linear_clinical['overall']['passed']}/3")

    logger.info("\n" + "="*60)
    logger.info("GRADIENT BOOSTING EVALUATION")
    logger.info("="*60)

    gbm_validator = ClinicalValidator(
        gbm_agent, data_loader.state_columns, data_loader.action_simplifier
    )
    gbm_clinical = gbm_validator.validate()

    gbm_wdr = wdr.estimate(
        test_data['df'], test_data['states'], test_data['actions'],
        test_data['rewards'], test_data['dones'],
        gbm_agent, behavior_policy
    )

    gbm_actions = gbm_agent.select_actions(test_data['states'])
    gbm_agreement = np.mean(gbm_actions == test_data['actions'])

    logger.info(f"\nGradient Boosting:")
    logger.info(f"  WDR: {gbm_wdr['wdr_estimate']:.3f} ± {1.96*gbm_wdr['wdr_se']:.3f}")
    logger.info(f"  Agreement: {gbm_agreement*100:.1f}%")
    logger.info(f"  Clinical: {gbm_clinical['overall']['passed']}/3")

    logger.info("\n" + "="*80)
    logger.info("FINAL COMPARISON")
    logger.info("="*80)

    logger.info(f"\n{'Model':<25} {'WDR Value':<15} {'Agreement':<12} {'Clinical'}")
    logger.info("-"*70)
    logger.info(f"{'Clinician':<25} {clin_value:<15.3f} {'100.0%':<12} {'N/A'}")
    logger.info(f"{'Linear Q-learning':<25} {linear_wdr['wdr_estimate']:<15.3f} {linear_agreement*100:<12.1f}% {linear_clinical['overall']['passed']}/3")
    logger.info(f"{'Gradient Boosting':<25} {gbm_wdr['wdr_estimate']:<15.3f} {gbm_agreement*100:<12.1f}% {gbm_clinical['overall']['passed']}/3")

    best_model = 'linear' if linear_wdr['wdr_estimate'] > gbm_wdr['wdr_estimate'] else 'gbm'
    logger.info(f"\nBest model: {best_model.upper()}")

    results = {
        'config': {
            'n_actions': n_actions,
            'n_states': n_states,
            'action_space': '3x3'
        },
        'clinician': {
            'value': float(clin_value),
            'std': float(clin_std)
        },
        'linear': {
            'wdr': float(linear_wdr['wdr_estimate']),
            'wdr_se': float(linear_wdr['wdr_se']),
            'agreement': float(linear_agreement),
            'clinical_passed': linear_clinical['overall']['passed']
        },
        'gbm': {
            'wdr': float(gbm_wdr['wdr_estimate']),
            'wdr_se': float(gbm_wdr['wdr_se']),
            'agreement': float(gbm_agreement),
            'clinical_passed': gbm_clinical['overall']['passed']
        },
        'best_model': best_model
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_dir / 'models.pkl', 'wb') as f:
        pickle.dump({
            'linear_weights': linear_agent.weights,
            'linear_scaler_mean': linear_agent.scaler.mean_,
            'linear_scaler_scale': linear_agent.scaler.scale_,
            'gbm_models': gbm_agent.models
        }, f)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Finished at: {datetime.now()}")

    return results


if __name__ == "__main__":
    main()
