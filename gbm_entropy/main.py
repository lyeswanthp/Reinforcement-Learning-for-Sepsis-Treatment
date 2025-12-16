"""Main execution: Train GBM ensemble with entropy regularization."""

import numpy as np
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

from config import Config
from data_loader import DataLoader
from behavior_policy import BehaviorPolicy
from gbm_agent import EnsembleGBMAgent
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


def evaluate_temperature_sweep(
    agent, test_data, behavior_policy, wdr_estimator, config, logger
):
    """Evaluate multiple temperatures to find best diversity-performance tradeoff."""
    logger.info("\n" + "=" * 60)
    logger.info("TEMPERATURE SWEEP")
    logger.info("=" * 60)

    temperatures = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results = []

    for temp in temperatures:
        actions = agent.select_actions(test_data['states'][:10000], temperature=temp)
        n_unique = len(np.unique(actions))
        counts = np.bincount(actions, minlength=config.action.n_actions)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        agreement = np.mean(actions == test_data['actions'][:10000])

        wdr_result = wdr_estimator.estimate(
            test_data['df'], test_data['states'], test_data['actions'],
            test_data['rewards'], test_data['dones'],
            agent, behavior_policy, temperature=temp
        )

        results.append({
            'temperature': temp,
            'wdr': wdr_result['wdr_estimate'],
            'n_unique': n_unique,
            'entropy': entropy,
            'agreement': agreement
        })

        logger.info(
            f"  T={temp:.1f}: WDR={wdr_result['wdr_estimate']:.3f}, "
            f"Actions={n_unique}/9, H={entropy:.2f}, Agree={agreement*100:.1f}%"
        )

    return results


def main():
    config = Config()
    output_dir = Path(config.data.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("GBM ENSEMBLE WITH ENTROPY REGULARIZATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Action space: {config.action.n_actions} (3x3)")
    logger.info(f"Ensemble size: {config.gbm.ensemble_size}")
    logger.info(f"Temperature: {config.entropy.temperature}")

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
    logger.info("TRAINING GBM ENSEMBLE")
    logger.info("=" * 80)

    agent = EnsembleGBMAgent(n_states, n_actions, config)
    agent.fit_scalers(train_data['states'])
    agent.train(train_data, val_data)

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 80)

    clin_returns = compute_clinician_baseline(
        test_data['df'], test_data['rewards'], config.gbm.gamma
    )
    clin_value = np.mean(clin_returns)
    clin_std = np.std(clin_returns)
    clin_se = clin_std / np.sqrt(len(clin_returns))

    logger.info(f"\nClinician policy: {clin_value:.3f} ± {1.96*clin_se:.3f}")

    wdr_estimator = WDREstimator(config.gbm.gamma, config.ope.max_importance_weight)

    temp_sweep = evaluate_temperature_sweep(
        agent, test_data, behavior_policy, wdr_estimator, config, logger
    )

    best_temp = max(temp_sweep, key=lambda x: x['wdr'])['temperature']
    logger.info(f"\nBest temperature: {best_temp}")

    logger.info("\n" + "=" * 60)
    logger.info("GBM EVALUATION (WITH ENTROPY)")
    logger.info("=" * 60)

    validator = ClinicalValidator(
        agent, data_loader.state_columns,
        data_loader.action_simplifier, config.entropy
    )
    clinical = validator.validate()

    gbm_wdr = wdr_estimator.estimate(
        test_data['df'], test_data['states'], test_data['actions'],
        test_data['rewards'], test_data['dones'],
        agent, behavior_policy, temperature=config.entropy.temperature
    )

    gbm_actions = agent.select_actions(test_data['states'], temperature=config.entropy.temperature)
    gbm_agreement = np.mean(gbm_actions == test_data['actions'])

    logger.info(f"\nGBM Ensemble:")
    logger.info(f"  WDR: {gbm_wdr['wdr_estimate']:.3f} ± {1.96*gbm_wdr['wdr_se']:.3f}")
    logger.info(f"  Agreement: {gbm_agreement*100:.1f}%")
    logger.info(f"  Clinical: {clinical['overall']['passed']}/3")

    action_counts = np.bincount(gbm_actions, minlength=n_actions)
    action_probs = action_counts / action_counts.sum()
    entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
    logger.info(f"  Action diversity: {len(np.unique(gbm_actions))}/{n_actions}, H={entropy:.2f}")

    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)

    logger.info(f"\n{'Model':<25} {'WDR Value':<15} {'Agreement':<12} {'Clinical'}")
    logger.info("-" * 70)
    logger.info(f"{'Clinician':<25} {clin_value:<15.3f} {'100.0%':<12} {'N/A'}")
    logger.info(f"{'GBM Ensemble':<25} {gbm_wdr['wdr_estimate']:<15.3f} {gbm_agreement*100:<12.1f}% {clinical['overall']['passed']}/3")

    results = {
        'config': {
            'n_actions': n_actions,
            'n_states': n_states,
            'action_space': '3x3',
            'ensemble_size': config.gbm.ensemble_size,
            'n_estimators': config.gbm.n_estimators,
            'temperature': config.entropy.temperature,
            'q_iterations': config.gbm.n_iterations
        },
        'clinician': {
            'value': float(clin_value),
            'std': float(clin_std)
        },
        'gbm': {
            'wdr': float(gbm_wdr['wdr_estimate']),
            'wdr_se': float(gbm_wdr['wdr_se']),
            'agreement': float(gbm_agreement),
            'clinical_passed': clinical['overall']['passed'],
            'action_diversity': int(len(np.unique(gbm_actions))),
            'entropy': float(entropy)
        },
        'temperature_sweep': temp_sweep,
        'best_temperature': float(best_temp)
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_dir / 'models.pkl', 'wb') as f:
        pickle.dump({
            'agents': agent.agents,
            'scaler_mean': agent.agents[0].scaler.mean_,
            'scaler_scale': agent.agents[0].scaler.scale_
        }, f)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Finished at: {datetime.now()}")

    return results


if __name__ == "__main__":
    main()
