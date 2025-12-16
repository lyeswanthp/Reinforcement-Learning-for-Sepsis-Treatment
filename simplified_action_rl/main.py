"""
Main execution pipeline for simplified action space RL.
"""
import numpy as np
import torch
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

from config import Config
from data_loader import DataLoader
from models import BehaviorPolicy
from agent import DoubleDQNAgent
from trainer import Trainer
from evaluator import WDREstimator, ClinicalValidator, KomorowskiValidator


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
    """Compute on-policy clinician returns."""
    returns = []
    for stay_id in df['stay_id'].unique():
        mask = (df['stay_id'] == stay_id).values
        traj_rewards = rewards[mask]
        G = 0.0
        for t in range(len(traj_rewards) - 1, -1, -1):
            G = traj_rewards[t] + gamma * G
        returns.append(G)
    return np.array(returns)


def bootstrap_confidence_interval(
    df, states, actions, rewards, dones,
    agent, behavior_policy, n_bootstrap, gamma
):
    """Bootstrap confidence intervals for WDR estimate."""
    logger = logging.getLogger(__name__)
    logger.info(f"Computing bootstrap CI ({n_bootstrap} samples)...")

    wdr = WDREstimator(gamma, max_weight=100.0)
    stay_ids = df['stay_id'].unique()
    estimates = []

    for b in range(n_bootstrap):
        if (b + 1) % 100 == 0:
            logger.info(f"  Bootstrap {b+1}/{n_bootstrap}")

        sampled_ids = np.random.choice(stay_ids, size=len(stay_ids), replace=True)
        indices = []
        for sid in sampled_ids:
            idx = np.where(df['stay_id'] == sid)[0]
            indices.extend(idx.tolist())

        boot_df = df.iloc[indices].copy()
        boot_df['stay_id'] = np.repeat(
            np.arange(len(sampled_ids)),
            [np.sum(df['stay_id'] == sid) for sid in sampled_ids]
        )

        result = wdr.estimate(
            boot_df, states[indices], actions[indices],
            rewards[indices], dones[indices],
            agent, behavior_policy
        )
        estimates.append(result['wdr_estimate'])

    estimates = np.array(estimates)
    return {
        'mean': np.mean(estimates),
        'ci_lower': np.percentile(estimates, 2.5),
        'ci_upper': np.percentile(estimates, 97.5),
        'estimates': estimates
    }


def main():
    config = Config()
    output_dir = Path(config.data.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("="*80)
    logger.info("SIMPLIFIED ACTION SPACE RL PIPELINE")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Action space: {config.action.n_actions} (3x3)")
    logger.info(f"Device: {config.model.device}")

    np.random.seed(config.model.random_seed)
    torch.manual_seed(config.model.random_seed)

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
        random_seed=config.model.random_seed
    )
    behavior_policy.fit(train_data['states'], train_data['actions'])

    logger.info("\n" + "="*60)
    logger.info("CREATING AGENT")
    logger.info("="*60)

    agent = DoubleDQNAgent(n_states, n_actions, config)
    agent.fit_scaler(train_data['states'])

    trainer = Trainer(agent, config)
    history = trainer.train(train_data, val_data)

    clinical_validator = ClinicalValidator(
        agent, data_loader.state_columns, data_loader.action_simplifier
    )
    clinical_results = clinical_validator.validate()

    komorowski_validator = KomorowskiValidator(
        agent, data_loader.action_simplifier
    )
    komorowski_results = komorowski_validator.validate(
        test_data['df'],
        test_data['states'],
        test_data['actions'],
        test_data['rewards']
    )

    logger.info("\n" + "="*60)
    logger.info("OFF-POLICY EVALUATION")
    logger.info("="*60)

    clin_returns = compute_clinician_baseline(
        test_data['df'], test_data['rewards'], config.model.gamma
    )
    clin_value = np.mean(clin_returns)
    clin_std = np.std(clin_returns)
    clin_se = clin_std / np.sqrt(len(clin_returns))

    logger.info(f"Clinician policy: {clin_value:.3f} ± {1.96*clin_se:.3f}")

    wdr = WDREstimator(config.model.gamma, config.ope.max_importance_weight)
    ai_wdr_result = wdr.estimate(
        test_data['df'],
        test_data['states'],
        test_data['actions'],
        test_data['rewards'],
        test_data['dones'],
        agent,
        behavior_policy
    )

    logger.info(f"AI policy (WDR): {ai_wdr_result['wdr_estimate']:.3f} ± {1.96*ai_wdr_result['wdr_se']:.3f}")
    logger.info(f"ESS ratio: {ai_wdr_result['ess_ratio']:.4f}")

    ai_actions = agent.select_actions(test_data['states'])
    agreement = np.mean(ai_actions == test_data['actions'])
    logger.info(f"AI-Clinician agreement: {agreement*100:.1f}%")

    bootstrap_result = bootstrap_confidence_interval(
        test_data['df'],
        test_data['states'],
        test_data['actions'],
        test_data['rewards'],
        test_data['dones'],
        agent,
        behavior_policy,
        n_bootstrap=config.ope.n_bootstrap,
        gamma=config.model.gamma
    )

    logger.info(f"\nBootstrap CI: {bootstrap_result['mean']:.3f} [{bootstrap_result['ci_lower']:.3f}, {bootstrap_result['ci_upper']:.3f}]")

    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"\n{'Policy':<20} {'Value':<12} {'95% CI':<25}")
    logger.info("-"*60)
    logger.info(f"{'Clinician':<20} {clin_value:<12.3f} [{clin_value-1.96*clin_se:.3f}, {clin_value+1.96*clin_se:.3f}]")
    logger.info(f"{'AI (Double DQN)':<20} {bootstrap_result['mean']:<12.3f} [{bootstrap_result['ci_lower']:.3f}, {bootstrap_result['ci_upper']:.3f}]")
    logger.info(f"\nAgreement: {agreement*100:.1f}%")
    logger.info(f"Clinical validation: {clinical_results['overall']['passed']}/3")
    logger.info(f"Komorowski validation: {'PASS' if komorowski_results['overall_passed'] else 'FAIL'}")

    torch.save({
        'q_network': agent.q_network.state_dict(),
        'scaler_mean': agent.scaler.mean_,
        'scaler_scale': agent.scaler.scale_,
        'config': config
    }, output_dir / 'model.pt')

    results = {
        'config': {
            'n_actions': n_actions,
            'n_states': n_states,
            'action_space': '3x3',
            'gamma': config.model.gamma
        },
        'training_history': history,
        'clinical_validation': {
            k: v for k, v in clinical_results.items()
            if k != 'overall'
        },
        'komorowski_validation': {
            'agreement_correlation': komorowski_results['agreement']['correlation'],
            'distance_correlation': komorowski_results['distance']['correlation'],
            'overall_passed': komorowski_results['overall_passed']
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
                'ess_ratio': float(ai_wdr_result['ess_ratio'])
            },
            'ai_bootstrap': {
                'mean': float(bootstrap_result['mean']),
                'ci_lower': float(bootstrap_result['ci_lower']),
                'ci_upper': float(bootstrap_result['ci_upper'])
            },
            'agreement': float(agreement)
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_dir / 'full_results.pkl', 'wb') as f:
        pickle.dump({
            'bootstrap_estimates': bootstrap_result['estimates'],
            'clinician_returns': clin_returns,
            'trajectory_values': ai_wdr_result['trajectory_values']
        }, f)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Finished at: {datetime.now()}")

    return results


if __name__ == "__main__":
    main()
