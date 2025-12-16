"""Main execution: Train both GBM and IQL with all refinements."""

import numpy as np
import json
import torch
import pickle
import logging
from pathlib import Path
from datetime import datetime

from config import Config
from data_loader import DataLoader
from behavior_policy import BehaviorPolicy
from gbm_agent import EnsembleGBMAgent
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
    logger.info("REFINED RL PIPELINE: GBM + IQL")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Action space: {config.action.n_actions} (3x3)")
    logger.info(f"Temporal split: {config.data.temporal_split}")
    logger.info(f"GBM: Ensemble with state noise (std={config.gbm.state_noise_std})")
    logger.info(f"IQL: Strong auxiliary tasks (BP={config.iql.bp_weight}, Lac={config.iql.lactate_weight})")

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
    logger.info("TRAINING GBM ENSEMBLE")
    logger.info("=" * 80)

    gbm_agent = EnsembleGBMAgent(n_states, n_actions, config)
    gbm_agent.fit_scalers(train_data['states'])
    gbm_agent.train(train_data, val_data)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING IQL AGENT")
    logger.info("=" * 80)

    iql_agent = IQLAgent(n_states, n_actions, config)
    iql_agent.fit_scaler(train_data['states'])

    trainer = IQLTrainer(iql_agent, config)
    iql_history = trainer.train(train_data, val_data)

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

    logger.info("\n" + "=" * 60)
    logger.info("GBM EVALUATION")
    logger.info("=" * 60)

    logger.info("\nWithout state noise (greedy):")
    gbm_validator_greedy = ClinicalValidator(
        gbm_agent, data_loader.state_columns,
        data_loader.action_simplifier, use_noise=False
    )
    gbm_clinical_greedy = gbm_validator_greedy.validate()

    gbm_wdr_greedy = wdr_estimator.estimate(
        test_data['df'], test_data['states'], test_data['actions'],
        test_data['rewards'], test_data['dones'],
        gbm_agent, behavior_policy, use_noise=False
    )

    gbm_actions_greedy = gbm_agent.select_actions(test_data['states'], use_noise=False)
    gbm_agreement_greedy = np.mean(gbm_actions_greedy == test_data['actions'])

    logger.info(f"  WDR: {gbm_wdr_greedy['wdr_estimate']:.3f}")
    logger.info(f"  Agreement: {gbm_agreement_greedy*100:.1f}%")
    logger.info(f"  Clinical: {gbm_clinical_greedy['overall']['passed']}/3")

    logger.info("\nWith state noise (diverse):")
    gbm_validator_noise = ClinicalValidator(
        gbm_agent, data_loader.state_columns,
        data_loader.action_simplifier, use_noise=True
    )
    gbm_clinical_noise = gbm_validator_noise.validate()

    gbm_wdr_noise = wdr_estimator.estimate(
        test_data['df'], test_data['states'], test_data['actions'],
        test_data['rewards'], test_data['dones'],
        gbm_agent, behavior_policy, use_noise=True
    )

    gbm_actions_noise = gbm_agent.select_actions(test_data['states'], use_noise=True)
    gbm_agreement_noise = np.mean(gbm_actions_noise == test_data['actions'])

    logger.info(f"  WDR: {gbm_wdr_noise['wdr_estimate']:.3f}")
    logger.info(f"  Agreement: {gbm_agreement_noise*100:.1f}%")
    logger.info(f"  Clinical: {gbm_clinical_noise['overall']['passed']}/3")

    logger.info("\n" + "=" * 60)
    logger.info("IQL EVALUATION")
    logger.info("=" * 60)

    iql_validator = ClinicalValidator(
        iql_agent, data_loader.state_columns,
        data_loader.action_simplifier, use_noise=False
    )
    iql_clinical = iql_validator.validate()

    iql_wdr = wdr_estimator.estimate(
        test_data['df'], test_data['states'], test_data['actions'],
        test_data['rewards'], test_data['dones'],
        iql_agent, behavior_policy, use_noise=False
    )

    iql_actions = iql_agent.select_actions(test_data['states'])
    iql_agreement = np.mean(iql_actions == test_data['actions'])

    logger.info(f"\nIQL Agent:")
    logger.info(f"  WDR: {iql_wdr['wdr_estimate']:.3f} ± {1.96*iql_wdr['wdr_se']:.3f}")
    logger.info(f"  Agreement: {iql_agreement*100:.1f}%")
    logger.info(f"  Clinical: {iql_clinical['overall']['passed']}/3")

    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)

    logger.info(f"\n{'Model':<30} {'WDR':<12} {'Clinical':<12} {'Diversity'}")
    logger.info("-" * 70)
    logger.info(f"{'Clinician':<30} {clin_value:<12.3f} {'N/A':<12} {'N/A'}")
    logger.info(f"{'GBM (greedy)':<30} {gbm_wdr_greedy['wdr_estimate']:<12.3f} {gbm_clinical_greedy['overall']['passed']}/3{'':<9} {gbm_clinical_greedy.get('diversity', {}).get('n_unique', 'N/A')}/9")
    logger.info(f"{'GBM (state noise)':<30} {gbm_wdr_noise['wdr_estimate']:<12.3f} {gbm_clinical_noise['overall']['passed']}/3{'':<9} {gbm_clinical_noise.get('diversity', {}).get('n_unique', 'N/A')}/9")
    logger.info(f"{'IQL (strong aux)':<30} {iql_wdr['wdr_estimate']:<12.3f} {iql_clinical['overall']['passed']}/3{'':<9} {iql_clinical.get('diversity', {}).get('n_unique', 'N/A')}/9")

    results = {
        'config': {
            'n_actions': n_actions,
            'n_states': n_states,
            'temporal_split': config.data.temporal_split,
            'gbm_state_noise': config.gbm.state_noise_std,
            'iql_bp_weight': config.iql.bp_weight,
            'iql_lactate_weight': config.iql.lactate_weight
        },
        'clinician': {
            'value': float(clin_value),
            'std': float(clin_std)
        },
        'gbm_greedy': {
            'wdr': float(gbm_wdr_greedy['wdr_estimate']),
            'wdr_se': float(gbm_wdr_greedy['wdr_se']),
            'agreement': float(gbm_agreement_greedy),
            'clinical_passed': int(gbm_clinical_greedy['overall']['passed']),
            'diversity': int(gbm_clinical_greedy.get('diversity', {}).get('n_unique', 0))
        },
        'gbm_noise': {
            'wdr': float(gbm_wdr_noise['wdr_estimate']),
            'wdr_se': float(gbm_wdr_noise['wdr_se']),
            'agreement': float(gbm_agreement_noise),
            'clinical_passed': int(gbm_clinical_noise['overall']['passed']),
            'diversity': int(gbm_clinical_noise.get('diversity', {}).get('n_unique', 0))
        },
        'iql': {
            'wdr': float(iql_wdr['wdr_estimate']),
            'wdr_se': float(iql_wdr['wdr_se']),
            'agreement': float(iql_agreement),
            'clinical_passed': int(iql_clinical['overall']['passed']),
            'diversity': int(iql_clinical.get('diversity', {}).get('n_unique', 0))
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_dir / 'gbm_models.pkl', 'wb') as f:
        pickle.dump(gbm_agent, f)

    torch.save({
        'q_network': iql_agent.q_network.state_dict(),
        'v_network': iql_agent.v_network.state_dict(),
        'target_network': iql_agent.target_network.state_dict(),
        'scaler_mean': iql_agent.scaler.mean_,
        'scaler_scale': iql_agent.scaler.scale_
    }, output_dir / 'iql_model.pt')

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Finished at: {datetime.now()}")

    return results


if __name__ == "__main__":
    main()
