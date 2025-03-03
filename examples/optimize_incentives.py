"""
Reinforcement learning optimization of incentive parameters.

This script uses reinforcement learning techniques to discover optimal
incentive parameters for different models.
"""

import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from drepsim.core.models import SimulationParameters, IncentiveParameters
from drepsim.core.simulation import run_simulation
from drepsim.analysis.reinforcement_learning import IncentiveOptimizer
from drepsim.visualization.plots import visualize_simulation_results

# Set up logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("optimize_incentives")

def main():
    # Define simulation parameters
    sim_params = SimulationParameters(
        total_epochs=15,  # Shorter for optimization
        drep_count=50,    # Smaller for faster simulation
        governance_actions_per_epoch=(3, 8),
        total_ada_delegated=1000000000,
        delegation_change_rate=0.05,
        veto_probability=0.05,
        drep_type_distribution={
            "Professional": 0.2,
            "Hobbyist": 0.4,
            "Passive": 0.3,
            "Strategic": 0.1
        },
        initial_delegation_distribution="pareto",
        delegation_distribution_params={"alpha": 1.5},
        peer_evaluation_sample_size=5,
        random_seed=42
    )
    
    # Define base incentive parameters
    base_incentives = IncentiveParameters(
        w1_delegation=0.2,
        w2_participation=0.3,
        w3_successful_votes=0.2,
        w4_veto_penalty=0.05,
        w5_decentralization=0.1,
        w6_peer_evaluation=0.1,
        w7_community_engagement=0.05,
        total_reward_per_epoch=100000,
        veto_penalty_factor=0.5,
        min_participation_threshold=0.1,
        model_type="non_linear"  # Optimize for non-linear model
    )
    
    # Define target metrics to optimize for
    target_metrics = [
        "participation_rate",     # Average participation rate
        "research_effort",        # Average research effort 
        "governance_effectiveness", # Effectiveness of governance
        "delegation_concentration",  # Gini coefficient (lower is better)
    ]
    
    # Define weights for target metrics
    target_weights = {
        "participation_rate": 0.3,       # Prioritize participation
        "research_effort": 0.2,          # Encourage research
        "governance_effectiveness": 0.3,  # Focus on good outcomes
        "delegation_concentration": 0.2,  # Discourage concentration
    }
    
    # Create output directory
    output_dir = "results/parameter_optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and run optimizer
    optimizer = IncentiveOptimizer(
        sim_params=sim_params,
        base_incentives=base_incentives,
        target_metrics=target_metrics,
        target_weights=target_weights,
        learning_rate=0.1,
        exploration_rate=0.2
    )
    
    logger.info("Starting incentive parameter optimization...")
    optimized_params = optimizer.optimize(iterations=50, random_seed=42)
    
    # Save optimization history
    history_df = optimizer.get_optimization_history()
    history_df.to_csv(os.path.join(output_dir, "optimization_history.csv"), index=False)
    
    # Visualize optimization process
    optimizer.visualize_optimization(os.path.join(output_dir, "optimization_progress.png"))
    
    # Save optimized parameters
    with open(os.path.join(output_dir, "optimized_parameters.json"), "w") as f:
        json.dump({
            "model_type": optimized_params.model_type,
            "w1_delegation": optimized_params.w1_delegation,
            "w2_participation": optimized_params.w2_participation,
            "w3_successful_votes": optimized_params.w3_successful_votes,
            "w4_veto_penalty": optimized_params.w4_veto_penalty,
            "w5_decentralization": optimized_params.w5_decentralization,
            "w6_peer_evaluation": optimized_params.w6_peer_evaluation,
            "w7_community_engagement": optimized_params.w7_community_engagement,
            "total_reward_per_epoch": optimized_params.total_reward_per_epoch,
            "veto_penalty_factor": optimized_params.veto_penalty_factor,
            "min_participation_threshold": optimized_params.min_participation_threshold
        }, f, indent=2)
    
    # Run final simulation with optimized parameters
    logger.info("Running simulation with optimized parameters...")
    
    # Use more epochs for final evaluation
    sim_params.total_epochs = 30
    
    final_state = run_simulation(
        sim_params=sim_params,
        incentive_params=optimized_params
    )
    
    # Generate visualizations
    visualize_simulation_results(
        final_state,
        optimized_params,
        output_dir=output_dir,
        show_plots=False
    )
    
    logger.info(f"Optimization complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 