"""
Monte Carlo simulation example for the DRep incentivization framework.

This script demonstrates how to set up and run a Monte Carlo simulation
to explore the parameter space and find optimal incentive structures.
"""

import os
import logging
import pickle
import numpy as np
from typing import Dict, Tuple

from drepsim.core.models import SimulationParameters, IncentiveParameters
from drepsim.analysis.monte_carlo import (
    run_monte_carlo_simulation, 
    analyze_monte_carlo_results,
    generate_parameter_set
)
from drepsim.visualization.plots import visualize_monte_carlo_results

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("monte_carlo_example")

# Define base simulation parameters
base_sim_params = SimulationParameters(
    total_epochs=15,  # Shorter for Monte Carlo
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
    peer_evaluation_sample_size=5
)

# Define parameter ranges to explore
param_ranges = {
    "w1_delegation": (0.05, 0.5),
    "w2_participation": (0.05, 0.6),
    "w3_successful_votes": (0.05, 0.5),
    "w4_veto_penalty": (0.01, 0.3),
    "w5_decentralization": (0.05, 0.4),
    "w6_peer_evaluation": (0.05, 0.4),
    "w7_community_engagement": (0.01, 0.3),
    "total_reward_per_epoch": (50000, 300000),
    "veto_penalty_factor": (0.1, 1.5),
    "min_participation_threshold": (0.05, 0.4)
}

def main():
    """Run the Monte Carlo simulation and analysis."""
    # Create output directory
    output_dir = "monte_carlo_results"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Running Monte Carlo simulation...")
    
    # Run the Monte Carlo simulation with parameter ranges
    results = run_monte_carlo_simulation(
        n_iterations=200,  # Increase from 50 to 200
        sim_params=base_sim_params,
        param_ranges=param_ranges,  # Pass param_ranges directly
        n_processes=4,
        base_random_seed=42,
        save_intermediate=os.path.join(output_dir, "intermediate_results.pkl")
    )
    
    logger.info("Analyzing Monte Carlo results...")
    analysis = analyze_monte_carlo_results(results)
    
    logger.info("Generating visualizations...")
    visualize_monte_carlo_results(results, analysis, output_dir=output_dir)
    
    # Save final results
    logger.info("Saving results...")
    with open(os.path.join(output_dir, "monte_carlo_results.pkl"), "wb") as f:
        pickle.dump({"results": results, "analysis": analysis}, f)
    
    logger.info("Monte Carlo analysis complete!")

if __name__ == "__main__":
    main() 