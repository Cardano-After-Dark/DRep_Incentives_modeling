"""
Basic simulation example for the DRep incentivization framework.

This script demonstrates how to set up and run a simple simulation,
analyze the results, and generate visualizations.
"""

import os
import logging
from drepsim.core.models import SimulationParameters, IncentiveParameters
from drepsim.core.simulation import run_simulation
from drepsim.visualization.plots import visualize_simulation_results
from drepsim.analysis.metrics import calculate_simulation_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define incentive parameters
incentive_params = IncentiveParameters(
    w1_delegation=0.2,
    w2_participation=0.3,
    w3_successful_votes=0.2,
    w4_veto_penalty=0.05,
    w5_decentralization=0.1,
    w6_peer_evaluation=0.1,
    w7_community_engagement=0.05,
    total_reward_per_epoch=100000,  # 100k ADA per epoch
    veto_penalty_factor=0.5,
    min_participation_threshold=0.1
)

# Define simulation parameters
sim_params = SimulationParameters(
    total_epochs=20,
    drep_count=100,
    governance_actions_per_epoch=(5, 15),
    total_ada_delegated=1000000000,  # 1 billion ADA delegated
    delegation_change_rate=0.05,  # 5% of delegations can move per epoch
    veto_probability=0.05,  # 5% chance of a veto
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

def progress_callback(current, total):
    """Simple progress callback function."""
    print(f"Progress: {current}/{total} epochs ({current/total*100:.1f}%)")

def main():
    # Create output directory
    output_dir = "example_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulation
    print("Running simulation...")
    final_state = run_simulation(sim_params, incentive_params, progress_callback)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_simulation_metrics(final_state)
    
    # Print key metrics
    print("\nSimulation Results:")
    print(f"Final Gini Coefficient: {metrics['final_gini']:.4f}")
    print(f"Average Participation Rate: {metrics['avg_participation']:.4f}")
    print(f"Average Research Effort: {metrics['avg_research_effort']:.4f}")
    print(f"Action Acceptance Rate: {metrics['action_acceptance_rate']:.4f}")
    
    # Generate visualizations
    print("Generating visualizations...")
    visualize_simulation_results(
        final_state, 
        incentive_params, 
        output_dir=os.path.join(output_dir, "visualizations")
    )
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 