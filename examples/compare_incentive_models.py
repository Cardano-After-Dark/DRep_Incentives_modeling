import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any

from drepsim.core.models import SimulationParameters, IncentiveParameters
from drepsim.core.simulation import run_simulation
from drepsim.analysis.metrics import calculate_simulation_metrics
from drepsim.visualization.plots import visualize_simulation_results

# Set up logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("compare_incentive_models")

# Define simulation parameters
base_sim_params = SimulationParameters(
    total_epochs=15,  # Shorter for optimization
    drep_count=50,    # Smaller for faster simulation
    governance_actions_per_epoch=(3, 8),
    total_ada_delegated=1000000000,
    delegation_change_rate=0.05,
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

# Define base incentive parameters
base_incentive_params = IncentiveParameters(
    model_type="linear",  # Will override this for each test
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

# Model configurations to test
model_configs = [
    {"name": "Linear (Baseline)", "model_type": "linear"},
    {"name": "Non-Linear", "model_type": "non_linear"},
    {"name": "Temporal", "model_type": "temporal"},
    {"name": "Game Theory", "model_type": "game_theory"}
]

# Metrics to track
metrics_to_compare = [
    "participation_rate",
    "research_effort",
    "delegation_concentration",
    "governance_effectiveness",
    "decentralization",
    "reward_distribution"
]

def run_model_comparison(output_dir: str = "model_comparison"):
    """Run simulation with different incentive models and compare results."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Results storage
    results = {}
    
    # Run simulations for each model
    for config in model_configs:
        logger.info(f"Running simulation with {config['name']} model")
        
        # Create incentive params with this model type
        incentive_params = IncentiveParameters(
            model_type=config["model_type"],
            w1_delegation=base_incentive_params.w1_delegation,
            w2_participation=base_incentive_params.w2_participation,
            w3_successful_votes=base_incentive_params.w3_successful_votes,
            w4_veto_penalty=base_incentive_params.w4_veto_penalty,
            w5_decentralization=base_incentive_params.w5_decentralization,
            w6_peer_evaluation=base_incentive_params.w6_peer_evaluation,
            w7_community_engagement=base_incentive_params.w7_community_engagement,
            total_reward_per_epoch=base_incentive_params.total_reward_per_epoch,
            veto_penalty_factor=base_incentive_params.veto_penalty_factor,
            min_participation_threshold=base_incentive_params.min_participation_threshold
        )
        
        # Run simulation
        final_state = run_simulation(
            sim_params=base_sim_params,
            incentive_params=incentive_params
        )
        
        # Calculate metrics
        metrics = calculate_simulation_metrics(final_state)
        results[config["name"]] = {
            "state": final_state,
            "metrics": metrics
        }
        
        # Create model-specific output directory
        model_dir = os.path.join(output_dir, config["model_type"])
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate visualizations
        visualize_simulation_results(
            final_state,
            incentive_params,
            output_dir=model_dir,
            show_plots=False
        )
    
    # Compare results
    compare_results(results, output_dir)

def compare_results(results: Dict[str, Dict], output_dir: str):
    """Compare results across different models and generate comparison charts."""
    
    # Extract metrics for each model
    comparison_data = {}
    
    for metric in metrics_to_compare:
        comparison_data[metric] = {}
        for model_name, model_results in results.items():
            if metric in model_results["metrics"]:
                comparison_data[metric][model_name] = model_results["metrics"][metric]
    
    # Create comparison charts
    for metric, model_values in comparison_data.items():
        plt.figure(figsize=(12, 6))
        
        # For metrics that are time series data (tracked per epoch)
        if isinstance(list(model_values.values())[0], dict):
            # Convert to DataFrame for easier plotting
            epochs = max(len(v) for v in model_values.values())
            df = pd.DataFrame(index=range(epochs))
            
            for model_name, epoch_values in model_values.items():
                # Fill missing epochs with the last value
                full_series = [epoch_values.get(e, None) for e in range(epochs)]
                for i in range(1, epochs):
                    if full_series[i] is None:
                        full_series[i] = full_series[i-1]
                df[model_name] = full_series
            
            # Plot time series
            df.plot(ax=plt.gca())
            plt.title(f"{metric.replace('_', ' ').title()} Over Time")
            plt.xlabel("Epoch")
            plt.ylabel(metric.replace('_', ' ').title())
        else:
            # For scalar metrics
            plt.bar(model_values.keys(), model_values.values())
            plt.title(f"{metric.replace('_', ' ').title()} by Model")
            plt.ylabel(metric.replace('_', ' ').title())
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=300)
        plt.close()
    
    # Create summary table
    summary_data = {}
    final_epoch = base_sim_params.total_epochs - 1
    
    for model_name, model_results in results.items():
        metrics = model_results["metrics"]
        summary_data[model_name] = {
            "Participation Rate": metrics.get("participation_rate", {}).get(final_epoch, 0),
            "Research Effort": metrics.get("research_effort", {}).get(final_epoch, 0),
            "Delegation Gini": metrics.get("delegation_concentration", {}).get(final_epoch, 0),
            "Governance Effectiveness": metrics.get("governance_effectiveness", 0),
            "Decentralization": metrics.get("decentralization", 0),
            "Reward Gini": metrics.get("reward_distribution", {}).get(final_epoch, 0)
        }
    
    # Convert to DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "model_comparison_summary.csv"))
    
    # Create a radar chart for models comparison
    create_radar_chart(summary_df, output_dir)

def create_radar_chart(df: pd.DataFrame, output_dir: str):
    """Create a radar chart comparing models across key metrics."""
    # Normalize data to 0-1 range for radar chart
    normalized_df = pd.DataFrame()
    
    for col in df.index:
        # For Gini coefficients, lower is better, so invert
        if "Gini" in col:
            normalized_df[col] = 1 - (df.loc[col] - df.loc[col].min()) / (df.loc[col].max() - df.loc[col].min() + 1e-10)
        else:
            normalized_df[col] = (df.loc[col] - df.loc[col].min()) / (df.loc[col].max() - df.loc[col].min() + 1e-10)
    
    # Transpose to get models as rows
    normalized_df = normalized_df.T
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Number of variables
    N = len(normalized_df.columns)
    
    # Angle for each variable
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Labels
    labels = normalized_df.columns.tolist()
    labels += labels[:1]  # Close the loop
    
    # Plot each model
    for i, model in enumerate(normalized_df.index):
        values = normalized_df.loc[model].tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add labels
    plt.xticks(angles[:-1], [col.replace("_", " ").title() for col in normalized_df.columns], size=12)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Comparison - Normalized Metrics", size=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_radar_comparison.png"), dpi=300)
    plt.close()

def main():
    """Main execution function."""
    output_dir = "results/model_comparison"
    run_model_comparison(output_dir)
    logger.info(f"Model comparison complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 