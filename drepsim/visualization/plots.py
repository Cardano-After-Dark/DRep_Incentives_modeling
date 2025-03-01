"""
Visualization functions for simulation results.

This module provides functions for generating various plots and visualizations
from simulation results to help understand and analyze the outcomes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import pandas as pd
import logging

from drepsim.core.models import SimulationState, IncentiveParameters

logger = logging.getLogger(__name__)


def visualize_simulation_results(
    state: SimulationState,
    incentive_params: IncentiveParameters,
    output_dir: str = "visualizations",
    show_plots: bool = False
) -> None:
    """
    Generate visualizations from a simulation run.
    
    Args:
        state: Final simulation state
        incentive_params: Incentive parameters used
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots interactively
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plots
    plt.style.use('ggplot')
    
    # 1. Delegation concentration over time
    epochs = list(state.delegation_concentration.keys())
    gini_values = list(state.delegation_concentration.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, gini_values, marker='o', linewidth=2)
    plt.title('Delegation Concentration (Gini Coefficient) Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Gini Coefficient')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(f"{output_dir}/delegation_concentration.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    
    # 2. Average participation rate over time
    participation_avg = [
        np.mean(list(state.participation_rates[epoch].values())) 
        for epoch in epochs if epoch in state.participation_rates
    ]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs[:len(participation_avg)], participation_avg, marker='o', linewidth=2)
    plt.title('Average Participation Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Participation Rate')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(f"{output_dir}/participation_rate.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    
    # 3. Rewards by DRep type
    drep_types = {}
    for drep_id, drep in state.dreps.items():
        drep_type = drep.profile.drep_type
        if drep_type not in drep_types:
            drep_types[drep_type] = []
        drep_types[drep_type].append(drep_id)
    
    avg_rewards_by_type = {}
    for epoch in epochs:
        if epoch not in state.rewards:
            continue
            
        for drep_type, drep_ids in drep_types.items():
            if drep_type not in avg_rewards_by_type:
                avg_rewards_by_type[drep_type] = []
            
            type_rewards = [state.rewards[epoch].get(drep_id, 0) for drep_id in drep_ids]
            avg_rewards_by_type[drep_type].append(np.mean(type_rewards))
    
    plt.figure(figsize=(12, 7))
    for drep_type, rewards in avg_rewards_by_type.items():
        epochs_with_rewards = epochs[:len(rewards)]
        plt.plot(epochs_with_rewards, rewards, marker='o', linewidth=2, label=drep_type)
    
    plt.title('Average Rewards by DRep Type')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward (ADA)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/rewards_by_type.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    
    # 4. Delegation share by DRep type
    delegation_share_by_type = {drep_type: [] for drep_type in drep_types}
    
    for epoch in epochs:
        total_delegation = sum(drep.delegation_history[epoch] for drep in state.dreps.values() 
                              if epoch < len(drep.delegation_history))
        
        for drep_type, drep_ids in drep_types.items():
            type_delegation = sum(state.dreps[drep_id].delegation_history[epoch] 
                                for drep_id in drep_ids 
                                if epoch < len(state.dreps[drep_id].delegation_history))
            
            share = type_delegation / total_delegation if total_delegation > 0 else 0
            delegation_share_by_type[drep_type].append(share)
    
    plt.figure(figsize=(12, 7))
    for drep_type, shares in delegation_share_by_type.items():
        plt.plot(epochs[:len(shares)], shares, marker='o', linewidth=2, label=drep_type)
    
    plt.title('Delegation Share by DRep Type')
    plt.xlabel('Epoch')
    plt.ylabel('Share of Total Delegation')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(f"{output_dir}/delegation_share.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    
    # 5. Distribution of final delegations
    final_delegations = [drep.delegated_ada for drep in state.dreps.values()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(final_delegations, bins=30, alpha=0.7)
    plt.title('Distribution of Final Delegations')
    plt.xlabel('Delegation Amount (ADA)')
    plt.ylabel('Number of DReps')
    plt.grid(True)
    plt.savefig(f"{output_dir}/delegation_distribution.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    
    # 6. Parameter importance
    params = {
        'w1_delegation': incentive_params.w1_delegation,
        'w2_participation': incentive_params.w2_participation,
        'w3_successful_votes': incentive_params.w3_successful_votes,
        'w4_veto_penalty': incentive_params.w4_veto_penalty,
        'w5_decentralization': incentive_params.w5_decentralization,
        'w6_peer_evaluation': incentive_params.w6_peer_evaluation,
        'w7_community_engagement': incentive_params.w7_community_engagement
    }
    
    plt.figure(figsize=(12, 6))
    plt.bar(params.keys(), params.values(), alpha=0.7)
    plt.title('Incentive Parameter Weights')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.savefig(f"{output_dir}/parameter_weights.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()


def visualize_monte_carlo_results(
    results: List[Dict],
    analysis: Dict[str, Any],
    output_dir: str = "monte_carlo_results",
    show_plots: bool = False
) -> None:
    """
    Generate visualizations from Monte Carlo simulation results.
    
    Args:
        results: List of result dictionaries
        analysis: Analysis results from analyze_monte_carlo_results
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots interactively
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    params_list = [r['parameters'] for r in results]
    metrics_list = [r['metrics'] for r in results]
    
    params_df = pd.DataFrame(params_list)
    metrics_df = pd.DataFrame(metrics_list)
    
    # 1. Parameter vs. key metrics correlation heatmap
    if 'correlation_matrix' in analysis and not analysis['correlation_matrix'].empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(analysis['correlation_matrix'], annot=True, cmap='coolwarm', center=0)
        plt.title('Parameter-Metric Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()
    else:
        logger.warning("Correlation matrix is empty - skipping heatmap visualization")
    
    # 2. Scatter plots for important relationships
    key_metrics = ['final_gini', 'avg_participation', 'avg_research_effort']
    key_params = ['w1_delegation', 'w2_participation', 'w3_successful_votes', 'w6_peer_evaluation']
    
    for param in key_params:
        if param not in params_df.columns:
            continue
            
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(15, 5))
        for i, metric in enumerate(key_metrics):
            if metric not in metrics_df.columns:
                continue
                
            axes[i].scatter(params_df[param], metrics_df[metric], alpha=0.6)
            axes[i].set_xlabel(param)
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{param} vs {metric}')
            axes[i].grid(True)
            
            # Add trend line
            z = np.polyfit(params_df[param], metrics_df[metric], 1)
            p = np.poly1d(z)
            axes[i].plot(params_df[param], p(params_df[param]), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/param_{param}_scatter.png", dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()
    
    # 3. Optimal parameter configurations
    if 'optimal_parameters' in analysis:
        # Plot radar chart for each optimal parameter set
        objectives = list(analysis['optimal_parameters'].keys())
        
        # Create a single figure with subplots for comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 14), subplot_kw=dict(polar=True))
        axes = axes.flatten()
        
        for i, obj_name in enumerate(objectives):
            obj_data = analysis['optimal_parameters'][obj_name]
            params = obj_data['parameters']
            
            # Filter to just the weight parameters
            weight_params = {k: v for k, v in params.items() if k.startswith('w')}
            
            # Prepare data for radar chart
            categories = list(weight_params.keys())
            values = list(weight_params.values())
            
            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Close the loop
            angles += angles[:1]  # Close the loop
            categories += categories[:1]  # Close the loop
            
            ax = axes[i]
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
            ax.set_ylim(0, 0.6)  # Fixed scale for better comparison
            ax.grid(True)
            ax.set_title(f'Optimal Parameters for {obj_name}')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/optimal_parameters_comparison.png", dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()
        
        # Also save individual charts as before
        for obj_name in objectives:
            obj_data = analysis['optimal_parameters'][obj_name]
            params = obj_data['parameters']
            
            # Filter to just the weight parameters
            weight_params = {k: v for k, v in params.items() if k.startswith('w')}
            
            # Prepare data for radar chart
            categories = list(weight_params.keys())
            values = list(weight_params.values())
            
            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Close the loop
            angles += angles[:1]  # Close the loop
            categories += categories[:1]  # Close the loop
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
            ax.set_ylim(0, max(values) * 1.1)
            ax.grid(True)
            plt.title(f'Optimal Parameters for {obj_name}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/optimal_{obj_name}_radar.png", dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            plt.close()
    
    # 4. Sensitivity analysis
    if 'sensitivity_analysis' in analysis:
        sensitivity = analysis['sensitivity_analysis']
        params = list(sensitivity.keys())
        metrics = ['final_gini', 'avg_participation', 'avg_research_effort']
        
        sensitivity_data = []
        for param in params:
            for metric in metrics:
                if metric in sensitivity[param]:
                    sensitivity_data.append({
                        'Parameter': param,
                        'Metric': metric,
                        'Sensitivity': sensitivity[param][metric]
                    })
        
        sensitivity_df = pd.DataFrame(sensitivity_data)
        
        plt.figure(figsize=(12, 8))
        pivot_table = sensitivity_df.pivot(index='Parameter', columns='Metric', values='Sensitivity')
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')
        plt.title('Parameter Sensitivity Analysis')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sensitivity_heatmap.png", dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()
    
    # 5. Distribution of key metrics
    for metric in key_metrics:
        if metric not in metrics_df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        plt.hist(metrics_df[metric], bins=20, alpha=0.7)
        plt.title(f'Distribution of {metric}')
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f"{output_dir}/metric_{metric}_distribution.png", dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close() 