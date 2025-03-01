"""
Monte Carlo simulation framework for exploring incentive parameter space.

This module provides tools for running multiple simulations with different
parameter combinations to identify optimal incentive structures.
"""

import os
import pickle
import logging
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Tuple, Any, Optional, Callable
from tqdm import tqdm
import pandas as pd

from drepsim.core.models import SimulationParameters, IncentiveParameters
from drepsim.core.simulation import run_simulation
from drepsim.analysis.metrics import calculate_simulation_metrics

logger = logging.getLogger("drepsim.monte_carlo")

# Global variables for the worker function
sim_params_dict = {}
base_seed = None
global_save_intermediate = None
global_save_frequency = 10

# Define this at module level, not inside another function
def _run_single_iteration(args) -> Dict[str, Any]:
    """
    Run a single Monte Carlo iteration.
    
    Args:
        args: Tuple of (iteration, incentive_params)
        
    Returns:
        Dictionary of metrics from the simulation
    """
    iteration, incentive_params_local = args
    
    # Create a unique random seed for this iteration
    random_seed = base_seed + iteration if base_seed is not None else np.random.randint(0, 2**32 - 1)
    
    # Create a copy of the SimulationParameters object with the new random seed
    sim_params_copy = SimulationParameters(
        total_epochs=sim_params_dict.get('total_epochs', 10),
        drep_count=sim_params_dict.get('drep_count', 50),
        governance_actions_per_epoch=sim_params_dict.get('governance_actions_per_epoch', (3, 8)),
        total_ada_delegated=sim_params_dict.get('total_ada_delegated', 1000000000),
        delegation_change_rate=sim_params_dict.get('delegation_change_rate', 0.05),
        veto_probability=sim_params_dict.get('veto_probability', 0.05),
        drep_type_distribution=sim_params_dict.get('drep_type_distribution', {
            "Professional": 0.2,
            "Hobbyist": 0.4,
            "Passive": 0.3,
            "Strategic": 0.1
        }),
        initial_delegation_distribution=sim_params_dict.get('initial_delegation_distribution', "pareto"),
        delegation_distribution_params=sim_params_dict.get('delegation_distribution_params', {"alpha": 1.5}),
        peer_evaluation_sample_size=sim_params_dict.get('peer_evaluation_sample_size', 5),
        random_seed=random_seed
    )
    
    # Run the simulation
    final_state = run_simulation(sim_params_copy, incentive_params_local)
    
    # Calculate metrics
    metrics = calculate_simulation_metrics(final_state)
    metrics["iteration"] = iteration
    metrics["random_seed"] = random_seed
    
    # Create a result dictionary with both metrics and parameters
    result = {
        'metrics': metrics,
        'parameters': {
            'w1_delegation': incentive_params_local.w1_delegation,
            'w2_participation': incentive_params_local.w2_participation,
            'w3_successful_votes': incentive_params_local.w3_successful_votes,
            'w4_veto_penalty': incentive_params_local.w4_veto_penalty,
            'w5_decentralization': incentive_params_local.w5_decentralization,
            'w6_peer_evaluation': incentive_params_local.w6_peer_evaluation,
            'w7_community_engagement': incentive_params_local.w7_community_engagement,
            'total_reward_per_epoch': incentive_params_local.total_reward_per_epoch,
            'veto_penalty_factor': incentive_params_local.veto_penalty_factor,
            'min_participation_threshold': incentive_params_local.min_participation_threshold
        }
    }
    
    # Save intermediate results if requested
    if global_save_intermediate and iteration % global_save_frequency == 0:
        try:
            with open(global_save_intermediate, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")
    
    return result

def generate_parameter_set(param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """
    Generate a random set of parameters within the specified ranges.
    
    Args:
        param_ranges: Dictionary mapping parameter names to (min, max) tuples
        
    Returns:
        Dictionary of sampled parameters
    """
    params = {}
    for param_name, (min_val, max_val) in param_ranges.items():
        params[param_name] = np.random.uniform(min_val, max_val)
    
    return params


def run_monte_carlo_simulation(
    n_iterations: int,
    sim_params: SimulationParameters,
    incentive_params_in: Optional[IncentiveParameters] = None,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    n_processes: int = None,
    base_random_seed: Optional[int] = None,
    save_intermediate: Optional[str] = None,
    save_frequency: int = 10
) -> List[Dict[str, Any]]:
    """
    Run a Monte Carlo simulation with multiple iterations.
    
    Args:
        n_iterations: Number of iterations to run
        sim_params: Simulation parameters
        incentive_params_in: Base incentive parameters (optional)
        param_ranges: Ranges for parameter sampling (required if incentive_params_in is None)
        n_processes: Number of processes to use (default: CPU count)
        base_random_seed: Base random seed (will be incremented for each iteration)
        save_intermediate: Path to save intermediate results
        save_frequency: How often to save intermediate results
        
    Returns:
        List of metric dictionaries from each iteration
    """
    logger.info(f"Starting Monte Carlo simulation with {n_iterations} iterations")
    
    # Set global variables for the worker function
    global sim_params_dict, base_seed, global_save_intermediate, global_save_frequency
    
    # Make sure we're using a copy of the parameters to avoid modifying the original
    sim_params_dict = sim_params.__dict__.copy()
    base_seed = base_random_seed
    global_save_intermediate = save_intermediate
    global_save_frequency = save_frequency
    
    # Validate parameters
    if incentive_params_in is None and param_ranges is None:
        raise ValueError("Either incentive_params_in or param_ranges must be provided")
    
    # Create argument tuples for each iteration
    args_list = []
    for i in range(n_iterations):
        # Generate a new parameter set for each iteration
        if param_ranges:
            params_dict = generate_parameter_set(param_ranges)
            
            # Create IncentiveParameters object
            incentive_params = IncentiveParameters(
                w1_delegation=params_dict.get('w1_delegation', 0.2),
                w2_participation=params_dict.get('w2_participation', 0.2),
                w3_successful_votes=params_dict.get('w3_successful_votes', 0.2),
                w4_veto_penalty=params_dict.get('w4_veto_penalty', 0.1),
                w5_decentralization=params_dict.get('w5_decentralization', 0.1),
                w6_peer_evaluation=params_dict.get('w6_peer_evaluation', 0.1),
                w7_community_engagement=params_dict.get('w7_community_engagement', 0.1),
                total_reward_per_epoch=params_dict.get('total_reward_per_epoch', 100000),
                veto_penalty_factor=params_dict.get('veto_penalty_factor', 0.5),
                min_participation_threshold=params_dict.get('min_participation_threshold', 0.1)
            )
        else:
            # Use the provided incentive parameters
            incentive_params = incentive_params_in
        
        args_list.append((i, incentive_params))
    
    # Run simulations in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(_run_single_iteration, args_list),
            total=n_iterations,
            desc="Monte Carlo Progress"
        ))
    
    logger.info(f"Completed {n_iterations} Monte Carlo iterations")
    return results


def analyze_monte_carlo_results(results: List[Dict]) -> Dict[str, Any]:
    """
    Analyze the results of a Monte Carlo simulation to find optimal parameters.
    
    Args:
        results: List of result dictionaries from monte_carlo_simulation
        
    Returns:
        Dictionary of analysis results
    """
    if not results:
        return {}
    
    logger.info("Analyzing Monte Carlo simulation results")
    
    # Extract parameters and key metrics
    params_list = []
    metrics_list = []
    
    for result in results:
        params_list.append(result['parameters'])
        metrics_list.append(result['metrics'])
    
    params_df = pd.DataFrame(params_list)
    metrics_df = pd.DataFrame(metrics_list)
    
    # Correlation analysis
    correlation_matrix = pd.DataFrame()
    for param in params_df.columns:
        for metric in metrics_df.columns:
            correlation = np.corrcoef(params_df[param], metrics_df[metric])[0, 1]
            correlation_matrix.loc[param, metric] = correlation
    
    # Find optimal parameter sets for different objectives
    objectives = {
        'min_gini': ('final_gini', min),  # Minimize concentration
        'max_participation': ('avg_participation', max),  # Maximize participation
        'max_research': ('avg_research_effort', max),  # Maximize research effort
        'balanced': None  # Defined below as a composite score
    }
    
    optimal_params = {}
    used_indices = set()  # Track which parameter sets have been used
    
    # Process objectives in order of priority
    for obj_name, obj_info in objectives.items():
        if obj_name == 'balanced':
            # Create a composite score
            composite_scores = (
                -1 * metrics_df['final_gini'] +  # Low Gini (negated)
                metrics_df['avg_participation'] +  # High participation
                metrics_df['avg_research_effort']  # High research
            ) / 3  # Average the normalized values
            
            # Get top 5 indices for balanced objective
            top_indices = composite_scores.nlargest(5).index.tolist()
            # Choose the first one not already used
            for idx in top_indices:
                if idx not in used_indices:
                    best_idx = idx
                    used_indices.add(idx)
                    break
            else:
                # If all are used, just take the best one
                best_idx = composite_scores.idxmax()
        else:
            metric, func = obj_info
            # Get top 5 indices for this objective
            if func == min:
                top_indices = metrics_df[metric].nsmallest(5).index.tolist()
            else:  # func == max
                top_indices = metrics_df[metric].nlargest(5).index.tolist()
            
            # Choose the first one not already used
            for idx in top_indices:
                if idx not in used_indices:
                    best_idx = idx
                    used_indices.add(idx)
                    break
            else:
                # If all are used, just take the best one
                if func == min:
                    best_idx = metrics_df[metric].idxmin()
                else:
                    best_idx = metrics_df[metric].idxmax()
        
        optimal_params[obj_name] = {
            'parameters': params_list[best_idx],
            'metrics': metrics_list[best_idx]
        }
    
    # Sensitivity analysis - how much each parameter affects key metrics
    sensitivity = {}
    for param in params_df.columns:
        sensitivity[param] = {}
        for metric in metrics_df.columns:
            if pd.api.types.is_numeric_dtype(metrics_df[metric]):
                # Check if this param-metric pair exists in the correlation matrix
                if param in correlation_matrix.index and metric in correlation_matrix.columns:
                    sensitivity[param][metric] = correlation_matrix.loc[param, metric]
                else:
                    # If not in correlation matrix, set to 0 (no correlation)
                    sensitivity[param][metric] = 0.0
    
    return {
        'correlation_matrix': correlation_matrix,
        'optimal_parameters': optimal_params,
        'sensitivity_analysis': sensitivity
    }


def save_monte_carlo_results(
    results: List[Dict], 
    analysis: Dict[str, Any], 
    filename: str
) -> None:
    """
    Save Monte Carlo simulation results and analysis to a file.
    
    Args:
        results: List of result dictionaries
        analysis: Analysis results dictionary
        filename: Path to save the results
    """
    data = {
        'results': results,
        'analysis': analysis
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Monte Carlo results saved to {filename}")


def load_monte_carlo_results(filename: str) -> Dict[str, Any]:
    """
    Load Monte Carlo simulation results from a file.
    
    Args:
        filename: Path to the saved results
        
    Returns:
        Dictionary containing results and analysis
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Monte Carlo results loaded from {filename}")
    return data