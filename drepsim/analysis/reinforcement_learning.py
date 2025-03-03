"""
Reinforcement learning module for optimizing incentive parameters.

This module uses reinforcement learning techniques to discover optimal
incentive parameters through repeated simulations and parameter adjustments.
"""

import numpy as np
import logging
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from tqdm import tqdm

from drepsim.core.models import SimulationParameters, IncentiveParameters
from drepsim.core.simulation import run_simulation
from drepsim.analysis.metrics import calculate_simulation_metrics

logger = logging.getLogger(__name__)

class IncentiveOptimizer:
    """Reinforcement learning optimizer for incentive parameters."""
    
    def __init__(
        self,
        sim_params: SimulationParameters,
        base_incentives: IncentiveParameters,
        target_metrics: List[str],
        target_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.05,
        exploration_rate: float = 0.2,
        min_values: Optional[Dict[str, float]] = None,
        max_values: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the incentive optimizer.
        
        Args:
            sim_params: Simulation parameters to use
            base_incentives: Initial incentive parameters
            target_metrics: List of metrics to optimize for
            target_weights: Weights for each target metric (importance)
            learning_rate: How quickly to adjust parameters
            exploration_rate: Probability of random exploration
            min_values: Minimum allowed values for parameters
            max_values: Maximum allowed values for parameters
        """
        self.sim_params = sim_params
        self.base_incentives = base_incentives
        self.target_metrics = target_metrics
        
        # Default to equal weights if not provided
        if target_weights is None:
            self.target_weights = {metric: 1.0 / len(target_metrics) for metric in target_metrics}
        else:
            # Normalize weights to sum to 1
            weight_sum = sum(target_weights.values())
            self.target_weights = {k: v / weight_sum for k, v in target_weights.items()}
        
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Parameter bounds
        self.min_values = min_values or {
            "w1_delegation": 0.05,
            "w2_participation": 0.05,
            "w3_successful_votes": 0.05,
            "w4_veto_penalty": 0.01,
            "w5_decentralization": 0.05,
            "w6_peer_evaluation": 0.01,
            "w7_community_engagement": 0.01,
            "veto_penalty_factor": 0.1,
            "min_participation_threshold": 0.05
        }
        
        self.max_values = max_values or {
            "w1_delegation": 0.5,
            "w2_participation": 0.5,
            "w3_successful_votes": 0.5,
            "w4_veto_penalty": 0.3,
            "w5_decentralization": 0.3,
            "w6_peer_evaluation": 0.3,
            "w7_community_engagement": 0.3,
            "veto_penalty_factor": 2.0,
            "min_participation_threshold": 0.4
        }
        
        # Best parameters found
        self.best_params = self._copy_incentive_params(base_incentives)
        self.best_fitness = float('-inf')
        
        # History tracking
        self.history = []
    
    def _copy_incentive_params(self, params: IncentiveParameters) -> IncentiveParameters:
        """Create a copy of incentive parameters."""
        return IncentiveParameters(
            model_type=params.model_type,
            w1_delegation=params.w1_delegation,
            w2_participation=params.w2_participation,
            w3_successful_votes=params.w3_successful_votes,
            w4_veto_penalty=params.w4_veto_penalty,
            w5_decentralization=params.w5_decentralization,
            w6_peer_evaluation=params.w6_peer_evaluation,
            w7_community_engagement=params.w7_community_engagement,
            total_reward_per_epoch=params.total_reward_per_epoch,
            veto_penalty_factor=params.veto_penalty_factor,
            min_participation_threshold=params.min_participation_threshold
        )
    
    def _mutate_params(self, params: IncentiveParameters) -> IncentiveParameters:
        """Create a mutated copy of the parameters."""
        # Extract current values
        current_values = {
            "w1_delegation": params.w1_delegation,
            "w2_participation": params.w2_participation,
            "w3_successful_votes": params.w3_successful_votes,
            "w4_veto_penalty": params.w4_veto_penalty,
            "w5_decentralization": params.w5_decentralization,
            "w6_peer_evaluation": params.w6_peer_evaluation,
            "w7_community_engagement": params.w7_community_engagement,
            "veto_penalty_factor": params.veto_penalty_factor,
            "min_participation_threshold": params.min_participation_threshold
        }
        
        # Select parameters to mutate (1-3 parameters)
        n_params_to_mutate = np.random.randint(1, 4)
        params_to_mutate = np.random.choice(list(current_values.keys()), n_params_to_mutate, replace=False)
        
        # Mutate selected parameters
        for param in params_to_mutate:
            min_val = self.min_values.get(param, 0.0)
            max_val = self.max_values.get(param, 1.0)
            
            # Current value
            current = current_values[param]
            
            # Calculate new value with Gaussian mutation
            delta = np.random.normal(0, (max_val - min_val) * self.learning_rate)
            new_val = current + delta
            
            # Keep within bounds
            new_val = max(min_val, min(max_val, new_val))
            current_values[param] = new_val
        
        # Create new IncentiveParameters object with mutated values
        new_params = IncentiveParameters(
            w1_delegation=current_values["w1_delegation"],
            w2_participation=current_values["w2_participation"],
            w3_successful_votes=current_values["w3_successful_votes"],
            w4_veto_penalty=current_values["w4_veto_penalty"],
            w5_decentralization=current_values["w5_decentralization"],
            w6_peer_evaluation=current_values["w6_peer_evaluation"],
            w7_community_engagement=current_values["w7_community_engagement"],
            total_reward_per_epoch=params.total_reward_per_epoch,
            veto_penalty_factor=current_values["veto_penalty_factor"],
            min_participation_threshold=current_values["min_participation_threshold"],
            model_type=params.model_type  # Keep the same model type
        )
        
        return new_params
    
    def _calculate_fitness(self, metrics: Dict) -> float:
        """
        Calculate fitness score based on target metrics.
        
        Higher is better for all metrics except Gini coefficients.
        """
        fitness = 0.0
        
        for metric, weight in self.target_weights.items():
            if metric not in metrics:
                logger.warning(f"Target metric {metric} not found in simulation results")
                continue
            
            metric_value = metrics[metric]
            
            # Handle time series metrics (take final value)
            if isinstance(metric_value, dict):
                final_epoch = max(metric_value.keys())
                metric_value = metric_value[final_epoch]
            
            # Invert Gini coefficients (lower is better)
            if "gini" in metric.lower() or "concentration" in metric.lower():
                metric_value = 1.0 - metric_value
            
            # Add weighted contribution to fitness
            fitness += weight * metric_value
        
        return fitness
    
    def optimize(self, iterations: int = 50, random_seed: Optional[int] = None) -> IncentiveParameters:
        """
        Run optimization process.
        
        Args:
            iterations: Number of iterations to run
            random_seed: Random seed for reproducibility
            
        Returns:
            Optimized incentive parameters
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize with base parameters
        current_params = self._copy_incentive_params(self.base_incentives)
        
        # Run initial simulation to get baseline
        state = run_simulation(
            self.sim_params,
            current_params
        )
        
        metrics = calculate_simulation_metrics(state)
        current_fitness = self._calculate_fitness(metrics)
        
        self.best_params = current_params
        self.best_fitness = current_fitness
        
        # Track history
        self.history.append({
            "iteration": 0,
            "fitness": current_fitness,
            "params": self._params_to_dict(current_params),
            "metrics": {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        })
        
        # Main optimization loop
        for i in tqdm(range(1, iterations + 1), desc="Optimizing incentive parameters"):
            # Create mutated parameters
            new_params = self._mutate_params(current_params)
            
            # Run simulation with new parameters
            state = run_simulation(
                self.sim_params,
                new_params
            )
            
            # Calculate fitness
            metrics = calculate_simulation_metrics(state)
            new_fitness = self._calculate_fitness(metrics)
            
            # Track history
            self.history.append({
                "iteration": i,
                "fitness": new_fitness,
                "params": self._params_to_dict(new_params),
                "metrics": {k: v for k, v in metrics.items() if not isinstance(v, dict)}
            })
            
            # Update best parameters if improved
            if new_fitness > self.best_fitness:
                self.best_fitness = new_fitness
                self.best_params = new_params
                logger.info(f"New best fitness: {new_fitness:.4f} at iteration {i}")
            
            # Accept new parameters with probability based on improvement
            if new_fitness > current_fitness or np.random.random() < self.exploration_rate:
                current_params = new_params
                current_fitness = new_fitness
        
        logger.info(f"Optimization complete. Best fitness: {self.best_fitness:.4f}")
        return self.best_params
    
    def _params_to_dict(self, params: IncentiveParameters) -> Dict[str, float]:
        """Convert parameters to dictionary for storage."""
        return {
            "model_type": params.model_type,
            "w1_delegation": params.w1_delegation,
            "w2_participation": params.w2_participation,
            "w3_successful_votes": params.w3_successful_votes,
            "w4_veto_penalty": params.w4_veto_penalty,
            "w5_decentralization": params.w5_decentralization,
            "w6_peer_evaluation": params.w6_peer_evaluation,
            "w7_community_engagement": params.w7_community_engagement,
            "total_reward_per_epoch": params.total_reward_per_epoch,
            "veto_penalty_factor": params.veto_penalty_factor,
            "min_participation_threshold": params.min_participation_threshold
        }
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as a DataFrame."""
        # Extract parameters and flatten structure
        flat_history = []
        for entry in self.history:
            flat_entry = {"iteration": entry["iteration"], "fitness": entry["fitness"]}
            # Add parameters
            for k, v in entry["params"].items():
                flat_entry[f"param_{k}"] = v
            # Add metrics
            for k, v in entry["metrics"].items():
                flat_entry[f"metric_{k}"] = v
            flat_history.append(flat_entry)
        
        return pd.DataFrame(flat_history)
    
    def visualize_optimization(self, output_path: str = "optimization_history.png"):
        """
        Visualize the optimization process.
        
        Args:
            output_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        
        history_df = self.get_optimization_history()
        
        # Create a figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot fitness over iterations
        ax1.plot(history_df["iteration"], history_df["fitness"], 'b-', linewidth=2)
        ax1.set_ylabel("Fitness Score")
        ax1.set_title("Incentive Parameter Optimization Progress")
        ax1.grid(True, alpha=0.3)
        
        # Plot parameter values over iterations
        param_cols = [col for col in history_df.columns if col.startswith("param_w")]
        for col in param_cols:
            param_name = col.replace("param_", "")
            ax2.plot(history_df["iteration"], history_df[col], label=param_name)
        
        ax2.set_ylabel("Parameter Values")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
        
        # Plot other parameters
        other_param_cols = [col for col in history_df.columns 
                           if col.startswith("param_") and not col.startswith("param_w")]
        for col in other_param_cols:
            param_name = col.replace("param_", "")
            if param_name != "model_type" and param_name != "total_reward_per_epoch":  # Skip non-numeric
                ax3.plot(history_df["iteration"], history_df[col], label=param_name)
        
        ax3.set_ylabel("Other Parameters")
        ax3.set_xlabel("Iteration")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close() 