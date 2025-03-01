import argparse
import json
import logging
import os
import sys
from typing import Dict, Any

from drepsim.core.models import SimulationParameters, IncentiveParameters
from drepsim.core.simulation import run_simulation
from drepsim.core.monte_carlo import run_monte_carlo_simulation, analyze_monte_carlo_results, save_monte_carlo_results
from drepsim.visualization.simulation import visualize_simulation_results
from drepsim.visualization.monte_carlo import visualize_monte_carlo_results
from drepsim.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="DRep Incentivization Simulation Framework")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run a single simulation
    sim_parser = subparsers.add_parser("simulate", help="Run a single simulation")
    sim_parser.add_argument(
        "--sim-config", 
        required=True, 
        help="Path to simulation configuration file"
    )
    sim_parser.add_argument(
        "--incentive-config", 
        required=True, 
        help="Path to incentive configuration file"
    )
    sim_parser.add_argument(
        "--output-dir", 
        default="simulation_results", 
        help="Directory to save results"
    )
    sim_parser.add_argument(
        "--visualize", 
        action="store_true", 
        help="Generate visualizations"
    )
    
    # Run a Monte Carlo simulation
    mc_parser = subparsers.add_parser("monte-carlo", help="Run a Monte Carlo simulation")
    mc_parser.add_argument(
        "--sim-config", 
        required=True, 
        help="Path to simulation configuration file"
    )
    mc_parser.add_argument(
        "--param-ranges", 
        required=True, 
        help="Path to parameter ranges configuration file"
    )
    mc_parser.add_argument(
        "--iterations", 
        type=int, 
        required=True, 
        help="Number of Monte Carlo iterations"
    )
    mc_parser.add_argument(
        "--epochs", 
        type=int, 
        required=True, 
        help="Number of epochs per iteration"
    )
    mc_parser.add_argument(
        "--parallel", 
        type=int, 
        default=1, 
        help="Number of parallel processes"
    )
    mc_parser.add_argument(
        "--output-dir", 
        default="monte_carlo_results", 
        help="Directory to save results"
    )
    mc_parser.add_argument(
        "--save-file", 
        default="monte_carlo_results.pkl", 
        help="File to save raw results"
    )
    mc_parser.add_argument(
        "--visualize", 
        action="store_true", 
        help="Generate visualizations"
    )
    
    # Analyze existing Monte Carlo results
    analyze_parser = subparsers.add_parser("analyze", help="Analyze existing Monte Carlo results")
    analyze_parser.add_argument(
        "--results-file", 
        required=True, 
        help="Path to saved Monte Carlo results file"
    )
    analyze_parser.add_argument(
        "--output-dir", 
        default="monte_carlo_analysis", 
        help="Directory to save analysis results"
    )
    
    # Visualize existing simulation results
    viz_parser = subparsers.add_parser("visualize", help="Visualize existing simulation results")
    viz_parser.add_argument(
        "--results-file", 
        required=True, 
        help="Path to saved simulation results file"
    )
    viz_parser.add_argument(
        "--output-dir", 
        default="visualizations", 
        help="Directory to save visualizations"
    )
    
    # Common options
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO", 
        help="Set logging level"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the configuration file is not valid JSON
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{config_path}' is not valid JSON")
        sys.exit(1)


def create_simulation_parameters(config: Dict[str, Any]) -> SimulationParameters:
    """
    Create a SimulationParameters object from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SimulationParameters object
    """
    return SimulationParameters(
        total_epochs=config.get("total_epochs", 20),
        drep_count=config.get("drep_count", 100),
        governance_actions_per_epoch=tuple(config.get("governance_actions_per_epoch", (5, 15))),
        total_ada_delegated=config.get("total_ada_delegated", 1000000000),
        delegation_change_rate=config.get("delegation_change_rate", 0.05),
        veto_probability=config.get("veto_probability", 0.05),
        drep_type_distribution=config.get("drep_type_distribution", {
            "Professional": 0.2,
            "Hobbyist": 0.4,
            "Passive": 0.3,
            "Strategic": 0.1
        }),
        initial_delegation_distribution=config.get("initial_delegation_distribution", "pareto"),
        delegation_distribution_params=config.get("delegation_distribution_params", {"alpha": 1.5}),
        peer_evaluation_sample_size=config.get("peer_evaluation_sample_size", 5),
        random_seed=config.get("random_seed", None)
    )


def create_incentive_parameters(config: Dict[str, Any]) -> IncentiveParameters:
    """
    Create an IncentiveParameters object from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        IncentiveParameters object
    """
    return IncentiveParameters(
        w1_delegation=config.get("w1_delegation", 0.2),
        w2_participation=config.get("w2_participation", 0.3),
        w3_successful_votes=config.get("w3_successful_votes", 0.2),
        w4_veto_penalty=config.get("w4_veto_penalty", 0.05),
        w5_decentralization=config.get("w5_decentralization", 0.1),
        w6_peer_evaluation=config.get("w6_peer_evaluation", 0.1),
        w7_community_engagement=config.get("w7_community_engagement", 0.05),
        total_reward_per_epoch=config.get("total_reward_per_epoch", 100000),
        veto_penalty_factor=config.get("veto_penalty_factor", 0.5),
        min_participation_threshold=config.get("min_participation_threshold", 0.1)
    )


def create_parameter_ranges(config: Dict[str, Any]) -> Dict[str, tuple]:
    """
    Create parameter ranges for Monte Carlo simulation from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of parameter ranges
    """
    param_ranges = {}
    
    for param, range_values in config.items():
        if isinstance(range_values, list) and len(range_values) == 2:
            param_ranges[param] = tuple(range_values)
    
    return param_ranges


def run_simulation_command(args):
    """Run a single simulation based on command-line arguments."""
    # Load configurations
    sim_config = load_config(args.sim_config)
    incentive_config = load_config(args.incentive_config)
    
    # Create parameter objects
    sim_params = create_simulation_parameters(sim_config)
    incentive_params = create_incentive_parameters(incentive_config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run simulation
    logging.info("Starting simulation")
    state = run_simulation(sim_params, incentive_params)
    logging.info("Simulation completed")
    
    # Save simulation state
    import pickle
    with open(os.path.join(args.output_dir, "simulation_state.pkl"), "wb") as f:
        pickle.dump(state, f)
    
    # Save parameters
    with open(os.path.join(args.output_dir, "simulation_params.json"), "w") as f:
        json.dump(sim_config, f, indent=2)
    
    with open(os.path.join(args.output_dir, "incentive_params.json"), "w") as f:
        json.dump(incentive_config, f, indent=2)
    
    # Generate visualizations if requested
    if args.visualize:
        logging.info("Generating visualizations")
        visualize_simulation_results(
            state, 
            incentive_params, 
            output_dir=os.path.join(args.output_dir, "visualizations")
        )
    
    logging.info(f"Results saved to {args.output_dir}")


def run_monte_carlo_command(args):
    """Run a Monte Carlo simulation based on command-line arguments."""
    # Load configurations
    sim_config = load_config(args.sim_config)
    param_ranges_config = load_config(args.param_ranges)
    
    # Create parameter objects
    base_sim_params = create_simulation_parameters(sim_config)
    param_ranges = create_parameter_ranges(param_ranges_config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run Monte Carlo simulation
    logging.info(f"Starting Monte Carlo simulation with {args.iterations} iterations")
    results = run_monte_carlo_simulation(
        n_iterations=args.iterations,
        epochs_per_iteration=args.epochs,
        param_ranges=param_ranges,
        base_sim_params=base_sim_params,
        n_parallel=args.parallel,
        save_intermediate=os.path.join(args.output_dir, "intermediate_results.pkl")
    )
    logging.info("Monte Carlo simulation completed")
    
    # Analyze results
    logging.info("Analyzing results")
    analysis = analyze_monte_carlo_results(results)
    
    # Save results
    save_path = os.path.join(args.output_dir, args.save_file)
    save_monte_carlo_results(results, analysis, save_path)
    
    # Save configurations
    with open(os.path.join(args.output_dir, "simulation_params.json"), "w") as f:
        json.dump(sim_config, f, indent=2)
    
    with open(os.path.join(args.output_dir, "parameter_ranges.json"), "w") as f:
        json.dump(param_ranges_config, f, indent=2)
    
    # Generate visualizations if requested
    if args.visualize:
        logging.info("Generating visualizations")
        visualize_monte_carlo_results(
            results, 
            analysis, 
            output_dir=os.path.join(args.output_dir, "visualizations")
        )
    
    logging.info(f"Results saved to {save_path}")


def run_analyze_command(args):
    """Analyze existing Monte Carlo results based on command-line arguments."""
    # Load results
    data = load_monte_carlo_results(args.results_file)
    results = data.get('results', [])
    
    # Analyze results
    logging.info("Analyzing results")
    analysis = analyze_monte_carlo_results(results)
    
    # Save analysis
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "analysis_results.pkl")
    
    with open(save_path, "wb") as f:
        import pickle
        pickle.dump(analysis, f)
    
    # Generate visualizations
    logging.info("Generating visualizations")
    visualize_monte_carlo_results(
        results, 
        analysis, 
        output_dir=args.output_dir
    )
    
    logging.info(f"Analysis results saved to {save_path}")


def run_visualize_command(args):
    """Visualize existing simulation results based on command-line arguments."""
    # Load results
    with open(args.results_file, "rb") as f:
        import pickle
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'results' in data and 'analysis' in data:
        # Monte Carlo results
        logging.info("Visualizing Monte Carlo results")
        visualize_monte_carlo_results(
            data['results'], 
            data['analysis'], 
            output_dir=args.output_dir
        )
    else:
        # Single simulation results
        logging.info("Visualizing simulation results")
        
        # Try to load incentive parameters
        incentive_params_path = os.path.join(os.path.dirname(args.results_file), "incentive_params.json")
        if os.path.exists(incentive_params_path):
            incentive_config = load_config(incentive_params_path)
            incentive_params = create_incentive_parameters(incentive_config)
        else:
            # Use default parameters if not found
            incentive_params = IncentiveParameters()
        
        visualize_simulation_results(
            data, 
            incentive_params, 
            output_dir=args.output_dir
        )
    
    logging.info(f"Visualizations saved to {args.output_dir}")


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Run the appropriate command
    if args.command == "simulate":
        run_simulation_command(args)
    elif args.command == "monte-carlo":
        run_monte_carlo_command(args)
    elif args.command == "analyze":
        run_analyze_command(args)
    elif args.command == "visualize":
        run_visualize_command(args)
    else:
        print("Error: No command specified. Use --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()