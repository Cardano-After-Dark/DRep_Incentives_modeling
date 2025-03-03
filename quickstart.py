#!/usr/bin/env python3
"""
Quickstart script for the DRep Incentivization Simulation Framework.
This script creates a virtual environment, installs dependencies,
and sets up initial configuration files.
"""

import os
import sys
import subprocess
import json
import platform
from pathlib import Path

# Configuration files content
SIM_CONFIG = {
    "total_epochs": 20,
    "drep_count": 100,
    "governance_actions_per_epoch": [5, 15],
    "total_ada_delegated": 1000000000,
    "delegation_change_rate": 0.05,
    "veto_probability": 0.05,
    "drep_type_distribution": {
        "Professional": 0.2,
        "Hobbyist": 0.4,
        "Passive": 0.3,
        "Strategic": 0.1
    },
    "initial_delegation_distribution": "pareto",
    "delegation_distribution_params": {"alpha": 1.5},
    "peer_evaluation_sample_size": 5,
    "random_seed": 42
}

INCENTIVE_CONFIG = {
    "w1_delegation": 0.2,
    "w2_participation": 0.3,
    "w3_successful_votes": 0.2,
    "w4_veto_penalty": 0.05,
    "w5_decentralization": 0.1,
    "w6_peer_evaluation": 0.1,
    "w7_community_engagement": 0.05,
    "total_reward_per_epoch": 100000,
    "veto_penalty_factor": 0.5,
    "min_participation_threshold": 0.1
}

def print_step(message):
    """Print a step message with formatting."""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)

def run_command(command, shell=False):
    """Run a shell command and print output."""
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            check=True, 
            text=True, 
            capture_output=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def create_virtual_env():
    """Create a virtual environment in .venv directory."""
    print_step("Creating virtual environment in .venv directory")
    return run_command([sys.executable, "-m", "venv", ".venv"])

def get_activate_command():
    """Get the appropriate activate command based on the OS."""
    if platform.system() == "Windows":
        return [".venv\\Scripts\\activate"]
    else:
        return ["source", ".venv/bin/activate"]

def install_package():
    """Install the package in development mode."""
    print_step("Installing the package in development mode")
    
    activate_cmd = get_activate_command()
    if platform.system() == "Windows":
        install_cmd = f"{activate_cmd[0]} && pip install -e ."
    else:
        install_cmd = f"{activate_cmd[0]} {activate_cmd[1]} && pip install -e ."
    
    return run_command(install_cmd, shell=True)

def create_config_files():
    """Create configuration files in the configs directory."""
    print_step("Creating configuration files")
    
    # Create configs directory if it doesn't exist
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Write simulation config
    with open(configs_dir / "sim_config.json", "w") as f:
        json.dump(SIM_CONFIG, f, indent=2)
    
    # Write incentive config
    with open(configs_dir / "incentive_config.json", "w") as f:
        json.dump(INCENTIVE_CONFIG, f, indent=2)
    
    print("Configuration files created in the 'configs' directory.")
    return True

def verify_installation():
    """Verify that the installation was successful."""
    print_step("Verifying installation")
    
    activate_cmd = get_activate_command()
    if platform.system() == "Windows":
        # Use pip list instead of the CLI command
        verify_cmd = f"{activate_cmd[0]} && pip list | findstr drepsim"
    else:
        # Use pip list instead of the CLI command
        verify_cmd = f"{activate_cmd[0]} {activate_cmd[1]} && pip list | grep drepsim"
    
    success = run_command(verify_cmd, shell=True)
    
    if success:
        print("Package successfully installed!")
    else:
        print("Warning: Could not verify package installation.")
    
    return success

def print_next_steps():
    """Print instructions for next steps."""
    print_step("Setup Complete!")
    
    activate_cmd = get_activate_command()
    if platform.system() == "Windows":
        activate_str = activate_cmd[0]
    else:
        activate_str = f"{activate_cmd[0]} {activate_cmd[1]}"
    
    print(f"""
To activate the virtual environment:
    {activate_str}

To run an example script:
    python examples/basic_simulation.py

To run a Monte Carlo simulation:
    python examples/monte_carlo_analysis.py

Simulation results will be saved in the 'simulation_results' directory.
    """)

def main():
    """Main function to set up the environment."""
    print("DRep Incentivization Simulation Framework Setup")
    print("----------------------------------------------")
    
    # Check if Python version is compatible
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        return False
    
    # Create virtual environment
    if not create_virtual_env():
        return False
    
    # Install package
    if not install_package():
        return False
    
    # Create config files
    if not create_config_files():
        return False
    
    # Verify installation
    if not verify_installation():
        print("Warning: Installation verification failed, but setup may still be partially complete.")
    
    # Print next steps
    print_next_steps()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 