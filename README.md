# DRep Incentivization Simulation Framework

A framework for simulating and analyzing incentive structures for Delegated Representatives (DReps) in blockchain governance ecosystems. This package combines Monte Carlo methods with agent-based modeling to evaluate different incentive mechanisms and their impact on governance outcomes.

## 🚀 Overview

The DRep Simulation Framework allows researchers, governance designers, and blockchain projects to model, test, and optimize different incentive structures for delegated governance systems. It simulates how different types of representatives behave under various incentive models and analyzes the resulting governance outcomes.

## ✨ Features

- **Agent-based modeling** of different DRep types (Professional, Hobbyist, Passive, Strategic)
- **Multiple incentive models** including Linear, Non-Linear, and Temporal reward distribution
- **Monte Carlo simulation** for parameter space exploration
- **Reinforcement learning** for incentive parameter optimization
- **Comprehensive metrics** for analyzing governance outcomes
- **Visualization tools** for simulation results
- **Command-line interface** for easy simulation execution

## 📋 Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## 🔧 Installation

### Option 1: Using quickstart script (recommended)

```bash
# Clone the repository
git clone https://github.com/Cardano-After-Dark/DRep_Incentives_modeling.git
cd drepsim

# Run the quickstart script
python quickstart.py
```

### Option 2: Manual installation

```bash
# Clone the repository
git clone https://github.com/username/drepsim.git
cd drepsim

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## 🏃‍♂️ Quick Start

### Running a basic simulation

```bash
# Run a simulation using the CLI
python -m drepsim simulate --sim-config examples/configs/basic_sim.json --incentive-config examples/configs/linear_incentives.json --output-dir results --visualize
```

### Running a Monte Carlo simulation

```bash
# Run a Monte Carlo simulation
python -m drepsim monte-carlo --sim-config examples/configs/monte_carlo_sim.json --param-ranges examples/configs/param_ranges.json --output-dir monte_carlo_results
```

### Using the Python API

```python
from drepsim.core.models import SimulationParameters, IncentiveParameters
from drepsim.core.simulation import run_simulation
from drepsim.visualization.plots import visualize_simulation_results

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

# Run the simulation
final_state = run_simulation(sim_params, incentive_params)

# Visualize results
visualize_simulation_results(final_state, incentive_params, "visualizations")
```

## 📊 Example Outputs

The simulation produces a variety of visualizations to help analyze results:

- Participation rates over time
- Rewards distribution by DRep type
- Delegation share by DRep type
- Final delegation distribution
- Governance outcome metrics

## 🏗️ Architecture

The framework is organized into several key components:

- **Core Models**: Data classes representing entities like DReps, governance actions, and votes
- **Simulation Engine**: Manages the progression of epochs and simulation state
- **Incentive Models**: Different reward calculation strategies (Linear, NonLinear, Temporal)
- **Analysis Tools**: Monte Carlo simulation and reinforcement learning optimization
- **Visualization**: Functions to generate plots and visualizations
- **CLI Interface**: Command-line interface for easy execution

![Architecture Diagram](architecture.md)

## 📘 Documentation

For detailed documentation, see the `docs/` directory:

- [Core Concepts](docs/core_concepts.md)
- [Incentive Models](docs/incentive_models.md)
- [Simulation Parameters](docs/simulation_parameters.md)
- [API Reference](docs/api_reference.md)
- [Example Scenarios](docs/example_scenarios.md)

## 🔬 Use Cases

- Testing different incentive structures for Delegated Proof of Stake (DPoS) systems
- Optimizing reward parameters for blockchain governance
- Analyzing the impact of different DRep distributions
- Exploring how incentives affect decentralization metrics
- Evaluating governance participation and decision quality

## 🔄 Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/username/drepsim.git
cd drepsim

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .

# Run tests
pytest
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

This simulation framework is inspired by real-world governance challenges in blockchain ecosystems, particularly the Cardano blockchain's Voltaire governance era.