"""
Tests for the simulation module.
"""

import pytest
import numpy as np

from drepsim.core.models import (
    SimulationParameters, IncentiveParameters, 
    DRepProfile, DRepState, GovernanceAction
)
from drepsim.core.simulation import (
    initialize_simulation, run_epoch, calculate_gini_coefficient
)


def test_initialize_simulation():
    """Test that simulation initialization creates the expected state."""
    params = SimulationParameters(
        total_epochs=10,
        drep_count=50,
        governance_actions_per_epoch=(3, 8),
        total_ada_delegated=1000000,
        delegation_change_rate=0.05,
        veto_probability=0.05,
        drep_type_distribution={
            "Professional": 0.2,
            "Hobbyist": 0.4,
            "Passive": 0.3,
            "Strategic": 0.1
        },
        initial_delegation_distribution="uniform",
        delegation_distribution_params={},
        peer_evaluation_sample_size=3,
        random_seed=42
    )
    
    state = initialize_simulation(params)
    
    # Check that the correct number of DReps were created
    assert len(state.dreps) == params.drep_count
    
    # Check that all ADA was allocated
    total_delegated = sum(drep.delegated_ada for drep in state.dreps.values())
    assert np.isclose(total_delegated, params.total_ada_delegated)
    
    # Check that DRep types were allocated according to distribution
    type_counts = {}
    for drep in state.dreps.values():
        drep_type = drep.profile.type
        type_counts[drep_type] = type_counts.get(drep_type, 0) + 1
    
    for drep_type, expected_fraction in params.drep_type_distribution.items():
        expected_count = int(expected_fraction * params.drep_count)
        # Allow for some randomness in allocation
        assert abs(type_counts.get(drep_type, 0) - expected_count) <= 5


def test_gini_coefficient():
    """Test the Gini coefficient calculation."""
    # Equal distribution should have Gini = 0
    equal = [100, 100, 100, 100, 100]
    assert np.isclose(calculate_gini_coefficient(equal), 0)
    
    # Complete inequality should have Gini = 1 - 1/n
    unequal = [0, 0, 0, 0, 100]
    assert np.isclose(calculate_gini_coefficient(unequal), 0.8)
    
    # Test a known distribution
    values = [10, 20, 30, 40, 50]
    # The expected Gini for this distribution is 0.3
    assert 0.25 <= calculate_gini_coefficient(values) <= 0.35


def test_run_epoch():
    """Test that running an epoch updates the simulation state correctly."""
    # Create a minimal simulation
    sim_params = SimulationParameters(
        total_epochs=5,
        drep_count=10,
        governance_actions_per_epoch=(2, 4),
        total_ada_delegated=100000,
        delegation_change_rate=0.05,
        veto_probability=0.05,
        drep_type_distribution={
            "Professional": 0.25,
            "Hobbyist": 0.25,
            "Passive": 0.25,
            "Strategic": 0.25
        },
        initial_delegation_distribution="uniform",
        delegation_distribution_params={},
        peer_evaluation_sample_size=2,
        random_seed=42
    )
    
    incentive_params = IncentiveParameters(
        w1_delegation=0.2,
        w2_participation=0.3,
        w3_successful_votes=0.2,
        w4_veto_penalty=0.05,
        w5_decentralization=0.1,
        w6_peer_evaluation=0.1,
        w7_community_engagement=0.05,
        total_reward_per_epoch=1000,
        veto_penalty_factor=0.5,
        min_participation_threshold=0.1
    )
    
    state = initialize_simulation(sim_params)
    initial_epoch = state.current_epoch
    
    # Run one epoch
    new_state = run_epoch(state, incentive_params, sim_params)
    
    # Check that epoch was incremented
    assert new_state.current_epoch == initial_epoch + 1
    
    # Check that governance actions were created
    assert len(new_state.active_actions.get(initial_epoch, [])) > 0
    
    # Check that votes were recorded
    assert len(new_state.votes.get(initial_epoch, [])) > 0
    
    # Check that rewards were calculated
    assert len(new_state.rewards.get(initial_epoch, {})) > 0
    
    # Check that delegation history was updated
    for drep in new_state.dreps.values():
        assert len(drep.delegation_history) > 0