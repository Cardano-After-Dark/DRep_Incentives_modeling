"""
Tests for the incentives module.
"""

import pytest
import numpy as np

from drepsim.core.models import (
    SimulationParameters, IncentiveParameters, 
    DRepProfile, DRepState, GovernanceAction, DRepVote,
    SimulationState
)
from drepsim.core.incentives import calculate_rewards


def test_calculate_rewards_basic():
    """Test basic reward calculation with simple inputs."""
    # Create a minimal simulation state
    drep1 = DRepState(
        profile=DRepProfile(
            id="drep_0001",
            type="Professional",
            base_participation_rate=0.9,
            base_research_capacity=0.8,
            base_community_engagement=0.7,
            adaptation_rate=0.5,
            profit_motivation=0.6,
            community_motivation=0.7,
            reputation_motivation=0.8,
            region="North America",
            stake_pool_operator=False,
            constitutional_committee=False
        ),
        delegated_ada=5000,
        delegation_history=[5000]
    )
    
    drep2 = DRepState(
        profile=DRepProfile(
            id="drep_0002",
            type="Passive",
            base_participation_rate=0.3,
            base_research_capacity=0.2,
            base_community_engagement=0.1,
            adaptation_rate=0.2,
            profit_motivation=0.3,
            community_motivation=0.2,
            reputation_motivation=0.1,
            region="Europe",
            stake_pool_operator=True,
            constitutional_committee=False
        ),
        delegated_ada=5000,
        delegation_history=[5000]
    )
    
    # Create a governance action
    action = GovernanceAction(
        id="action_001",
        title="Test Action",
        description="A test governance action",
        complexity=0.5,
        community_impact=0.7,
        creation_epoch=0,
        voting_periods=1,
        category="Technical",
        accepted=True
    )
    
    # Create votes
    vote1 = DRepVote(
        drep_id="drep_0001",
        action_id="action_001",
        epoch=0,
        choice="Yes",
        research_effort=0.8
    )
    
    vote2 = DRepVote(
        drep_id="drep_0002",
        action_id="action_001",
        epoch=0,
        choice="No",
        research_effort=0.2
    )
    
    # Create simulation state
    state = SimulationState(
        dreps={"drep_0001": drep1, "drep_0002": drep2},
        actions={"action_001": action},
        current_epoch=1,
        active_actions={0: ["action_001"]},
        votes={0: [vote1, vote2]},
        participation_rates={0: {"drep_0001": 1.0, "drep_0002": 1.0}},
        research_efforts={0: {"drep_0001": 0.8, "drep_0002": 0.2}},
        peer_scores={0: {"drep_0001": 0.9, "drep_0002": 0.3}},
        community_engagements={0: {"drep_0001": 0.7, "drep_0002": 0.1}}
    )
    
    # Create incentive parameters
    incentives = IncentiveParameters(
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
    
    # Calculate rewards
    scores, rewards = calculate_rewards(state, incentives, 0)
    
    # Check that both DReps received scores
    assert "drep_0001" in scores
    assert "drep_0002" in scores
    
    # Check that both DReps received rewards
    assert "drep_0001" in rewards
    assert "drep_0002" in rewards
    
    # Check that the professional DRep got a higher score than the passive one
    assert scores["drep_0001"] > scores["drep_0002"]
    
    # Check that rewards sum to the total reward pool
    assert np.isclose(sum(rewards.values()), incentives.total_reward_per_epoch)


def test_calculate_rewards_zero_participation():
    """Test that DReps with zero participation get zero rewards."""
    # Create a minimal simulation state with one participating and one non-participating DRep
    drep1 = DRepState(
        profile=DRepProfile(
            id="drep_0001",
            type="Professional",
            base_participation_rate=0.9,
            base_research_capacity=0.8,
            base_community_engagement=0.7,
            adaptation_rate=0.5,
            profit_motivation=0.6,
            community_motivation=0.7,
            reputation_motivation=0.8,
            region="North America",
            stake_pool_operator=False,
            constitutional_committee=False
        ),
        delegated_ada=5000,
        delegation_history=[5000]
    )
    
    drep2 = DRepState(
        profile=DRepProfile(
            id="drep_0002",
            type="Passive",
            base_participation_rate=0.3,
            base_research_capacity=0.2,
            base_community_engagement=0.1,
            adaptation_rate=0.2,
            profit_motivation=0.3,
            community_motivation=0.2,
            reputation_motivation=0.1,
            region="Europe",
            stake_pool_operator=True,
            constitutional_committee=False
        ),
        delegated_ada=5000,
        delegation_history=[5000]
    )
    
    # Create a governance action
    action = GovernanceAction(
        id="action_001",
        title="Test Action",
        description="A test governance action",
        complexity=0.5,
        community_impact=0.7,
        creation_epoch=0,
        voting_periods=1,
        category="Technical",
        accepted=True
    )
    
    # Create vote only for drep1
    vote1 = DRepVote(
        drep_id="drep_0001",
        action_id="action_001",
        epoch=0,
        choice="Yes",
        research_effort=0.8
    )
    
    # Create simulation state
    state = SimulationState(
        dreps={"drep_0001": drep1, "drep_0002": drep2},
        actions={"action_001": action},
        current_epoch=1,
        active_actions={0: ["action_001"]},
        votes={0: [vote1]},
        participation_rates={0: {"drep_0001": 1.0, "drep_0002": 0.0}},
        research_efforts={0: {"drep_0001": 0.8, "drep_0002": 0.0}},
        peer_scores={0: {"drep_0001": 0.9, "drep_0002": 0.3}},
        community_engagements={0: {"drep_0001": 0.7, "drep_0002": 0.1}}
    )
    
    # Create incentive parameters
    incentives = IncentiveParameters(
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
    
    # Calculate rewards
    scores, rewards = calculate_rewards(state, incentives, 0)
    
    # Check that both DReps received scores
    assert "drep_0001" in scores
    assert "drep_0002" in scores
    
    # Check that only the participating DRep received rewards
    assert rewards["drep_0001"] > 0
    assert rewards["drep_0002"] == 0
    
    # Check that rewards sum to the total reward pool
    assert np.isclose(sum(rewards.values()), incentives.total_reward_per_epoch) 