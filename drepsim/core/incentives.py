"""
Incentive calculation module for the DRep simulation framework.

This module contains functions for calculating rewards based on DRep performance
and the specified incentive model.
"""

import numpy as np
from typing import Dict, Tuple, List
from drepsim.core.models import SimulationState, IncentiveParameters


def calculate_rewards(
    state: SimulationState,
    incentives: IncentiveParameters,
    epoch: int
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate reward scores and ADA rewards for all DReps based on their performance.
    
    Mathematical Model:
    ------------------
    For each DRep i:
    
    delegation_ratio_i = delegated_ada_i / total_delegated_ada
    
    participation_rate_i = actions_voted_i / total_active_actions
    
    successful_votes_ratio_i = successful_votes_i / total_votes_i
    
    decentralization_score_i = 1 - 0.3*is_cc_member - 0.5*is_spo
    
    peer_evaluation_score_i = average of evaluation scores received
    
    community_engagement_score_i = measure of community involvement
    
    veto_penalty_i = number_of_vetoed_votes * veto_penalty_factor
    
    DRep_Score_i = w₁ × delegation_ratio_i + 
                   w₂ × participation_rate_i + 
                   w₃ × successful_votes_ratio_i - 
                   w₄ × veto_penalty_i + 
                   w₅ × decentralization_score_i + 
                   w₆ × peer_evaluation_score_i + 
                   w₇ × community_engagement_score_i
    
    DRep_Reward_i = (DRep_Score_i / Sum_of_All_DRep_Scores) × Total_Reward_Pool
    
    Args:
        state: The current simulation state
        incentives: Incentive parameters
        epoch: Current epoch
        
    Returns:
        Tuple of (scores, rewards) dictionaries mapping DRep IDs to values
    """
    scores = {}
    rewards = {}
    
    # Skip if no DReps or no rewards
    if not state.dreps or incentives.total_reward_per_epoch <= 0:
        return scores, rewards
    
    # Get total delegated ADA
    total_delegated = sum(drep.delegated_ada for drep in state.dreps.values())
    
    # Calculate scores for each DRep
    for drep_id, drep in state.dreps.items():
        # 1. Delegation ratio component
        delegation_ratio = drep.delegated_ada / total_delegated if total_delegated > 0 else 0
        delegation_component = incentives.w1_delegation * delegation_ratio
        
        # 2. Participation rate component
        participation_rate = state.participation_rates.get(epoch, {}).get(drep_id, 0)
        participation_component = incentives.w2_participation * participation_rate
        
        # 3. Successful votes component
        # Count votes that aligned with ultimately accepted actions
        drep_votes = [v for v in state.votes.get(epoch, []) if v.drep_id == drep_id]
        total_votes = len(drep_votes)
        
        successful_votes = 0
        for vote in drep_votes:
            action = state.actions.get(vote.action_id)
            if action and action.accepted and vote.choice == "Yes":
                successful_votes += 1
            elif action and not action.accepted and vote.choice == "No":
                successful_votes += 1
        
        successful_votes_ratio = successful_votes / total_votes if total_votes > 0 else 0
        successful_component = incentives.w3_successful_votes * successful_votes_ratio
        
        # 4. Veto penalty component
        vetoed_votes = 0
        for vote in drep_votes:
            action = state.actions.get(vote.action_id)
            if action and action.vetoed and vote.choice == "Yes":
                vetoed_votes += 1
        
        veto_penalty = vetoed_votes * incentives.veto_penalty_factor
        veto_component = -incentives.w4_veto_penalty * veto_penalty  # Negative component
        
        # 5. Decentralization score component
        decentralization_score = 1.0
        if drep.profile.constitutional_committee:
            decentralization_score -= 0.3
        if drep.profile.stake_pool_operator:
            decentralization_score -= 0.5
        
        decentralization_component = incentives.w5_decentralization * decentralization_score
        
        # 6. Peer evaluation component
        peer_score = state.peer_scores.get(epoch, {}).get(drep_id, 0)
        peer_component = incentives.w6_peer_evaluation * peer_score
        
        # 7. Community engagement component
        engagement_score = state.community_engagements.get(epoch, {}).get(drep_id, 0)
        engagement_component = incentives.w7_community_engagement * engagement_score
        
        # Calculate total score
        total_score = (
            delegation_component +
            participation_component +
            successful_component +
            veto_component +
            decentralization_component +
            peer_component +
            engagement_component
        )
        
        # Apply minimum participation threshold
        if participation_rate < incentives.min_participation_threshold:
            total_score = 0
        
        scores[drep_id] = max(0, total_score)  # Ensure non-negative
    
    # Calculate rewards based on scores
    total_score = sum(scores.values())
    
    if total_score > 0:
        for drep_id, score in scores.items():
            reward_share = score / total_score
            rewards[drep_id] = reward_share * incentives.total_reward_per_epoch
    
    return scores, rewards 