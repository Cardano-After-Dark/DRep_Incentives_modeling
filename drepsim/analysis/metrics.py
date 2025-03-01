"""
Metrics calculation for analyzing simulation results.

This module provides functions for calculating various metrics from
simulation results to evaluate the effectiveness of incentive structures.
"""

import numpy as np
from typing import Dict, List, Any
from drepsim.core.models import SimulationState
from drepsim.core.simulation import calculate_gini_coefficient


def calculate_simulation_metrics(state: SimulationState) -> Dict[str, Any]:
    """
    Calculate overall metrics for the simulation.
    
    Args:
        state: Final simulation state
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Calculate basic metrics
    total_actions = len(state.actions)
    accepted_actions = sum(1 for a in state.actions.values() if a.accepted)
    vetoed_actions = sum(1 for a in state.actions.values() if a.vetoed)
    
    metrics['total_actions'] = total_actions
    metrics['accepted_actions'] = accepted_actions
    metrics['vetoed_actions'] = vetoed_actions
    metrics['action_acceptance_rate'] = accepted_actions / total_actions if total_actions > 0 else 0
    metrics['action_veto_rate'] = vetoed_actions / total_actions if total_actions > 0 else 0
    
    # DRep type performance metrics
    drep_types = {}
    for drep_id, drep in state.dreps.items():
        drep_type = drep.profile.drep_type
        if drep_type not in drep_types:
            drep_types[drep_type] = []
        drep_types[drep_type].append(drep_id)
    
    for drep_type, drep_ids in drep_types.items():
        # Average rewards by type
        type_rewards = []
        for epoch in range(state.current_epoch):
            if epoch in state.rewards:
                epoch_rewards = [
                    state.rewards[epoch].get(drep_id, 0) 
                    for drep_id in drep_ids
                ]
                type_rewards.extend(epoch_rewards)
        
        metrics[f'avg_reward_{drep_type}'] = np.mean(type_rewards) if type_rewards else 0
        
        # Final delegation share by type
        type_delegation = sum(state.dreps[drep_id].delegated_ada for drep_id in drep_ids)
        total_delegation = sum(drep.delegated_ada for drep in state.dreps.values())
        metrics[f'delegation_share_{drep_type}'] = (
            type_delegation / total_delegation if total_delegation > 0 else 0
        )
    
    # Delegation concentration (Gini coefficient)
    final_epoch = state.current_epoch - 1
    metrics['final_gini'] = state.delegation_concentration.get(final_epoch, 0)
    metrics['avg_gini'] = np.mean(list(state.delegation_concentration.values()))
    metrics['gini_trend'] = state.delegation_concentration.get(final_epoch, 0) - state.delegation_concentration.get(0, 0)
    
    # Participation metrics
    participation_values = [
        np.mean(list(rates.values())) 
        for epoch, rates in state.participation_rates.items()
    ]
    metrics['avg_participation'] = np.mean(participation_values) if participation_values else 0
    metrics['final_participation'] = participation_values[-1] if participation_values else 0
    
    # Research effort metrics
    research_values = [
        np.mean(list(efforts.values())) 
        for epoch, efforts in state.research_efforts.items()
    ]
    metrics['avg_research_effort'] = np.mean(research_values) if research_values else 0
    
    # Reward distribution metrics
    all_rewards = []
    for epoch in range(state.current_epoch):
        if epoch in state.rewards:
            all_rewards.extend(list(state.rewards[epoch].values()))
    
    metrics['reward_gini'] = calculate_gini_coefficient(all_rewards) if all_rewards else 0
    
    return metrics


def calculate_drep_performance_metrics(state: SimulationState, drep_id: str) -> Dict[str, Any]:
    """
    Calculate detailed performance metrics for a specific DRep.
    
    Args:
        state: Simulation state
        drep_id: ID of the DRep to analyze
        
    Returns:
        Dictionary of metrics for the DRep
    """
    if drep_id not in state.dreps:
        return {}
    
    drep = state.dreps[drep_id]
    metrics = {}
    
    # Basic information
    metrics['type'] = drep.profile.drep_type
    metrics['final_delegation'] = drep.delegated_ada
    
    # Delegation history
    metrics['delegation_history'] = drep.delegation_history
    
    # Participation metrics
    participation_rates = [
        state.participation_rates.get(epoch, {}).get(drep_id, 0)
        for epoch in range(state.current_epoch)
    ]
    metrics['avg_participation'] = np.mean(participation_rates) if participation_rates else 0
    
    # Research effort metrics
    research_efforts = [
        state.research_efforts.get(epoch, {}).get(drep_id, 0)
        for epoch in range(state.current_epoch)
    ]
    metrics['avg_research_effort'] = np.mean(research_efforts) if research_efforts else 0
    
    # Reward metrics
    rewards = [
        state.rewards.get(epoch, {}).get(drep_id, 0)
        for epoch in range(state.current_epoch)
    ]
    metrics['total_rewards'] = sum(rewards)
    metrics['avg_reward'] = np.mean(rewards) if rewards else 0
    metrics['reward_history'] = rewards
    
    # Voting metrics
    votes = [v for v in drep.vote_history]
    metrics['total_votes'] = len(votes)
    
    # Calculate vote success rate
    successful_votes = sum(
        1 for v in votes 
        if state.actions.get(v.action_id) and 
        ((state.actions[v.action_id].accepted and v.choice == "Yes") or
         (not state.actions[v.action_id].accepted and v.choice == "No"))
    )
    metrics['vote_success_rate'] = successful_votes / len(votes) if votes else 0
    
    # Peer evaluation metrics
    received_evals = drep.peer_evaluations_received
    avg_score = np.mean([e.score for e in received_evals]) if received_evals else 0
    metrics['avg_peer_score'] = avg_score
    
    return metrics


def calculate_epoch_metrics(state: SimulationState, epoch: int) -> Dict[str, Any]:
    """
    Calculate detailed metrics for a specific epoch.
    
    Args:
        state: Simulation state
        epoch: Epoch to analyze
        
    Returns:
        Dictionary of metrics for the epoch
    """
    if epoch < 0 or epoch >= state.current_epoch:
        return {}
    
    metrics = {}
    
    # Delegation metrics
    delegations = [
        drep.delegation_history[epoch] 
        for drep in state.dreps.values() 
        if epoch < len(drep.delegation_history)
    ]
    metrics['total_delegation'] = sum(delegations)
    metrics['delegation_gini'] = state.delegation_concentration.get(epoch, 0)
    
    # Participation metrics
    participation_rates = state.participation_rates.get(epoch, {})
    metrics['avg_participation'] = np.mean(list(participation_rates.values())) if participation_rates else 0
    
    # Research effort metrics
    research_efforts = state.research_efforts.get(epoch, {})
    metrics['avg_research_effort'] = np.mean(list(research_efforts.values())) if research_efforts else 0
    
    # Reward metrics
    rewards = state.rewards.get(epoch, {})
    metrics['total_rewards'] = sum(rewards.values())
    metrics['reward_gini'] = calculate_gini_coefficient(list(rewards.values())) if rewards else 0
    
    # Action metrics
    active_actions = state.active_actions.get(epoch, [])
    metrics['active_actions'] = len(active_actions)
    
    # Vote metrics
    votes = state.votes.get(epoch, [])
    metrics['total_votes'] = len(votes)
    
    # DRep type metrics
    drep_types = {}
    for drep_id, drep in state.dreps.items():
        drep_type = drep.profile.drep_type
        if drep_type not in drep_types:
            drep_types[drep_type] = []
        drep_types[drep_type].append(drep_id)
    
    for drep_type, drep_ids in drep_types.items():
        # Delegation share by type
        type_delegation = sum(
            drep.delegation_history[epoch] 
            for drep_id in drep_ids 
            for drep in [state.dreps[drep_id]]
            if epoch < len(drep.delegation_history)
        )
        total_delegation = sum(delegations)
        metrics[f'delegation_share_{drep_type}'] = (
            type_delegation / total_delegation if total_delegation > 0 else 0
        )
        
        # Participation by type
        type_participation = [
            participation_rates.get(drep_id, 0) for drep_id in drep_ids
        ]
        metrics[f'participation_{drep_type}'] = np.mean(type_participation) if type_participation else 0
        
        # Rewards by type
        type_rewards = [rewards.get(drep_id, 0) for drep_id in drep_ids]
        metrics[f'rewards_{drep_type}'] = np.mean(type_rewards) if type_rewards else 0
    
    return metrics 