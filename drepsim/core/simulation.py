"""
Core simulation engine for the DRep incentivization framework.

This module contains the functions that drive the simulation, including
initialization, epoch processing, and delegation updates.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Callable, Any
from drepsim.core.models import (
    GovernanceAction, DRepVote, PeerEvaluation, 
    DRepProfile, DRepState, IncentiveParameters, 
    SimulationParameters, SimulationState
)
from drepsim.core.incentives import calculate_rewards

logger = logging.getLogger("drepsim.simulation")


def initialize_simulation(params: SimulationParameters) -> SimulationState:
    """
    Initialize a new simulation with the given parameters.
    
    This function creates the initial state of the simulation, including
    generating DReps with different profiles, allocating initial delegations,
    and setting up the tracking structures.
    
    Args:
        params: Simulation parameters
        
    Returns:
        Initial simulation state
        
    Raises:
        ValueError: If any parameters are invalid
    """
    # Validate parameters
    if params.drep_count <= 0:
        raise ValueError("drep_count must be positive")
    if params.total_ada_delegated <= 0:
        raise ValueError("total_ada_delegated must be positive")
    if not (0 <= params.delegation_change_rate <= 1):
        raise ValueError("delegation_change_rate must be between 0 and 1")
    if not (0 <= params.veto_probability <= 1):
        raise ValueError("veto_probability must be between 0 and 1")
    if not isinstance(params.governance_actions_per_epoch, tuple) or len(params.governance_actions_per_epoch) != 2:
        raise ValueError("governance_actions_per_epoch must be a tuple of (min, max)")
    if params.governance_actions_per_epoch[0] > params.governance_actions_per_epoch[1]:
        raise ValueError("governance_actions_per_epoch min must be <= max")
    if params.governance_actions_per_epoch[0] < 0:
        raise ValueError("governance_actions_per_epoch min must be non-negative")
    if not isinstance(params.drep_type_distribution, dict):
        raise ValueError("drep_type_distribution must be a dictionary")
    if abs(sum(params.drep_type_distribution.values()) - 1.0) > 1e-6:
        raise ValueError("drep_type_distribution values must sum to 1.0")
    if params.peer_evaluation_sample_size < 0:
        raise ValueError("peer_evaluation_sample_size must be non-negative")
    if params.peer_evaluation_sample_size > params.drep_count:
        raise ValueError("peer_evaluation_sample_size cannot exceed drep_count")
    
    # Set random seed for reproducibility
    if params.random_seed is not None:
        np.random.seed(params.random_seed)
    
    logger.info(f"Initializing simulation with {params.drep_count} DReps and {params.total_ada_delegated:,} ADA")
    
    # Create initial state
    state = SimulationState(
        dreps={},
        actions={},
        current_epoch=0,
        active_actions={},
        votes={},
        evaluations={},
        rewards={},
        scores={},
        delegation_concentration={},
        participation_rates={},
        research_efforts={},
        peer_scores={},
        community_engagements={}
    )
    
    # Generate DRep profiles based on type distribution
    drep_types = []
    for drep_type, proportion in params.drep_type_distribution.items():
        count = int(params.drep_count * proportion)
        drep_types.extend([drep_type] * count)
    
    # Adjust if rounding caused incorrect total
    while len(drep_types) < params.drep_count:
        drep_types.append(np.random.choice(list(params.drep_type_distribution.keys())))
    while len(drep_types) > params.drep_count:
        drep_types.pop()
    
    # Shuffle the types
    np.random.shuffle(drep_types)
    
    # Generate initial delegation distribution
    if params.initial_delegation_distribution == "uniform":
        # Equal distribution
        delegations = [params.total_ada_delegated / params.drep_count] * params.drep_count
    
    elif params.initial_delegation_distribution == "pareto":
        # Pareto distribution (power law)
        alpha = params.delegation_distribution_params.get("alpha", 1.5)
        raw_values = np.random.pareto(alpha, params.drep_count) + 1
        delegations = raw_values / raw_values.sum() * params.total_ada_delegated
    
    elif params.initial_delegation_distribution == "exponential":
        # Exponential distribution
        scale = params.delegation_distribution_params.get("scale", 1.0)
        raw_values = np.random.exponential(scale, params.drep_count)
        delegations = raw_values / raw_values.sum() * params.total_ada_delegated
    
    elif params.initial_delegation_distribution == "lognormal":
        # Log-normal distribution
        sigma = params.delegation_distribution_params.get("sigma", 1.0)
        raw_values = np.random.lognormal(0, sigma, params.drep_count)
        delegations = raw_values / raw_values.sum() * params.total_ada_delegated
    
    else:
        raise ValueError(f"Unknown delegation distribution: {params.initial_delegation_distribution}")
    
    # Create DReps
    for i in range(params.drep_count):
        drep_id = f"drep_{i+1:04d}"
        drep_type = drep_types[i]
        
        # Generate profile
        profile = create_drep_profile(i, drep_type)
        
        # Create DRep state
        drep = DRepState(
            profile=profile,
            delegated_ada=delegations[i],
            delegation_history=[delegations[i]],
            rewards_history=[],
            vote_history=[],
            current_participation_rate=profile.reliability,
            current_research_effort=profile.research_capacity,
            current_community_engagement=profile.community_involvement
        )
        
        # Initialize strategy weights
        drep.initialize_strategy()
        
        state.dreps[drep_id] = drep
    
    # Calculate initial delegation concentration (Gini coefficient)
    delegations = [drep.delegated_ada for drep in state.dreps.values()]
    gini = calculate_gini_coefficient(delegations)
    state.delegation_concentration[0] = gini
    
    # Initialize metrics for epoch 0
    state.participation_rates[0] = {
        drep_id: drep.current_participation_rate for drep_id, drep in state.dreps.items()
    }
    state.research_efforts[0] = {
        drep_id: drep.current_research_effort for drep_id, drep in state.dreps.items()
    }
    state.community_engagements[0] = {
        drep_id: drep.current_community_engagement for drep_id, drep in state.dreps.items()
    }
    
    # Initialize peer scores with neutral values
    state.peer_scores[0] = {drep_id: 0.5 for drep_id in state.dreps}
    
    logger.info(f"Simulation initialized with Gini coefficient: {gini:.4f}")
    
    return state


def create_drep_profile(index: int, drep_type: str) -> DRepProfile:
    """
    Create a DRep profile with characteristics based on the DRep type.
    
    Args:
        index: Index of the DRep (for ID generation)
        drep_type: Type of DRep to create
        
    Returns:
        DRepProfile object
    """
    drep_id = f"drep_{index+1:04d}"
    
    # Set base parameters based on DRep type
    if drep_type == "Professional":
        base_participation = np.random.uniform(0.7, 0.95)
        base_research = np.random.uniform(0.7, 0.95)
        base_engagement = np.random.uniform(0.5, 0.9)
        adaptation = np.random.uniform(0.5, 0.8)
        profit_motivation = np.random.uniform(0.6, 0.9)
        community_motivation = np.random.uniform(0.5, 0.9)
        reputation_motivation = np.random.uniform(0.7, 0.95)
    elif drep_type == "Hobbyist":
        base_participation = np.random.uniform(0.5, 0.8)
        base_research = np.random.uniform(0.4, 0.8)
        base_engagement = np.random.uniform(0.6, 0.9)
        adaptation = np.random.uniform(0.3, 0.7)
        profit_motivation = np.random.uniform(0.3, 0.7)
        community_motivation = np.random.uniform(0.6, 0.9)
        reputation_motivation = np.random.uniform(0.5, 0.8)
    elif drep_type == "Passive":
        base_participation = np.random.uniform(0.2, 0.5)
        base_research = np.random.uniform(0.1, 0.4)
        base_engagement = np.random.uniform(0.1, 0.4)
        adaptation = np.random.uniform(0.1, 0.4)
        profit_motivation = np.random.uniform(0.5, 0.9)
        community_motivation = np.random.uniform(0.1, 0.5)
        reputation_motivation = np.random.uniform(0.2, 0.6)
    elif drep_type == "Strategic":
        base_participation = np.random.uniform(0.5, 0.9)
        base_research = np.random.uniform(0.3, 0.8)
        base_engagement = np.random.uniform(0.3, 0.7)
        adaptation = np.random.uniform(0.7, 0.95)
        profit_motivation = np.random.uniform(0.7, 0.95)
        community_motivation = np.random.uniform(0.3, 0.7)
        reputation_motivation = np.random.uniform(0.5, 0.9)
    else:
        # Default "Balanced" type
        base_participation = np.random.uniform(0.4, 0.7)
        base_research = np.random.uniform(0.4, 0.7)
        base_engagement = np.random.uniform(0.4, 0.7)
        adaptation = np.random.uniform(0.4, 0.6)
        profit_motivation = np.random.uniform(0.4, 0.7)
        community_motivation = np.random.uniform(0.4, 0.7)
        reputation_motivation = np.random.uniform(0.4, 0.7)
    
    # Assign additional characteristics
    regions = ["North America", "Europe", "Asia", "Africa", "South America", "Oceania"]
    region = np.random.choice(regions, p=[0.3, 0.3, 0.2, 0.05, 0.1, 0.05])
    
    # Determine if DRep has other roles (with low probability)
    is_spo = np.random.random() < 0.15  # 15% chance to also be an SPO
    is_cc = np.random.random() < 0.03   # 3% chance to be on Constitutional Committee
    
    return DRepProfile(
        id=drep_id,
        drep_type=drep_type,
        reliability=base_participation,
        research_capacity=base_research,
        strategic_voting=adaptation,
        community_involvement=base_engagement,
        constitutional_committee=is_cc,
        profit_motivation=profit_motivation,
        community_motivation=community_motivation,
        reputation_motivation=reputation_motivation,
        region=region,
        stake_pool_operator=is_spo
    )


def allocate_initial_delegations(
    dreps: Dict[str, DRepState],
    total_ada: float,
    distribution_type: str,
    distribution_params: Dict[str, float]
) -> None:
    """
    Allocate initial ADA delegations to DReps according to the specified distribution.
    
    Args:
        dreps: Dictionary of DRep states
        total_ada: Total ADA to allocate
        distribution_type: Type of distribution to use
        distribution_params: Parameters for the distribution
    
    Raises:
        ValueError: If an unknown distribution type is specified
    """
    drep_ids = list(dreps.keys())
    n_dreps = len(drep_ids)
    
    if n_dreps == 0:
        logger.warning("No DReps to allocate delegations to")
        return
    
    logger.info(f"Allocating {total_ada:,.0f} ADA using {distribution_type} distribution")
    
    if distribution_type == "uniform":
        # Equal distribution across all DReps
        ada_per_drep = total_ada / n_dreps
        delegations = [ada_per_drep] * n_dreps
    
    elif distribution_type == "pareto":
        # Pareto (power law) distribution - highly concentrated
        alpha = distribution_params.get("alpha", 1.5)  # Shape parameter
        
        # Generate raw Pareto values
        raw_values = np.random.pareto(alpha, n_dreps) + 1  # +1 to avoid zeros
        
        # Normalize to sum to total_ada
        delegations = (raw_values / raw_values.sum()) * total_ada
    
    elif distribution_type == "log-normal":
        # Log-normal distribution - moderate concentration
        sigma = distribution_params.get("sigma", 1.0)  # Standard deviation parameter
        
        # Generate raw log-normal values
        raw_values = np.random.lognormal(0, sigma, n_dreps)
        
        # Normalize to sum to total_ada
        delegations = (raw_values / raw_values.sum()) * total_ada
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    # Assign delegations to DReps
    for drep_id, delegation in zip(drep_ids, delegations):
        dreps[drep_id].delegated_ada = delegation
        dreps[drep_id].delegation_history.append(delegation)


def generate_governance_actions(
    epoch: int, 
    params: SimulationParameters
) -> List[GovernanceAction]:
    """
    Generate new governance actions for the current epoch.
    
    Args:
        epoch: Current epoch number
        params: Simulation parameters
        
    Returns:
        List of new governance actions
    """
    # Determine number of new actions this epoch
    min_actions, max_actions = params.governance_actions_per_epoch
    n_actions = np.random.randint(min_actions, max_actions + 1)
    
    logger.debug(f"Generating {n_actions} new governance actions for epoch {epoch}")
    
    actions = []
    categories = ["Technical", "Treasury", "Parameter", "Constitutional"]
    category_weights = [0.4, 0.3, 0.2, 0.1]
    
    for i in range(n_actions):
        action_id = f"action_{epoch:04d}_{i:02d}"
        
        # Select category
        category = np.random.choice(categories, p=category_weights)
        
        # Generate complexity and impact based partly on category
        if category == "Technical":
            complexity = np.random.uniform(0.6, 0.9)
            impact = np.random.uniform(0.3, 0.8)
        elif category == "Treasury":
            complexity = np.random.uniform(0.4, 0.7)
            impact = np.random.uniform(0.5, 0.9)
        elif category == "Parameter":
            complexity = np.random.uniform(0.5, 0.8)
            impact = np.random.uniform(0.4, 0.7)
        else:  # Constitutional
            complexity = np.random.uniform(0.7, 0.95)
            impact = np.random.uniform(0.6, 0.95)
        
        # Determine voting period length based on complexity
        voting_periods = max(1, int(np.ceil(complexity * 3)))
        
        action = GovernanceAction(
            id=action_id,
            title=f"Governance Action {action_id}",
            description=f"Description for governance action {action_id}",
            complexity=complexity,
            community_impact=impact,
            creation_epoch=epoch,
            voting_periods=voting_periods,
            category=category
        )
        
        actions.append(action)
    
    return actions


def run_epoch(
    state: SimulationState,
    incentive_params: IncentiveParameters,
    sim_params: SimulationParameters
) -> SimulationState:
    """
    Run a single simulation epoch.
    
    This function:
    1. Creates new governance actions
    2. Collects votes from DReps
    3. Processes votes and determines outcomes
    4. Calculates rewards and updates scores
    5. Updates delegations based on performance
    6. Updates DRep strategies
    
    Args:
        state: Current simulation state
        incentive_params: Incentive parameters
        sim_params: Simulation parameters
        
    Returns:
        Updated simulation state
    """
    epoch = state.current_epoch
    logger.info(f"Running epoch {epoch}")
    
    # Generate new governance actions
    new_actions = generate_governance_actions(epoch, sim_params)
    for action in new_actions:
        state.actions[action.id] = action
    
    # Determine active actions for this epoch
    active_action_ids = []
    for action_id, action in state.actions.items():
        if action.is_active(epoch):
            active_action_ids.append(action_id)
    
    state.active_actions[epoch] = active_action_ids
    logger.debug(f"Epoch {epoch}: {len(active_action_ids)} active governance actions")
    
    # DReps vote on active actions
    votes = []
    participation_rates = {}
    research_efforts = {}
    
    for drep_id, drep in state.dreps.items():
        actions_voted = 0
        total_research = 0
        
        for action_id in active_action_ids:
            action = state.actions[action_id]
            choice, effort = drep.decide_vote(action, incentive_params)
            
            if choice is not None:  # DRep decided to vote
                actions_voted += 1
                total_research += effort
                
                vote = DRepVote(
                    drep_id=drep_id,
                    action_id=action_id,
                    choice=choice,
                    epoch=epoch,
                    research_effort=effort,
                    rationale=f"Rationale from {drep_id} on {action_id}"  # Simplified
                )
                
                votes.append(vote)
                drep.vote_history.append(vote)
        
        # Calculate participation rate and average research effort
        participation_rate = actions_voted / len(active_action_ids) if active_action_ids else 0
        avg_research = total_research / actions_voted if actions_voted > 0 else 0
        
        # Update DRep state with current metrics
        drep.current_participation_rate = participation_rate
        drep.current_research_effort = avg_research
        
        # Store metrics for analysis
        participation_rates[drep_id] = participation_rate
        research_efforts[drep_id] = avg_research
    
    # Store votes for this epoch
    state.votes[epoch] = votes
    state.participation_rates[epoch] = participation_rates
    state.research_efforts[epoch] = research_efforts
    
    # Process peer evaluations
    evaluations = []
    peer_scores = {}
    
    for drep_id, drep in state.dreps.items():
        # Each DRep evaluates peers
        drep_evaluations = drep.evaluate_peers(votes, incentive_params)
        evaluations.extend(drep_evaluations)
        
        # Calculate average peer score received
        received_evals = [e for e in evaluations if e.evaluated_id == drep_id]
        avg_score = np.mean([e.score for e in received_evals]) if received_evals else 0
        peer_scores[drep_id] = avg_score
        
        # Update DRep state with evaluations
        for eval in drep_evaluations:
            drep.peer_evaluations_given.append(eval)
        
        for eval in received_evals:
            drep.peer_evaluations_received.append(eval)
    
    # Store evaluations for this epoch
    state.evaluations[epoch] = evaluations
    state.peer_scores[epoch] = peer_scores
    
    # Process community engagement
    community_engagements = {}
    for drep_id, drep in state.dreps.items():
        # Simplified model: engagement is a function of base engagement and participation
        engagement = drep.profile.community_involvement * (0.5 + 0.5 * drep.current_participation_rate)
        drep.current_community_engagement = engagement
        community_engagements[drep_id] = engagement
    
    state.community_engagements[epoch] = community_engagements
    
    # Calculate rewards using the specified incentive model
    scores, rewards = calculate_rewards(
        state, 
        incentive_params, 
        epoch, 
        model_name=incentive_params.model_type
    )
    
    # Store rewards and update DRep histories
    state.rewards[epoch] = rewards
    state.scores[epoch] = scores
    
    for drep_id, reward in rewards.items():
        drep = state.dreps[drep_id]
        drep.rewards_history.append(reward)
        
        # Add score to history if available
        if drep_id in scores:
            drep.score_history.append(scores[drep_id])
        else:
            drep.score_history.append(0)
    
    # Finalize action outcomes based on votes
    for action_id in active_action_ids:
        action = state.actions[action_id]
        action_votes = [v for v in votes if v.action_id == action_id]
        
        if action_votes:
            # Count votes by choice
            vote_counts = {}
            for choice in action.choices:
                count = sum(1 for v in action_votes if v.choice == choice)
                vote_counts[choice] = count
            
            # Calculate proportions
            total_votes = sum(vote_counts.values())
            vote_proportions = {choice: count / total_votes for choice, count in vote_counts.items()}
            
            # Store outcomes
            action.outcomes = vote_proportions
            
            # Determine if action is accepted (simple majority of "Yes" votes)
            action.accepted = vote_proportions.get("Yes", 0) > 0.5
            
            # Determine if action is vetoed (with random probability)
            action.vetoed = np.random.random() < sim_params.veto_probability
    
    # DReps adapt their strategies based on rewards
    epoch_metrics = {
        "participation_rates": [state.participation_rates.get(e, {}) for e in range(epoch + 1)],
        "research_efforts": [state.research_efforts.get(e, {}) for e in range(epoch + 1)],
        "peer_scores": [state.peer_scores.get(e, {}) for e in range(epoch + 1)],
        "community_engagements": [state.community_engagements.get(e, {}) for e in range(epoch + 1)]
    }
    
    for drep_id, drep in state.dreps.items():
        reward = rewards.get(drep_id, 0)
        drep.adapt_strategy(reward, epoch_metrics)
    
    # Update delegations based on performance
    update_delegations(state, sim_params, epoch)
    
    # Calculate concentration metrics
    delegations = [drep.delegated_ada for drep in state.dreps.values()]
    gini = calculate_gini_coefficient(delegations)
    state.delegation_concentration[epoch] = gini
    
    # Increment epoch
    state.current_epoch += 1
    
    return state


def update_delegations(
    state: SimulationState,
    params: SimulationParameters,
    epoch: int
) -> None:
    """
    Update delegations based on DRep performance.
    
    This simulates delegators moving their ADA between DReps based on observed
    performance (rewards, participation, etc.).
    
    Args:
        state: Current simulation state
        params: Simulation parameters
        epoch: Current epoch
    """
    # Skip the first epoch as there's no performance history
    if epoch == 0:
        return
    
    # Determine what percentage of total delegations will move this epoch
    change_rate = params.delegation_change_rate
    total_delegated = sum(drep.delegated_ada for drep in state.dreps.values())
    ada_to_move = total_delegated * change_rate
    
    logger.debug(f"Epoch {epoch}: Moving {ada_to_move:,.0f} ADA between DReps")
    
    # Calculate "attractiveness" scores for each DRep based on recent performance
    attractiveness_scores = {}
    
    for drep_id, drep in state.dreps.items():
        # Base attractiveness on recent rewards and participation
        recent_rewards = drep.rewards_history[-1] if drep.rewards_history else 0
        participation = drep.current_participation_rate
        research = drep.current_research_effort
        
        # Weight factors (could be parameters)
        reward_weight = 0.6
        participation_weight = 0.3
        research_weight = 0.1
        
        # Normalize rewards relative to delegation
        reward_ratio = recent_rewards / drep.delegated_ada if drep.delegated_ada > 0 else 0
        
        # Calculate attractiveness
        attractiveness = (
            reward_weight * reward_ratio +
            participation_weight * participation +
            research_weight * research
        )
        
        attractiveness_scores[drep_id] = max(0, attractiveness)
    
    # Normalize attractiveness scores
    total_attractiveness = sum(attractiveness_scores.values())
    if total_attractiveness > 0:
        for drep_id in attractiveness_scores:
            attractiveness_scores[drep_id] /= total_attractiveness
    
    # Move ADA based on attractiveness
    new_delegations = {drep_id: drep.delegated_ada for drep_id, drep in state.dreps.items()}
    
    for drep_id, drep in state.dreps.items():
        # Amount this DRep will lose
        outflow = drep.delegated_ada * change_rate
        new_delegations[drep_id] -= outflow
    
    for drep_id in attractiveness_scores:
        # Amount this DRep will gain
        inflow = ada_to_move * attractiveness_scores[drep_id]
        new_delegations[drep_id] += inflow
    
    # Update delegations and histories
    for drep_id, new_amount in new_delegations.items():
        state.dreps[drep_id].delegated_ada = new_amount
        state.dreps[drep_id].delegation_history.append(new_amount)


def calculate_gini_coefficient(values: List[float]) -> float:
    """
    Calculate the Gini coefficient, a measure of inequality.
    
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    
    Args:
        values: List of values (e.g., delegations)
        
    Returns:
        Gini coefficient
    """
    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    if n <= 1 or sum(sorted_values) == 0:
        return 0  # No inequality with 0 or 1 elements
    
    # Calculate cumulative sum
    cumulative = np.cumsum(sorted_values)
    
    # Calculate Gini coefficient using the area under the Lorenz curve
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def run_simulation(
    sim_params: SimulationParameters,
    incentive_params: IncentiveParameters,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> SimulationState:
    """
    Run a complete simulation for the specified number of epochs.
    
    Args:
        sim_params: Simulation parameters
        incentive_params: Incentive parameters
        progress_callback: Optional callback function to report progress
        
    Returns:
        Final simulation state
    """
    # Initialize simulation
    state = initialize_simulation(sim_params)
    
    # Run for specified number of epochs
    for epoch in range(sim_params.total_epochs):
        state = run_epoch(state, incentive_params, sim_params)
        
        # Report progress if callback provided
        if progress_callback:
            progress_callback(epoch + 1, sim_params.total_epochs)
    
    logger.info(f"Simulation completed after {sim_params.total_epochs} epochs")
    return state


def calculate_simulation_metrics(state: SimulationState) -> Dict[str, Any]:
    """
    Calculate overall metrics for the simulation.
    
    Args:
        state: Final simulation state
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Calculate delegation concentration over time
    metrics['delegation_concentration'] = {
        epoch: gini for epoch, gini in state.delegation_concentration.items()
    }
    
    # Calculate participation rates over time
    metrics['participation_rates'] = {
        epoch: np.mean(list(rates.values())) 
        for epoch, rates in state.participation_rates.items()
    }
    
    # Calculate research efforts over time
    metrics['research_efforts'] = {
        epoch: np.mean(list(efforts.values())) 
        for epoch, efforts in state.research_efforts.items()
    }
    
    # Calculate metrics by DRep type
    drep_types = {}
    for drep_id, drep in state.dreps.items():
        drep_type = drep.profile.drep_type
        if drep_type not in drep_types:
            drep_types[drep_type] = []
        drep_types[drep_type].append(drep_id)
    
    metrics['drep_type_metrics'] = {}
    for drep_type, drep_ids in drep_types.items():
        type_metrics = {
            'count': len(drep_ids),
            'avg_delegation': np.mean([state.dreps[d].delegated_ada for d in drep_ids]),
            'avg_participation': np.mean([state.participation_rates[state.current_epoch][d] for d in drep_ids]),
            'avg_research': np.mean([state.research_efforts[state.current_epoch][d] for d in drep_ids]),
            'total_rewards': sum([sum(state.dreps[d].rewards_history) for d in drep_ids])
        }
        metrics['drep_type_metrics'][drep_type] = type_metrics
    
    # Calculate action metrics
    action_metrics = {
        'total': len(state.actions),
        'accepted': sum(1 for a in state.actions.values() if a.accepted),
        'vetoed': sum(1 for a in state.actions.values() if a.vetoed),
        'by_category': {}
    }
    
    for action in state.actions.values():
        if action.category not in action_metrics['by_category']:
            action_metrics['by_category'][action.category] = {
                'total': 0, 'accepted': 0, 'vetoed': 0
            }
        action_metrics['by_category'][action.category]['total'] += 1
        if action.accepted:
            action_metrics['by_category'][action.category]['accepted'] += 1
        if action.vetoed:
            action_metrics['by_category'][action.category]['vetoed'] += 1
    
    metrics['action_metrics'] = action_metrics
    
    return metrics