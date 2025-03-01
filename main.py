"""
DRep Incentivization Hybrid Simulation Framework
===============================================

This framework combines Monte Carlo methods with agent-based modeling to simulate
and analyze different incentive structures for Delegated Representatives (DReps) in
a blockchain governance ecosystem.

Author: [Your Name]
Date: February 28, 2025
Version: 1.0

Mathematical notation for key formulas is included as docstrings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional, Union, Any
from tqdm import tqdm
import multiprocessing as mp
from scipy import stats
import json
import os
import logging
from datetime import datetime
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("drep_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DRepSimulation")

# Constants
EPOCH_LENGTH = 5  # days
ADA_TOTAL_SUPPLY = 45_000_000_000  # 45 billion ADA
DEFAULT_DECIMALS = 6  # ADA has 6 decimal places

#############################################################################
# Core Data Structures
#############################################################################

@dataclass
class GovernanceAction:
    """Represents a governance proposal that DReps can vote on."""
    id: str
    title: str
    description: str
    complexity: float  # How difficult it is to evaluate (0-1)
    community_impact: float  # Expected impact on ecosystem (0-1)
    creation_epoch: int
    voting_periods: int  # How many epochs this is available for voting
    category: str  # e.g., "Technical", "Treasury", "Constitutional", etc.
    choices: List[str] = field(default_factory=lambda: ["Yes", "No", "Abstain"])
    outcomes: Dict[str, float] = field(default_factory=dict)  # Actual voting outcomes
    accepted: bool = False  # Whether the proposal was ultimately accepted
    vetoed: bool = False  # Whether it was vetoed by the Constitutional Committee

    def is_active(self, current_epoch: int) -> bool:
        """Check if this action is currently available for voting."""
        return (
            current_epoch >= self.creation_epoch and 
            current_epoch < self.creation_epoch + self.voting_periods
        )


@dataclass
class DRepVote:
    """Records a vote by a DRep on a governance action."""
    drep_id: str
    action_id: str
    choice: str
    epoch: int
    research_effort: float  # How much effort the DRep put into research (0-1)
    rationale: str  # Explanation for the vote
    aligned_with_outcome: bool = False  # Whether vote aligned with final outcome
    
    
@dataclass
class PeerEvaluation:
    """Records a DRep's evaluation of another DRep's voting rationale."""
    evaluator_id: str
    evaluated_id: str
    action_id: str
    score: float  # 0-1 score of the quality of the rationale
    epoch: int


@dataclass
class DRepProfile:
    """Static characteristics of a DRep that don't change during simulation."""
    id: str
    type: str  # "Professional", "Hobbyist", "Passive", "Strategic"
    
    # Base behavioral parameters
    base_participation_rate: float  # Probability of voting on an action (0-1)
    base_research_capacity: float  # Maximum research effort possible (0-1)
    base_community_engagement: float  # Baseline community involvement (0-1)
    
    # Strategic adaptability
    adaptation_rate: float  # How quickly the DRep adapts strategies (0-1)
    profit_motivation: float  # How motivated by financial rewards (0-1)
    community_motivation: float  # How motivated by community benefit (0-1)
    reputation_motivation: float  # How motivated by reputation/status (0-1)
    
    # Demographics (for diversity analysis)
    region: str  # Geographic region
    stake_pool_operator: bool = False  # Is also an SPO
    constitutional_committee: bool = False  # Is also on the CC


@dataclass
class DRepState:
    """Dynamic state of a DRep that evolves during simulation."""
    profile: DRepProfile
    delegated_ada: float = 0.0  # Amount of ADA delegated to this DRep
    
    # Current epoch metrics
    current_participation_rate: float = 0.0
    current_research_effort: float = 0.0
    current_community_engagement: float = 0.0
    
    # Historical data
    delegation_history: List[float] = field(default_factory=list)
    vote_history: List[DRepVote] = field(default_factory=list)
    rewards_history: List[float] = field(default_factory=list)
    score_history: List[float] = field(default_factory=list)
    peer_evaluations_given: List[PeerEvaluation] = field(default_factory=list)
    peer_evaluations_received: List[PeerEvaluation] = field(default_factory=list)
    
    # Strategy adaptation
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    
    def initialize_strategy(self) -> None:
        """Initialize the DRep's strategy weights based on profile."""
        self.strategy_weights = {
            "participation": 0.3,
            "research_effort": 0.3,
            "peer_evaluation": 0.2,
            "community_engagement": 0.2
        }
        
        # Adjust based on DRep type
        if self.profile.type == "Professional":
            self.strategy_weights["research_effort"] += 0.1
            self.strategy_weights["participation"] += 0.1
            self.strategy_weights["community_engagement"] -= 0.1
            self.strategy_weights["peer_evaluation"] -= 0.1
        elif self.profile.type == "Hobbyist":
            self.strategy_weights["community_engagement"] += 0.1
            self.strategy_weights["research_effort"] -= 0.1
        elif self.profile.type == "Passive":
            self.strategy_weights["participation"] -= 0.2
            self.strategy_weights["research_effort"] -= 0.1
            self.strategy_weights["peer_evaluation"] += 0.2
            self.strategy_weights["community_engagement"] += 0.1
        elif self.profile.type == "Strategic":
            # Will be more dynamic, updated based on rewards
            pass
    
    def decide_vote(self, action: GovernanceAction, incentives: 'IncentiveParameters') -> Tuple[Optional[str], float]:
        """
        Determine whether to vote on an action and how much research effort to invest.
        
        Returns:
            Tuple of (vote_choice, research_effort) or (None, 0) if not voting
        """
        # Calculate base probability of participation
        vote_probability = self.profile.base_participation_rate
        
        # Adjust based on incentives
        incentive_factor = (
            (incentives.w2_participation * self.strategy_weights["participation"]) +
            (incentives.w3_successful_votes * self.strategy_weights["research_effort"])
        )
        vote_probability *= (1 + incentive_factor)
        
        # Cap at 1.0
        vote_probability = min(vote_probability, 1.0)
        
        # Decide whether to vote
        if np.random.random() > vote_probability:
            return None, 0.0
        
        # If voting, determine research effort
        max_effort = self.profile.base_research_capacity
        
        # Adjust effort based on incentives and action complexity
        effort_factor = (
            (incentives.w3_successful_votes * self.strategy_weights["research_effort"]) +
            (action.complexity * 0.5) +  # Higher complexity demands more effort
            (action.community_impact * 0.5)  # Higher impact motivates more effort
        )
        
        research_effort = max_effort * (0.5 + 0.5 * effort_factor)
        research_effort = min(max(research_effort, 0.1), 1.0)  # Bound between 0.1 and 1.0
        
        # Simple vote choice model - could be made more sophisticated
        if research_effort > 0.7:
            # Higher research leads to more "correct" choices (simplified)
            choice = "Yes" if action.community_impact > 0.5 else "No"
        else:
            # Lower research means more randomness in choice
            if np.random.random() < 0.7:
                choice = "Yes" if action.community_impact > 0.5 else "No"
            else:
                choice = "No" if action.community_impact > 0.5 else "Yes"
        
        return choice, research_effort
    
    def evaluate_peers(self, votes: List[DRepVote], incentives: 'IncentiveParameters') -> List[PeerEvaluation]:
        """Evaluate the voting rationales of other DReps."""
        evaluations = []
        
        # Skip if DRep doesn't prioritize peer evaluation
        if self.strategy_weights["peer_evaluation"] < 0.1:
            return evaluations
        
        # Consider only a subset of votes from other DReps (simulation optimization)
        other_votes = [v for v in votes if v.drep_id != self.profile.id]
        if not other_votes:
            return evaluations
        
        # Sample votes to evaluate based on DRep's engagement level
        sample_size = max(1, int(len(other_votes) * self.profile.base_participation_rate))
        votes_to_evaluate = np.random.choice(other_votes, 
                                             size=min(sample_size, len(other_votes)), 
                                             replace=False)
        
        for vote in votes_to_evaluate:
            # Base score on research effort with some randomness
            # This is a simplified model - in reality, this would involve NLP or human judgment
            base_score = vote.research_effort
            noise = np.random.normal(0, 0.15)  # Add some noise
            score = max(0, min(1, base_score + noise))
            
            # Strategic DReps might manipulate scores
            if self.profile.type == "Strategic" and self.profile.profit_motivation > 0.7:
                # Potential score manipulation based on competitive dynamics
                pass
            
            evaluation = PeerEvaluation(
                evaluator_id=self.profile.id,
                evaluated_id=vote.drep_id,
                action_id=vote.action_id,
                score=score,
                epoch=vote.epoch
            )
            evaluations.append(evaluation)
        
        return evaluations
    
    def adapt_strategy(self, rewards: float, epoch_metrics: Dict[str, float]) -> None:
        """
        Update strategy based on observed rewards and metrics.
        
        Parameters:
            rewards: The rewards received in the last epoch
            epoch_metrics: Dictionary of performance metrics from the last epoch
        """
        if not self.rewards_history:
            # No history yet to adapt from
            return
        
        # Only strategic DReps adapt their strategy significantly
        if self.profile.type != "Strategic" and np.random.random() > self.profile.adaptation_rate:
            return
        
        # Compare rewards to previous epochs
        prev_reward = self.rewards_history[-1] if len(self.rewards_history) > 1 else 0
        reward_change = rewards - prev_reward
        
        # Identify which metrics correlate with higher rewards
        correlation_data = {
            "participation": [],
            "research_effort": [],
            "peer_evaluation": [],
            "community_engagement": []
        }
        
        # Gather historical data for correlation
        for i in range(min(5, len(self.rewards_history))):
            if i < len(epoch_metrics.get("participation_rates", [])):
                correlation_data["participation"].append(
                    (epoch_metrics["participation_rates"][-i-1], self.rewards_history[-i-1])
                )
            if i < len(epoch_metrics.get("research_efforts", [])):
                correlation_data["research_effort"].append(
                    (epoch_metrics["research_efforts"][-i-1], self.rewards_history[-i-1])
                )
            if i < len(epoch_metrics.get("peer_scores", [])):
                correlation_data["peer_evaluation"].append(
                    (epoch_metrics["peer_scores"][-i-1], self.rewards_history[-i-1])
                )
            if i < len(epoch_metrics.get("community_engagements", [])):
                correlation_data["community_engagement"].append(
                    (epoch_metrics["community_engagements"][-i-1], self.rewards_history[-i-1])
                )
        
        # Calculate simplified correlations
        correlations = {}
        for metric, data in correlation_data.items():
            if len(data) > 1:
                x = [d[0] for d in data]
                y = [d[1] for d in data]
                if len(set(x)) > 1:  # Ensure variability in x
                    correlations[metric] = np.corrcoef(x, y)[0, 1]
                else:
                    correlations[metric] = 0
            else:
                correlations[metric] = 0
        
        # Adjust strategy weights based on correlations
        adjustment_rate = self.profile.adaptation_rate * 0.2  # Small adjustment per epoch
        
        for metric, correlation in correlations.items():
            if not np.isnan(correlation):
                adjustment = correlation * adjustment_rate
                self.strategy_weights[metric] = max(0.1, min(0.5, 
                                                         self.strategy_weights[metric] + adjustment))
        
        # Normalize weights to sum to 1
        total = sum(self.strategy_weights.values())
        for metric in self.strategy_weights:
            self.strategy_weights[metric] /= total


@dataclass
class IncentiveParameters:
    """
    Parameters defining the incentive model for rewarding DReps.
    
    Mathematical notation:
    ---------------------
    DRep_Score = w₁ × Delegation_Ratio + 
                 w₂ × Participation_Rate + 
                 w₃ × Successful_Votes_Ratio - 
                 w₄ × Veto_Penalty + 
                 w₅ × Decentralization_Score + 
                 w₆ × Peer_Evaluation_Score + 
                 w₇ × Community_Engagement_Score
    
    DRep_Reward = (DRep_Score / Sum_of_All_DRep_Scores) × Total_Reward_Pool
    """
    # Weight parameters
    w1_delegation: float  # Weight for delegation ratio
    w2_participation: float  # Weight for participation rate
    w3_successful_votes: float  # Weight for successful votes ratio
    w4_veto_penalty: float  # Weight for veto penalty
    w5_decentralization: float  # Weight for decentralization score
    w6_peer_evaluation: float  # Weight for peer evaluation score
    w7_community_engagement: float  # Weight for community engagement
    
    # Other parameters
    total_reward_per_epoch: float  # Total ADA to distribute per epoch
    veto_penalty_factor: float  # How much a veto reduces score
    min_participation_threshold: float  # Minimum participation to get rewards
    
    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        weights_sum = (
            self.w1_delegation + 
            self.w2_participation + 
            self.w3_successful_votes + 
            self.w4_veto_penalty + 
            self.w5_decentralization + 
            self.w6_peer_evaluation + 
            self.w7_community_engagement
        )
        
        if abs(weights_sum - 1.0) > 1e-6:
            self.w1_delegation /= weights_sum
            self.w2_participation /= weights_sum
            self.w3_successful_votes /= weights_sum
            self.w4_veto_penalty /= weights_sum
            self.w5_decentralization /= weights_sum
            self.w6_peer_evaluation /= weights_sum
            self.w7_community_engagement /= weights_sum


@dataclass
class SimulationParameters:
    """Parameters controlling the overall simulation environment."""
    total_epochs: int  # Number of epochs to simulate
    drep_count: int  # Number of DReps to simulate
    governance_actions_per_epoch: Tuple[int, int]  # (min, max) actions per epoch
    total_ada_delegated: float  # Total ADA delegated to all DReps
    delegation_change_rate: float  # Rate at which delegations change per epoch
    veto_probability: float  # Probability of a governance action being vetoed
    
    # Distribution parameters for DRep population
    drep_type_distribution: Dict[str, float]  # Percentage of each DRep type
    initial_delegation_distribution: str  # "uniform", "pareto", "log-normal"
    delegation_distribution_params: Dict[str, float]  # Parameters for the distribution
    
    # Evaluation parameters
    peer_evaluation_sample_size: int  # Number of peers each DRep evaluates
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None


@dataclass
class SimulationState:
    """The current state of the simulation across all epochs."""
    dreps: Dict[str, DRepState]  # All DReps in the simulation
    actions: Dict[str, GovernanceAction]  # All governance actions
    current_epoch: int = 0
    
    # Epoch-specific data
    active_actions: Dict[int, List[str]] = field(default_factory=dict)  # Actions by epoch
    votes: Dict[int, List[DRepVote]] = field(default_factory=dict)  # Votes by epoch
    evaluations: Dict[int, List[PeerEvaluation]] = field(default_factory=dict)  # Evals by epoch
    rewards: Dict[int, Dict[str, float]] = field(default_factory=dict)  # Rewards by epoch
    scores: Dict[int, Dict[str, float]] = field(default_factory=dict)  # Scores by epoch
    
    # Metrics tracking
    delegation_concentration: Dict[int, float] = field(default_factory=dict)  # Gini by epoch
    participation_rates: Dict[int, Dict[str, float]] = field(default_factory=dict)
    research_efforts: Dict[int, Dict[str, float]] = field(default_factory=dict)
    peer_scores: Dict[int, Dict[str, float]] = field(default_factory=dict)
    community_engagements: Dict[int, Dict[str, float]] = field(default_factory=dict)


#############################################################################
# Simulation Core Functions
#############################################################################

def initialize_simulation(params: SimulationParameters) -> SimulationState:
    """
    Initialize a new simulation with the given parameters.
    
    Parameters:
        params: The parameters to use for initialization
        
    Returns:
        A new SimulationState with initialized DReps and empty actions
    """
    if params.random_seed is not None:
        np.random.seed(params.random_seed)
    
    # Initialize DReps
    dreps = {}
    for i in range(params.drep_count):
        # Determine DRep type based on distribution
        drep_type = np.random.choice(
            list(params.drep_type_distribution.keys()),
            p=list(params.drep_type_distribution.values())
        )
        
        # Generate DRep profile with type-specific parameters
        profile = create_drep_profile(i, drep_type)
        
        # Create initial DRep state
        drep_state = DRepState(profile=profile)
        drep_state.initialize_strategy()
        
        dreps[profile.id] = drep_state
    
    # Allocate initial delegations
    allocate_initial_delegations(
        dreps, 
        params.total_ada_delegated, 
        params.initial_delegation_distribution,
        params.delegation_distribution_params
    )
    
    # Create simulation state
    return SimulationState(dreps=dreps, actions={})


def create_drep_profile(index: int, drep_type: str) -> DRepProfile:
    """Create a DRep profile with characteristics based on the DRep type."""
    drep_id = f"drep_{index:04d}"
    
    # Base parameters for different DRep types
    if drep_type == "Professional":
        base_participation = np.random.uniform(0.8, 0.95)
        base_research = np.random.uniform(0.7, 0.9)
        base_engagement = np.random.uniform(0.5, 0.8)
        adaptation = np.random.uniform(0.4, 0.7)
        profit_motivation = np.random.uniform(0.6, 0.9)
        community_motivation = np.random.uniform(0.5, 0.8)
        reputation_motivation = np.random.uniform(0.7, 0.9)
    elif drep_type == "Hobbyist":
        base_participation = np.random.uniform(0.5, 0.8)
        base_research = np.random.uniform(0.4, 0.7)
        base_engagement = np.random.uniform(0.4, 0.9)
        adaptation = np.random.uniform(0.3, 0.6)
        profit_motivation = np.random.uniform(0.3, 0.7)
        community_motivation = np.random.uniform(0.6, 0.9)
        reputation_motivation = np.random.uniform(0.5, 0.8)
    elif drep_type == "Passive":
        base_participation = np.random.uniform(0.1, 0.4)
        base_research = np.random.uniform(0.1, 0.5)
        base_engagement = np.random.uniform(0.1, 0.4)
        adaptation = np.random.uniform(0.1, 0.3)
        profit_motivation = np.random.uniform(0.2, 0.6)
        community_motivation = np.random.uniform(0.3, 0.7)
        reputation_motivation = np.random.uniform(0.2, 0.5)
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
        type=drep_type,
        base_participation_rate=base_participation,
        base_research_capacity=base_research,
        base_community_engagement=base_engagement,
        adaptation_rate=adaptation,
        profit_motivation=profit_motivation,
        community_motivation=community_motivation,
        reputation_motivation=reputation_motivation,
        region=region,
        stake_pool_operator=is_spo,
        constitutional_committee=is_cc
    )


def allocate_initial_delegations(
    dreps: Dict[str, DRepState],
    total_ada: float,
    distribution_type: str,
    distribution_params: Dict[str, float]
) -> None:
    """
    Allocate initial ADA delegations to DReps according to the specified distribution.
    
    Parameters:
        dreps: Dictionary of DRep states
        total_ada: Total ADA to allocate
        distribution_type: Type of distribution to use
        distribution_params: Parameters for the distribution
    """
    drep_ids = list(dreps.keys())
    n_dreps = len(drep_ids)
    
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
    
    Parameters:
        epoch: Current epoch number
        params: Simulation parameters
        
    Returns:
        List of new governance actions
    """
    # Determine number of new actions this epoch
    min_actions, max_actions = params.governance_actions_per_epoch
    n_actions = np.random.randint(min_actions, max_actions + 1)
    
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
    incentives: IncentiveParameters,
    sim_params: SimulationParameters
) -> SimulationState:
    """
    Run a single epoch of the simulation.
    
    Parameters:
        state: Current simulation state
        incentives: Incentive parameters
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
    
    # DReps vote on active actions
    votes = []
    participation_rates = {}
    research_efforts = {}
    
    for drep_id, drep in state.dreps.items():
        actions_voted = 0
        total_research = 0
        
        for action_id in active_action_ids:
            action = state.actions[action_id]
            choice, effort = drep.decide_vote(action, incentives)
            
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
        drep_evaluations = drep.evaluate_peers(votes, incentives)
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
        engagement = drep.profile.base_community_engagement * (0.5 + 0.5 * drep.current_participation_rate)
        drep.current_community_engagement = engagement
        community_engagements[drep_id] = engagement
    
    state.community_engagements[epoch] = community_engagements
    
    # Calculate rewards based on incentive model
    scores, rewards = calculate_rewards(state, incentives, epoch)
    
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
    
    Parameters:
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


def update_delegations(
    state: SimulationState,
    params: SimulationParameters,
    epoch: int
) -> None:
    """
    Update delegations based on DRep performance.
    
    This simulates delegators moving their ADA between DReps based on observed
    performance (rewards, participation, etc.).
    
    Parameters:
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
    
    Parameters:
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


#############################################################################
# Monte Carlo Simulation Framework
#############################################################################

def generate_parameter_set(param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """
    Generate a random set of parameters within the specified ranges.
    
    Parameters:
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
    epochs_per_iteration: int,
    param_ranges: Dict[str, Tuple[float, float]],
    base_sim_params: SimulationParameters,
    n_parallel: int = 1
) -> List[Dict]:
    """
    Run a Monte Carlo simulation with randomly sampled parameters.
    
    Parameters:
        n_iterations: Number of parameter combinations to test
        epochs_per_iteration: Number of epochs to simulate for each parameter set
        param_ranges: Ranges for incentive parameters to sample from
        base_sim_params: Base simulation parameters
        n_parallel: Number of parallel processes to use
        
    Returns:
        List of result dictionaries containing parameters and metrics
    """
    logger.info(f"Starting Monte Carlo simulation with {n_iterations} iterations")
    
    def run_single_iteration(iteration: int) -> Dict:
        """Run a single iteration of the Monte Carlo simulation."""
        # Sample incentive parameters
        incentive_params_dict = generate_parameter_set(param_ranges)
        
        # Create IncentiveParameters object
        incentive_params = IncentiveParameters(
            w1_delegation=incentive_params_dict.get('w1_delegation', 0.2),
            w2_participation=incentive_params_dict.get('w2_participation', 0.2),
            w3_successful_votes=incentive_params_dict.get('w3_successful_votes', 0.2),
            w4_veto_penalty=incentive_params_dict.get('w4_veto_penalty', 0.1),
            w5_decentralization=incentive_params_dict.get('w5_decentralization', 0.1),
            w6_peer_evaluation=incentive_params_dict.get('w6_peer_evaluation', 0.1),
            w7_community_engagement=incentive_params_dict.get('w7_community_engagement', 0.1),
            total_reward_per_epoch=incentive_params_dict.get('total_reward_per_epoch', 100000),
            veto_penalty_factor=incentive_params_dict.get('veto_penalty_factor', 0.5),
            min_participation_threshold=incentive_params_dict.get('min_participation_threshold', 0.1)
        )
        
        # Use a unique random seed for this iteration
        sim_params = SimulationParameters(
            **{k: v for k, v in vars(base_sim_params).items() if k != 'random_seed'},
            random_seed=np.random.randint(0, 2**32 - 1),
            total_epochs=epochs_per_iteration
        )
        
        # Initialize and run simulation
        sim_state = initialize_simulation(sim_params)
        
        for epoch in range(epochs_per_iteration):
            sim_state = run_epoch(sim_state, incentive_params, sim_params)
        
        # Calculate metrics
        metrics = calculate_simulation_metrics(sim_state)
        
        return {
            'iteration': iteration,
            'parameters': incentive_params_dict,
            'metrics': metrics
        }
    
    # Run iterations (parallel if requested)
    results = []
    
    if n_parallel > 1:
        with mp.Pool(n_parallel) as pool:
            results = list(tqdm(
                pool.imap(run_single_iteration, range(n_iterations)),
                total=n_iterations,
                desc="Monte Carlo Progress"
            ))
    else:
        for i in tqdm(range(n_iterations), desc="Monte Carlo Progress"):
            results.append(run_single_iteration(i))
    
    logger.info(f"Completed Monte Carlo simulation with {len(results)} results")
    return results


def calculate_simulation_metrics(state: SimulationState) -> Dict[str, Any]:
    """
    Calculate metrics summarizing the simulation results.
    
    Parameters:
        state: Final simulation state
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
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
    
    # Voting outcome metrics
    accepted_actions = sum(1 for action in state.actions.values() if action.accepted)
    total_actions = len(state.actions)
    metrics['action_acceptance_rate'] = accepted_actions / total_actions if total_actions > 0 else 0
    
    vetoed_actions = sum(1 for action in state.actions.values() if action.vetoed)
    metrics['action_veto_rate'] = vetoed_actions / total_actions if total_actions > 0 else 0
    
    # DRep type performance metrics
    drep_types = {}
    for drep_id, drep in state.dreps.items():
        drep_type = drep.profile.type
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
    
    # Reward distribution metrics
    all_rewards = []
    for epoch in range(state.current_epoch):
        if epoch in state.rewards:
            all_rewards.extend(list(state.rewards[epoch].values()))
    
    metrics['reward_gini'] = calculate_gini_coefficient(all_rewards) if all_rewards else 0
    
    return metrics


#############################################################################
# Analysis and Visualization
#############################################################################

def analyze_monte_carlo_results(results: List[Dict]) -> Dict[str, Any]:
    """
    Analyze the results of a Monte Carlo simulation to find optimal parameters.
    
    Parameters:
        results: List of result dictionaries from monte_carlo_simulation
        
    Returns:
        Dictionary of analysis results
    """
    if not results:
        return {}
    
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
    for obj_name, (metric, func) in objectives.items():
        if obj_name == 'balanced':
            # Create a composite score
            composite_scores = (
                -1 * metrics_df['final_gini'] +  # Low Gini (negated)
                metrics_df['avg_participation'] +  # High participation
                metrics_df['avg_research_effort']  # High research
            ) / 3  # Average the normalized values
            
            best_idx = composite_scores.idxmax()
        else:
            best_idx = metrics_df[metric].apply(func).idxmax()
        
        optimal_params[obj_name] = {
            'parameters': params_list[best_idx],
            'metrics': metrics_list[best_idx]
        }
    
    # Sensitivity analysis - how much each parameter affects key metrics
    sensitivity = {}
    for param in params_df.columns:
        param_sensitivity = {}
        param_values = params_df[param].unique()
        
        if len(param_values) <= 1:
            continue
            
        for metric in ['final_gini', 'avg_participation', 'avg_research_effort']:
            values_metrics = []
            for val in np.linspace(params_df[param].min(), params_df[param].max(), 10):
                # Find results close to this parameter value
                closest_idx = (params_df[param] - val).abs().argsort()[:max(3, len(results)//10)]
                avg_metric = metrics_df.iloc[closest_idx][metric].mean()
                values_metrics.append((val, avg_metric))
            
            # Calculate coefficient of variation across parameter values
            metric_values = [m for _, m in values_metrics]
            param_sensitivity[metric] = np.std(metric_values) / np.mean(metric_values) if np.mean(metric_values) != 0 else 0
        
        sensitivity[param] = param_sensitivity
    
    return {
        'correlation_matrix': correlation_matrix,
        'optimal_parameters': optimal_params,
        'sensitivity_analysis': sensitivity
    }


def visualize_simulation_results(
    state: SimulationState,
    incentive_params: IncentiveParameters,
    output_dir: str = "visualizations"
) -> None:
    """
    Generate visualizations from a simulation run.
    
    Parameters:
        state: Final simulation state
        incentive_params: Incentive parameters used
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plots
    plt.style.use('ggplot')
    
    # 1. Delegation concentration over time
    epochs = list(state.delegation_concentration.keys())
    gini_values = list(state.delegation_concentration.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, gini_values, marker='o', linewidth=2)
    plt.title('Delegation Concentration (Gini Coefficient) Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Gini Coefficient')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(f"{output_dir}/delegation_concentration.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Average participation rate over time
    participation_avg = [
        np.mean(list(state.participation_rates[epoch].values())) 
        for epoch in epochs if epoch in state.participation_rates
    ]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs[:len(participation_avg)], participation_avg, marker='o', linewidth=2)
    plt.title('Average Participation Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Participation Rate')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(f"{output_dir}/participation_rate.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Rewards by DRep type
    drep_types = {}
    for drep_id, drep in state.dreps.items():
        drep_type = drep.profile.type
        if drep_type not in drep_types:
            drep_types[drep_type] = []
        drep_types[drep_type].append(drep_id)
    
    avg_rewards_by_type = {}
    for epoch in epochs:
        if epoch not in state.rewards:
            continue
            
        for drep_type, drep_ids in drep_types.items():
            if drep_type not in avg_rewards_by_type:
                avg_rewards_by_type[drep_type] = []
            
            type_rewards = [state.rewards[epoch].get(drep_id, 0) for drep_id in drep_ids]
            avg_rewards_by_type[drep_type].append(np.mean(type_rewards))
    
    plt.figure(figsize=(12, 7))
    for drep_type, rewards in avg_rewards_by_type.items():
        epochs_with_rewards = epochs[:len(rewards)]
        plt.plot(epochs_with_rewards, rewards, marker='o', linewidth=2, label=drep_type)
    
    plt.title('Average Rewards by DRep Type')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward (ADA)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/rewards_by_type.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Delegation share by DRep type
    delegation_share_by_type = {drep_type: [] for drep_type in drep_types}
    
    for epoch in epochs:
        total_delegation = sum(drep.delegation_history[epoch] for drep in state.dreps.values() 
                              if epoch < len(drep.delegation_history))
        
        for drep_type, drep_ids in drep_types.items():
            type_delegation = sum(state.dreps[drep_id].delegation_history[epoch] 
                                for drep_id in drep_ids 
                                if epoch < len(state.dreps[drep_id].delegation_history))
            
            share = type_delegation / total_delegation if total_delegation > 0 else 0
            delegation_share_by_type[drep_type].append(share)
    
    plt.figure(figsize=(12, 7))
    for drep_type, shares in delegation_share_by_type.items():
        plt.plot(epochs[:len(shares)], shares, marker='o', linewidth=2, label=drep_type)
    
    plt.title('Delegation Share by DRep Type')
    plt.xlabel('Epoch')
    plt.ylabel('Share of Total Delegation')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(f"{output_dir}/delegation_share.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Distribution of final delegations
    final_delegations = [drep.delegated_ada for drep in state.dreps.values()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(final_delegations, bins=30, alpha=0.7)
    plt.title('Distribution of Final Delegations')
    plt.xlabel('Delegation Amount (ADA)')
    plt.ylabel('Number of DReps')
    plt.grid(True)
    plt.savefig(f"{output_dir}/delegation_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Parameter importance
    params = {
        'w1_delegation': incentive_params.w1_delegation,
        'w2_participation': incentive_params.w2_participation,
        'w3_successful_votes': incentive_params.w3_successful_votes,
        'w4_veto_penalty': incentive_params.w4_veto_penalty,
        'w5_decentralization': incentive_params.w5_decentralization,
        'w6_peer_evaluation': incentive_params.w6_peer_evaluation,
        'w7_community_engagement': incentive_params.w7_community_engagement
    }
    
    plt.figure(figsize=(12, 6))
    plt.bar(params.keys(), params.values(), alpha=0.7)
    plt.title('Incentive Parameter Weights')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.savefig(f"{output_dir}/parameter_weights.png", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_monte_carlo_results(
    results: List[Dict],
    analysis: Dict[str, Any],
    output_dir: str = "monte_carlo_results"
) -> None:
    """
    Generate visualizations from Monte Carlo simulation results.
    
    Parameters:
        results: List of result dictionaries
        analysis: Analysis results from analyze_monte_carlo_results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    params_list = [r['parameters'] for r in results]
    metrics_list = [r['metrics'] for r in results]
    
    params_df = pd.DataFrame(params_list)
    metrics_df = pd.DataFrame(metrics_list)
    
    # 1. Parameter vs. key metrics correlation heatmap
    if 'correlation_matrix' in analysis:
        plt.figure(figsize=(12, 10))
        sns.heatmap(analysis['correlation_matrix'], annot=True, cmap='coolwarm', center=0)
        plt.title('Parameter-Metric Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Scatter plots for important relationships
    key_metrics = ['final_gini', 'avg_participation', 'avg_research_effort']
    key_params = ['w1_delegation', 'w2_participation', 'w3_successful_votes', 'w6_peer_evaluation']
    
    for param in key_params:
        if param not in params_df.columns:
            continue
            
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(15, 5))
        for i, metric in enumerate(key_metrics):
            if metric not in metrics_df.columns:
                continue
                
            axes[i].scatter(params_df[param], metrics_df[metric], alpha=0.6)
            axes[i].set_xlabel(param)
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{param} vs {metric}')
            axes[i].grid(True)
            
            # Add trend line
            z = np.polyfit(params_df[param], metrics_df[metric], 1)
            p = np.poly1d(z)
            axes[i].plot(params_df[param], p(params_df[param]), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/param_{param}_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Optimal parameter configurations
    if 'optimal_parameters' in analysis:
        # Plot radar chart for each optimal parameter set
        objectives = list(analysis['optimal_parameters'].keys())
        
        for obj_name in objectives:
            obj_data = analysis['optimal_parameters'][obj_name]
            params = obj_data['parameters']
            
            # Filter to just the weight parameters
            weight_params = {k: v for k, v in params.items() if k.startswith('w')}
            
            # Prepare data for radar chart
            categories = list(weight_params.keys())
            values = list(weight_params.values())
            
            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Close the loop
            angles += angles[:1]  # Close the loop
            categories += categories[:1]  # Close the loop
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
            ax.set_ylim(0, max(values) * 1.1)
            ax.grid(True)
            plt.title(f'Optimal Parameters for {obj_name}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/optimal_{obj_name}_radar.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Sensitivity analysis
    if 'sensitivity_analysis' in analysis:
        sensitivity = analysis['sensitivity_analysis']
        params = list(sensitivity.keys())
        metrics = ['final_gini', 'avg_participation', 'avg_research_effort']
        
        sensitivity_data = []
        for param in params:
            for metric in metrics:
                if metric in sensitivity[param]:
                    sensitivity_data.append({
                        'Parameter': param,
                        'Metric': metric,
                        'Sensitivity': sensitivity[param][metric]
                    })
        
        sensitivity_df = pd.DataFrame(sensitivity_data)
        
        plt.figure(figsize=(12, 8))
        pivot_table = sensitivity_df.pivot(index='Parameter', columns='Metric', values='Sensitivity')
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')
        plt.title('Parameter Sensitivity Analysis')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sensitivity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Distribution of key metrics
    for metric in key_metrics:
        if metric not in metrics_df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        plt.hist(metrics_df[metric], bins=20, alpha=0.7)
        plt.title(f'Distribution of {metric}')
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f"{output_dir}/metric_{metric}_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()


#############################################################################
# Example Usage
#############################################################################

def run_example_simulation():
    """Run a simple example simulation to demonstrate the framework."""
    logger.info("Starting example simulation")
    
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
    
    # Initialize and run simulation
    sim_state = initialize_simulation(sim_params)
    
    for epoch in range(sim_params.total_epochs):
        sim_state = run_epoch(sim_state, incentive_params, sim_params)
    
    # Generate visualizations
    visualize_simulation_results(sim_state, incentive_params)
    
    logger.info("Example simulation completed")
    return sim_state


def run_example_monte_carlo():
    """Run a simple example Monte Carlo simulation."""
    logger.info("Starting example Monte Carlo simulation")
    
    # Define parameter ranges to explore
    param_ranges = {
        'w1_delegation': (0.1, 0.5),
        'w2_participation': (0.1, 0.5),
        'w3_successful_votes': (0.1, 0.5),
        'w4_veto_penalty': (0.01, 0.2),
        'w5_decentralization': (0.05, 0.3),
        'w6_peer_evaluation': (0.05, 0.3),
        'w7_community_engagement': (0.05, 0.3),
        'total_reward_per_epoch': (50000, 200000),
        'veto_penalty_factor': (0.1, 1.0),
        'min_participation_threshold': (0.05, 0.3)
    }
    
    # Base simulation parameters
    base_sim_params = SimulationParameters(
        total_epochs=10,  # Shorter for Monte Carlo
        drep_count=50,    # Fewer DReps for efficiency
        governance_actions_per_epoch=(3, 8),
        total_ada_delegated=1000000000,
        delegation_change_rate=0.05,
        veto_probability=0.05,
        drep_type_distribution={
            "Professional": 0.2,
            "Hobbyist": 0.4,
            "Passive": 0.3,
            "Strategic": 0.1
        },
        initial_delegation_distribution="pareto",
        delegation_distribution_params={"alpha": 1.5},
        peer_evaluation_sample_size=3
    )
    
    # Run Monte Carlo simulation (reduced iterations for example)
    results = run_monte_carlo_simulation(
        n_iterations=20,  # Use more iterations for real analysis
        epochs_per_iteration=10,
        param_ranges=param_ranges,
        base_sim_params=base_sim_params,
        n_parallel=4  # Adjust based on available CPU cores
    )
    
    # Analyze results
    analysis = analyze_monte_carlo_results(results)
    
    # Generate visualizations
    visualize_monte_carlo_results(results, analysis)
    
    # Save results for further analysis
    with open('monte_carlo_results.pkl', 'wb') as f:
        pickle.dump({'results': results, 'analysis': analysis}, f)
    
    logger.info("Monte Carlo simulation completed")
    return results, analysis


if __name__ == "__main__":
    # Run example simulations
    print("Running example simulation...")
    sim_state = run_example_simulation()
    
    print("\nRunning example Monte Carlo simulation...")
    results, analysis = run_example_monte_carlo()
    
    print("\nSimulation complete. Check the visualization folders for results.")