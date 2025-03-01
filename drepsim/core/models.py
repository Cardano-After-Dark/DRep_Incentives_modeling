"""
Core data models for the DRep simulation framework.

This module contains the dataclasses that represent the key entities
in the simulation, including DReps, governance actions, votes, and
simulation parameters.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np


@dataclass
class GovernanceAction:
    """
    Represents a governance proposal that DReps can vote on.
    
    Attributes:
        id: Unique identifier for the action
        title: Title of the governance action
        description: Detailed description of the action
        complexity: How difficult it is to evaluate (0-1)
        community_impact: Expected impact on ecosystem (0-1)
        creation_epoch: Epoch when this action was created
        voting_periods: How many epochs this is available for voting
        category: Category of the action (e.g., "Technical", "Treasury")
        choices: Available voting options
        outcomes: Actual voting outcomes
        accepted: Whether the proposal was ultimately accepted
        vetoed: Whether it was vetoed by the Constitutional Committee
    """
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
        """
        Check if this action is currently available for voting.
        
        Args:
            current_epoch: The current epoch number
            
        Returns:
            True if the action is active in the current epoch, False otherwise
        """
        return (
            current_epoch >= self.creation_epoch and 
            current_epoch < self.creation_epoch + self.voting_periods
        )


@dataclass
class DRepVote:
    """
    Records a vote by a DRep on a governance action.
    
    Attributes:
        drep_id: ID of the DRep who cast the vote
        action_id: ID of the governance action being voted on
        choice: The vote choice (e.g., "Yes", "No", "Abstain")
        epoch: Epoch when the vote was cast
        research_effort: How much effort the DRep put into research (0-1)
        rationale: Explanation for the vote
        aligned_with_outcome: Whether vote aligned with final outcome
    """
    drep_id: str
    action_id: str
    choice: str
    epoch: int
    research_effort: float  # How much effort the DRep put into research (0-1)
    rationale: str  # Explanation for the vote
    aligned_with_outcome: bool = False  # Whether vote aligned with final outcome
    
    
@dataclass
class PeerEvaluation:
    """
    Records a DRep's evaluation of another DRep's voting rationale.
    
    Attributes:
        evaluator_id: ID of the DRep providing the evaluation
        evaluated_id: ID of the DRep being evaluated
        action_id: ID of the governance action
        score: Quality score for the rationale (0-1)
        epoch: Epoch when the evaluation was made
    """
    evaluator_id: str
    evaluated_id: str
    action_id: str
    score: float  # 0-1 score of the quality of the rationale
    epoch: int


@dataclass
class DRepProfile:
    """
    Static characteristics of a DRep that don't change during simulation.
    
    Attributes:
        id: Unique identifier for the DRep
        type: Type of DRep ("Professional", "Hobbyist", "Passive", "Strategic")
        base_participation_rate: Probability of voting on an action (0-1)
        base_research_capacity: Maximum research effort possible (0-1)
        base_community_engagement: Baseline community involvement (0-1)
        adaptation_rate: How quickly the DRep adapts strategies (0-1)
        profit_motivation: How motivated by financial rewards (0-1)
        community_motivation: How motivated by community benefit (0-1)
        reputation_motivation: How motivated by reputation/status (0-1)
        region: Geographic region
        stake_pool_operator: Whether the DRep is also an SPO
        constitutional_committee: Whether the DRep is on the Constitutional Committee
    """
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

    def __init__(
        self,
        id: str,
        drep_type: str,
        reliability: float,
        research_capacity: float,
        strategic_voting: float,
        community_involvement: float,
        constitutional_committee: bool = False,
        profit_motivation: float = 0.5,
        community_motivation: float = 0.5,
        reputation_motivation: float = 0.5,
        region: str = "Unknown",
        stake_pool_operator: bool = False
    ):
        self.id = id
        self.drep_type = drep_type
        self.reliability = reliability
        self.research_capacity = research_capacity
        self.strategic_voting = strategic_voting
        self.community_involvement = community_involvement
        self.constitutional_committee = constitutional_committee
        self.profit_motivation = profit_motivation
        self.community_motivation = community_motivation
        self.reputation_motivation = reputation_motivation
        self.region = region
        self.stake_pool_operator = stake_pool_operator
        
        # Initialize strategy weights based on DRep type
        # These determine how much each incentive component influences behavior
        self.strategy_weights = {
            "delegation": 0.0,
            "participation": 0.0,
            "research_effort": 0.0,
            "successful_votes": 0.0,
            "veto_avoidance": 0.0,
            "peer_evaluation": 0.0,
            "community_engagement": 0.0
        }
        
        # Set strategy weights based on DRep type
        if drep_type == "Professional":
            self.strategy_weights = {
                "delegation": 0.3,
                "participation": 0.2,
                "research_effort": 0.2,
                "successful_votes": 0.2,
                "veto_avoidance": 0.1,
                "peer_evaluation": 0.1,
                "community_engagement": 0.1
            }
        elif drep_type == "Hobbyist":
            self.strategy_weights = {
                "delegation": 0.1,
                "participation": 0.3,
                "research_effort": 0.2,
                "successful_votes": 0.2,
                "veto_avoidance": 0.1,
                "peer_evaluation": 0.2,
                "community_engagement": 0.1
            }
        elif drep_type == "Passive":
            self.strategy_weights = {
                "delegation": 0.5,
                "participation": 0.1,
                "research_effort": 0.1,
                "successful_votes": 0.1,
                "veto_avoidance": 0.2,
                "peer_evaluation": 0.0,
                "community_engagement": 0.1
            }
        elif drep_type == "Strategic":
            self.strategy_weights = {
                "delegation": 0.4,
                "participation": 0.1,
                "research_effort": 0.1,
                "successful_votes": 0.3,
                "veto_avoidance": 0.1,
                "peer_evaluation": 0.0,
                "community_engagement": 0.1
            }


@dataclass
class DRepState:
    """
    Dynamic state of a DRep that evolves during simulation.
    
    Attributes:
        profile: Static profile of the DRep
        delegated_ada: Amount of ADA delegated to this DRep
        current_participation_rate: Current epoch participation rate
        current_research_effort: Current epoch research effort
        current_community_engagement: Current epoch community engagement
        delegation_history: History of delegations by epoch
        vote_history: History of votes cast
        rewards_history: History of rewards received
        score_history: History of performance scores
        peer_evaluations_given: Evaluations given to other DReps
        peer_evaluations_received: Evaluations received from other DReps
        strategy_weights: Weights for different strategic components
    """
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
        # Set strategy weights based on DRep type
        if self.profile.drep_type == "Professional":
            self.strategy_weights = {
                "delegation": 0.3,
                "participation": 0.2,
                "research_effort": 0.2,
                "successful_votes": 0.2,
                "veto_avoidance": 0.1,
                "peer_evaluation": 0.1,
                "community_engagement": 0.1
            }
        elif self.profile.drep_type == "Hobbyist":
            self.strategy_weights = {
                "delegation": 0.2,
                "participation": 0.2,
                "research_effort": 0.2,
                "successful_votes": 0.1,
                "veto_avoidance": 0.1,
                "peer_evaluation": 0.1,
                "community_engagement": 0.1
            }
        elif self.profile.drep_type == "Passive":
            self.strategy_weights = {
                "delegation": 0.5,
                "participation": 0.1,
                "research_effort": 0.1,
                "successful_votes": 0.1,
                "veto_avoidance": 0.1,
                "peer_evaluation": 0.0,
                "community_engagement": 0.1
            }
        elif self.profile.drep_type == "Strategic":
            self.strategy_weights = {
                "delegation": 0.4,
                "participation": 0.1,
                "research_effort": 0.1,
                "successful_votes": 0.2,
                "veto_avoidance": 0.1,
                "peer_evaluation": 0.0,
                "community_engagement": 0.1
            }
        else:
            # Default balanced weights
            self.strategy_weights = {
                "delegation": 0.3,
                "participation": 0.15,
                "research_effort": 0.15,
                "successful_votes": 0.15,
                "veto_avoidance": 0.1,
                "peer_evaluation": 0.05,
                "community_engagement": 0.1
            }
    
    def decide_vote(self, action: GovernanceAction, incentives: 'IncentiveParameters') -> Tuple[str, float]:
        """
        Decide how to vote on a governance action.
        
        Args:
            action: The governance action to vote on
            incentives: Current incentive parameters
            
        Returns:
            Tuple of (vote choice, research effort)
        """
        # Determine if DRep participates in this vote
        vote_probability = self.profile.reliability  # Changed from base_participation_rate
        
        # Adjust based on incentives and strategy
        incentive_factor = incentives.w2_participation * self.profile.strategy_weights.get("participation", 0.2)
        vote_probability = min(0.95, vote_probability * (1 + incentive_factor * 0.5))
        
        # Skip voting with some probability
        if np.random.random() > vote_probability:
            return "Abstain", 0.0
        
        # Determine research effort
        base_effort = self.profile.research_capacity  # Changed from base_research_capacity
        
        # Adjust based on action complexity and incentives
        complexity_factor = action.complexity * 0.5
        incentive_factor = incentives.w3_successful_votes * self.profile.strategy_weights.get("research_effort", 0.2)
        
        research_effort = base_effort * (1 + complexity_factor) * (1 + incentive_factor * 0.5)
        research_effort = min(1.0, research_effort)
        
        # Determine vote based on research effort and strategic considerations
        strategic_bias = self.profile.strategic_voting  # Changed from adaptation_rate
        
        # Higher research effort leads to better decision quality
        # For simplicity, we'll assume "Yes" is the correct choice with 70% probability
        correct_choice_prob = 0.5 + (research_effort * 0.4)
        
        # Strategic voting considers what others might vote
        # For simplicity, we'll assume strategic voters lean toward "Yes"
        strategic_yes_bias = strategic_bias * 0.2
        
        # Final probability of voting "Yes"
        yes_probability = correct_choice_prob + strategic_yes_bias
        
        # Make the choice
        if np.random.random() < yes_probability:
            return "Yes", research_effort
        else:
            return "No", research_effort
    
    def evaluate_peers(self, votes: List[DRepVote], incentives: 'IncentiveParameters') -> List[PeerEvaluation]:
        """
        Evaluate the voting rationales of other DReps.
        
        Args:
            votes: List of votes to potentially evaluate
            incentives: Current incentive parameters
            
        Returns:
            List of peer evaluations created
        """
        evaluations = []
        
        # Skip if DRep doesn't prioritize peer evaluation
        if self.strategy_weights["peer_evaluation"] < 0.1:
            return evaluations
        
        # Consider only a subset of votes from other DReps (simulation optimization)
        other_votes = [v for v in votes if v.drep_id != self.profile.id]
        if not other_votes:
            return evaluations
        
        # Sample votes to evaluate based on DRep's engagement level
        sample_size = max(1, int(len(other_votes) * self.profile.reliability))  # Changed from base_participation_rate
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
            if self.profile.drep_type == "Strategic" and self.profile.profit_motivation > 0.7:  # Changed from type
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
        
        Args:
            rewards: The rewards received in the last epoch
            epoch_metrics: Dictionary of performance metrics from the last epoch
        """
        if not self.rewards_history:
            # No history yet to adapt from
            return
        
        # Only strategic DReps adapt their strategy significantly
        if self.profile.drep_type != "Strategic" and np.random.random() > self.profile.strategic_voting:
            return
        
        # Compare rewards to previous epochs
        prev_reward = self.rewards_history[-1] if len(self.rewards_history) > 1 else 0
        reward_change = rewards - prev_reward
        
        # Determine adjustment rate based on profile
        adjustment_rate = self.profile.strategic_voting * 0.2  # Small adjustment per epoch
        
        # Adjust strategy weights based on reward change
        if reward_change > 0:
            # Reward increased, reinforce current strategy
            pass
        else:
            # Reward decreased or stayed the same, adjust strategy
            # For simplicity, we'll just make small random adjustments
            keys = list(self.strategy_weights.keys())
            increase_key = np.random.choice(keys)
            decrease_key = np.random.choice([k for k in keys if k != increase_key])
            
            # Make small adjustments
            self.strategy_weights[increase_key] += adjustment_rate
            self.strategy_weights[decrease_key] -= adjustment_rate
            
            # Ensure weights stay in valid range
            self.strategy_weights[increase_key] = min(0.5, self.strategy_weights[increase_key])
            self.strategy_weights[decrease_key] = max(0.0, self.strategy_weights[decrease_key])


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
    
    Attributes:
        w1_delegation: Weight for delegation ratio
        w2_participation: Weight for participation rate
        w3_successful_votes: Weight for successful votes ratio
        w4_veto_penalty: Weight for veto penalty
        w5_decentralization: Weight for decentralization score
        w6_peer_evaluation: Weight for peer evaluation score
        w7_community_engagement: Weight for community engagement
        total_reward_per_epoch: Total ADA to distribute per epoch
        veto_penalty_factor: How much a veto reduces score
        min_participation_threshold: Minimum participation to get rewards
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
    """
    Parameters controlling the overall simulation environment.
    
    Attributes:
        total_epochs: Number of epochs to simulate
        drep_count: Number of DReps to simulate
        governance_actions_per_epoch: (min, max) actions per epoch
        total_ada_delegated: Total ADA delegated to all DReps
        delegation_change_rate: Rate at which delegations change per epoch
        veto_probability: Probability of a governance action being vetoed
        drep_type_distribution: Percentage of each DRep type
        initial_delegation_distribution: "uniform", "pareto", "log-normal"
        delegation_distribution_params: Parameters for the distribution
        peer_evaluation_sample_size: Number of peers each DRep evaluates
        random_seed: Seed for reproducibility
    """
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
    """
    The current state of the simulation across all epochs.
    
    Attributes:
        dreps: All DReps in the simulation
        actions: All governance actions
        current_epoch: Current epoch number
        active_actions: Actions by epoch
        votes: Votes by epoch
        evaluations: Evaluations by epoch
        rewards: Rewards by epoch
        scores: Scores by epoch
        delegation_concentration: Gini coefficient by epoch
        participation_rates: Participation rates by epoch and DRep
        research_efforts: Research efforts by epoch and DRep
        peer_scores: Peer scores by epoch and DRep
        community_engagements: Community engagement by epoch and DRep
    """
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