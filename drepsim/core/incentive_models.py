"""
Incentive model implementations for the DRep simulation framework.

This module contains various incentive model implementations ranging from
simple linear models to sophisticated game-theoretic and Bayesian approaches.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, List, Optional
from drepsim.core.models import SimulationState, IncentiveParameters

class IncentiveModel(ABC):
    """Base abstract class for incentive models."""
    
    @abstractmethod
    def calculate_rewards(
        self, 
        state: SimulationState,
        incentive_params: IncentiveParameters,
        epoch: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate reward scores and ADA rewards for all DReps.
        
        Args:
            state: The current simulation state
            incentive_params: Incentive parameters
            epoch: Current epoch
            
        Returns:
            Tuple of (scores, rewards) dictionaries mapping DRep IDs to values
        """
        pass 

class LinearIncentiveModel(IncentiveModel):
    """
    Basic linear incentive model using weighted sum of components.
    
    This is the original model implemented in the simulation.
    """
    
    def calculate_rewards(
        self, 
        state: SimulationState,
        incentives: IncentiveParameters,
        epoch: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
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
            veto_component = -incentives.w4_veto_penalty * veto_penalty
            
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

class NonLinearIncentiveModel(IncentiveModel):
    """
    Non-linear incentive model with diminishing returns.
    
    This model implements:
    1. Logarithmic/square root scaling for delegation (anti-plutocratic)
    2. Threshold-based participation bonuses
    3. Quadratic voting principles
    """
    
    def calculate_rewards(
        self, 
        state: SimulationState,
        incentives: IncentiveParameters,
        epoch: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        scores = {}
        rewards = {}
        
        # Skip if no DReps or no rewards
        if not state.dreps or incentives.total_reward_per_epoch <= 0:
            return scores, rewards
        
        # Get total delegated ADA
        total_delegated = sum(drep.delegated_ada for drep in state.dreps.values())
        
        # Calculate scores for each DRep
        for drep_id, drep in state.dreps.items():
            # 1. Non-linear delegation component (square root scaling)
            delegation_ratio = drep.delegated_ada / total_delegated if total_delegated > 0 else 0
            # Apply square root scaling for diminishing returns
            delegation_component = incentives.w1_delegation * np.sqrt(delegation_ratio)
            
            # 2. Threshold-based participation component
            participation_rate = state.participation_rates.get(epoch, {}).get(drep_id, 0)
            # Base participation component
            participation_component = incentives.w2_participation * participation_rate
            
            # Add bonus for high participation (threshold bonus)
            if participation_rate >= 0.8:
                participation_component *= 1.2  # 20% bonus for 80%+ participation
            
            # 3. Successful votes with quadratic scaling
            drep_votes = [v for v in state.votes.get(epoch, []) if v.drep_id == drep_id]
            total_votes = len(drep_votes)
            
            successful_votes = 0
            research_effort_sum = 0
            for vote in drep_votes:
                action = state.actions.get(vote.action_id)
                if action and ((action.accepted and vote.choice == "Yes") or 
                              (not action.accepted and vote.choice == "No")):
                    successful_votes += 1
                    research_effort_sum += vote.research_effort
            
            # Success ratio weighted by research effort
            if total_votes > 0:
                avg_research = research_effort_sum / total_votes if total_votes > 0 else 0
                successful_votes_ratio = successful_votes / total_votes
                # Square the success ratio weighted by research effort (rewards quality)
                successful_component = incentives.w3_successful_votes * (successful_votes_ratio * (0.5 + 0.5 * avg_research)) ** 2
            else:
                successful_component = 0
            
            # 4. Veto penalty component with progressive penalty
            vetoed_votes = 0
            for vote in drep_votes:
                action = state.actions.get(vote.action_id)
                if action and action.vetoed and vote.choice == "Yes":
                    vetoed_votes += 1
            
            # Progressive penalty (grows quadratically with number of vetoed votes)
            veto_penalty = vetoed_votes * incentives.veto_penalty_factor
            veto_component = -incentives.w4_veto_penalty * (veto_penalty ** 2)
            
            # 5. Decentralization score component
            decentralization_score = 1.0
            if drep.profile.constitutional_committee:
                decentralization_score -= 0.3
            if drep.profile.stake_pool_operator:
                decentralization_score -= 0.5
            
            decentralization_component = incentives.w5_decentralization * decentralization_score
            
            # 6. Peer evaluation with confidence weighting
            peer_score = state.peer_scores.get(epoch, {}).get(drep_id, 0)
            # Get the number of evaluations received
            peer_eval_count = sum(1 for e in state.evaluations.get(epoch, []) 
                                 if e.evaluated_id == drep_id)
            
            # Weight peer score by the square root of the number of evaluations (more evals = more confidence)
            confidence_factor = min(1.0, np.sqrt(peer_eval_count / 5))  # Normalize to reasonable threshold
            peer_component = incentives.w6_peer_evaluation * peer_score * confidence_factor
            
            # 7. Community engagement with threshold bonuses
            engagement_score = state.community_engagements.get(epoch, {}).get(drep_id, 0)
            if engagement_score > 0.7:
                # Bonus for highly engaged DReps
                engagement_component = incentives.w7_community_engagement * engagement_score * 1.3
            else:
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

class TemporalIncentiveModel(IncentiveModel):
    """
    Time-dependent incentive model that considers historical performance.
    
    This model implements:
    1. Consistency bonuses for sustained participation
    2. Reputation system based on historical performance
    3. Time-weighted success metrics
    """
    
    def calculate_rewards(
        self, 
        state: SimulationState,
        incentives: IncentiveParameters,
        epoch: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        scores = {}
        rewards = {}
        
        # Skip if no DReps or no rewards
        if not state.dreps or incentives.total_reward_per_epoch <= 0:
            return scores, rewards
        
        # Skip for epoch 0 since we need history
        if epoch == 0:
            # Fall back to linear model for first epoch
            linear_model = LinearIncentiveModel()
            return linear_model.calculate_rewards(state, incentives, epoch)
        
        # Get total delegated ADA
        total_delegated = sum(drep.delegated_ada for drep in state.dreps.values())
        
        # Look back window (how many epochs to consider for history)
        look_back = min(epoch, 5)  # Look back up to 5 epochs
        
        # Calculate scores for each DRep
        for drep_id, drep in state.dreps.items():
            # 1. Delegation component
            delegation_ratio = drep.delegated_ada / total_delegated if total_delegated > 0 else 0
            delegation_component = incentives.w1_delegation * delegation_ratio
            
            # 2. Participation with consistency bonus
            current_participation = state.participation_rates.get(epoch, {}).get(drep_id, 0)
            
            # Calculate participation history
            participation_history = []
            for e in range(epoch - look_back, epoch):
                if e in state.participation_rates and drep_id in state.participation_rates[e]:
                    participation_history.append(state.participation_rates[e][drep_id])
            
            # Calculate consistency bonus (exponential moving average)
            if participation_history:
                alpha = 0.7  # Weight for EMA
                historical_participation = participation_history[0]
                for p in participation_history[1:]:
                    historical_participation = alpha * p + (1 - alpha) * historical_participation
                
                # Bonus for consistent participation
                consistency_factor = 1.0 + 0.2 * min(1.0, historical_participation)
            else:
                consistency_factor = 1.0
            
            participation_component = incentives.w2_participation * current_participation * consistency_factor
            
            # 3. Successful votes with historical weighting
            current_votes = [v for v in state.votes.get(epoch, []) if v.drep_id == drep_id]
            total_current_votes = len(current_votes)
            
            # Count successful current votes
            current_successful = 0
            for vote in current_votes:
                action = state.actions.get(vote.action_id)
                if action and ((action.accepted and vote.choice == "Yes") or 
                              (not action.accepted and vote.choice == "No")):
                    current_successful += 1
            
            current_success_ratio = current_successful / total_current_votes if total_current_votes > 0 else 0
            
            # Calculate historical success ratio
            historical_votes = []
            historical_successes = 0
            for e in range(epoch - look_back, epoch):
                e_votes = [v for v in state.votes.get(e, []) if v.drep_id == drep_id]
                for vote in e_votes:
                    historical_votes.append(vote)
                    action = state.actions.get(vote.action_id)
                    if action and ((action.accepted and vote.choice == "Yes") or 
                                  (not action.accepted and vote.choice == "No")):
                        historical_successes += 1
            
            historical_success_ratio = historical_successes / len(historical_votes) if historical_votes else 0
            
            # Combine current and historical with time decay (70% current, 30% historical)
            success_ratio = 0.7 * current_success_ratio + 0.3 * historical_success_ratio
            successful_component = incentives.w3_successful_votes * success_ratio
            
            # 4. Veto penalty with memory
            current_vetoes = 0
            for vote in current_votes:
                action = state.actions.get(vote.action_id)
                if action and action.vetoed and vote.choice == "Yes":
                    current_vetoes += 1
            
            # Calculate historical vetoes
            historical_vetoes = 0
            for e in range(epoch - look_back, epoch):
                e_votes = [v for v in state.votes.get(e, []) if v.drep_id == drep_id]
                for vote in e_votes:
                    action = state.actions.get(vote.action_id)
                    if action and action.vetoed and vote.choice == "Yes":
                        historical_vetoes += 1
            
            # Combine with time decay (recent vetoes matter more)
            decay_factor = 0.8  # Decay rate for historical vetoes
            veto_penalty = current_vetoes + historical_vetoes * decay_factor
            veto_component = -incentives.w4_veto_penalty * veto_penalty * incentives.veto_penalty_factor
            
            # 5. Decentralization score component (static)
            decentralization_score = 1.0
            if drep.profile.constitutional_committee:
                decentralization_score -= 0.3
            if drep.profile.stake_pool_operator:
                decentralization_score -= 0.5
            
            decentralization_component = incentives.w5_decentralization * decentralization_score
            
            # 6. Reputation system based on peer evaluations over time
            current_peer_score = state.peer_scores.get(epoch, {}).get(drep_id, 0)
            
            # Calculate historical peer scores
            historical_peer_scores = []
            for e in range(epoch - look_back, epoch):
                if e in state.peer_scores and drep_id in state.peer_scores[e]:
                    historical_peer_scores.append(state.peer_scores[e][drep_id])
            
            # Calculate reputation score
            if historical_peer_scores:
                # Exponential growth for consistently good peer evaluations
                reputation_score = current_peer_score * (1.0 + 0.1 * len(historical_peer_scores) * 
                                                       np.mean(historical_peer_scores))
            else:
                reputation_score = current_peer_score
            
            peer_component = incentives.w6_peer_evaluation * reputation_score
            
            # 7. Community engagement with growth factor
            current_engagement = state.community_engagements.get(epoch, {}).get(drep_id, 0)
            
            # Calculate historical engagement
            historical_engagement = []
            for e in range(epoch - look_back, epoch):
                if e in state.community_engagements and drep_id in state.community_engagements[e]:
                    historical_engagement.append(state.community_engagements[e][drep_id])
            
            # Calculate engagement trajectory
            if historical_engagement:
                # Reward improvement in engagement
                engagement_trend = current_engagement - np.mean(historical_engagement)
                growth_factor = 1.0 + max(0, engagement_trend * 2)  # Bonus for improvement
            else:
                growth_factor = 1.0
            
            engagement_component = incentives.w7_community_engagement * current_engagement * growth_factor
            
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
            if current_participation < incentives.min_participation_threshold:
                total_score = 0
            
            scores[drep_id] = max(0, total_score)  # Ensure non-negative
        
        # Calculate rewards based on scores
        total_score = sum(scores.values())
        
        if total_score > 0:
            for drep_id, score in scores.items():
                reward_share = score / total_score
                rewards[drep_id] = reward_share * incentives.total_reward_per_epoch
        
        return scores, rewards 

class GameTheoryIncentiveModel(IncentiveModel):
    """
    Game theory-based incentive model.
    
    This model implements:
    1. Quadratic funding principles
    2. Schelling point mechanisms
    3. Coalition detection and prevention
    """
    
    def calculate_rewards(
        self, 
        state: SimulationState,
        incentives: IncentiveParameters,
        epoch: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        scores = {}
        rewards = {}
        
        # Skip if no DReps or no rewards
        if not state.dreps or incentives.total_reward_per_epoch <= 0:
            return scores, rewards
        
        # Get total delegated ADA
        total_delegated = sum(drep.delegated_ada for drep in state.dreps.values())
        
        # Get all votes for this epoch
        epoch_votes = state.votes.get(epoch, [])
        
        # Create a mapping of action_id -> list of votes for coalition detection
        action_votes = {}
        for vote in epoch_votes:
            if vote.action_id not in action_votes:
                action_votes[vote.action_id] = []
            action_votes[vote.action_id].append(vote)
        
        # Calculate scores for each DRep
        for drep_id, drep in state.dreps.items():
            # 1. Delegation with quadratic funding principle
            delegation_ratio = drep.delegated_ada / total_delegated if total_delegated > 0 else 0
            # Square root scaling for quadratic funding
            delegation_component = incentives.w1_delegation * np.sqrt(delegation_ratio)
            
            # 2. Participation rate component
            participation_rate = state.participation_rates.get(epoch, {}).get(drep_id, 0)
            participation_component = incentives.w2_participation * participation_rate
            
            # 3. Schelling point success 
            drep_votes = [v for v in epoch_votes if v.drep_id == drep_id]
            total_votes = len(drep_votes)
            
            schelling_score = 0
            for vote in drep_votes:
                action = state.actions.get(vote.action_id)
                if not action:
                    continue
                
                # Get all votes for this action
                action_vote_list = action_votes.get(vote.action_id, [])
                
                # Count votes for each choice
                yes_votes = sum(1 for v in action_vote_list if v.choice == "Yes")
                no_votes = sum(1 for v in action_vote_list if v.choice == "No")
                abstain_votes = sum(1 for v in action_vote_list if v.choice == "Abstain")
                
                # Calculate majority percentage
                total_action_votes = yes_votes + no_votes + abstain_votes
                if total_action_votes <= 1:  # Skip if this is the only vote
                    continue
                
                # Determine majority vote
                if yes_votes >= no_votes and yes_votes >= abstain_votes:
                    majority_choice = "Yes"
                    majority_percentage = yes_votes / total_action_votes
                elif no_votes >= yes_votes and no_votes >= abstain_votes:
                    majority_choice = "No"
                    majority_percentage = no_votes / total_action_votes
                else:
                    majority_choice = "Abstain"
                    majority_percentage = abstain_votes / total_action_votes
                
                # Reward for voting with qualified majority (stronger reward for stronger consensus)
                if vote.choice == majority_choice and majority_percentage > 0.5:
                    # More reward for being part of a strong consensus
                    confidence_factor = (majority_percentage - 0.5) * 2  # Scales 0-1 for 50-100% majority
                    # Also weight by research effort
                    schelling_score += vote.research_effort * (1 + confidence_factor)
            
            # Normalize by number of votes
            schelling_component = 0
            if total_votes > 0:
                avg_schelling_score = schelling_score / total_votes
                schelling_component = incentives.w3_successful_votes * avg_schelling_score
            
            # 4. Coalition detection and penalty
            coalition_penalty = 0
            
            # Skip coalition detection for DReps with few votes
            if total_votes >= 3:
                # For each other DRep, calculate vote correlation
                vote_correlations = []
                
                for other_drep_id, other_drep in state.dreps.items():
                    if other_drep_id == drep_id:
                        continue
                    
                    other_votes = [v for v in epoch_votes if v.drep_id == other_drep_id]
                    
                    # Find actions voted on by both DReps
                    common_actions = set(v.action_id for v in drep_votes) & set(v.action_id for v in other_votes)
                    
                    if len(common_actions) >= 3:  # Need some minimum overlap
                        # Create vote choice maps
                        drep_choices = {v.action_id: v.choice for v in drep_votes if v.action_id in common_actions}
                        other_choices = {v.action_id: v.choice for v in other_votes if v.action_id in common_actions}
                        
                        # Count matching votes
                        matches = sum(1 for action_id in common_actions 
                                     if drep_choices[action_id] == other_choices[action_id])
                        
                        # Calculate correlation
                        correlation = matches / len(common_actions)
                        vote_correlations.append(correlation)
                
                # Check for suspiciously high correlations (potential coalition)
                if vote_correlations and max(vote_correlations) > 0.9:
                    coalition_penalty = max(vote_correlations) * 0.5  # Penalty scales with correlation strength
            
            veto_component = -incentives.w4_veto_penalty * coalition_penalty
            
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
                schelling_component +
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