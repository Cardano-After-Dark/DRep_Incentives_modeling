"""
Incentive calculation module for the DRep simulation framework.

This module contains functions for calculating rewards based on DRep performance
and the specified incentive model.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Type
from drepsim.core.models import SimulationState, IncentiveParameters
from drepsim.core.incentive_models import (
    IncentiveModel, 
    LinearIncentiveModel,
    NonLinearIncentiveModel,
    TemporalIncentiveModel,
    GameTheoryIncentiveModel
)

# Registry of available incentive models
INCENTIVE_MODELS = {
    "linear": LinearIncentiveModel,
    "non_linear": NonLinearIncentiveModel,
    "temporal": TemporalIncentiveModel,
    "game_theory": GameTheoryIncentiveModel
}

def get_incentive_model(model_name: str) -> Type[IncentiveModel]:
    """
    Get an incentive model by name.
    
    Args:
        model_name: Name of the model to retrieve
        
    Returns:
        IncentiveModel class
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in INCENTIVE_MODELS:
        raise ValueError(f"Unknown incentive model: {model_name}. Available models: {list(INCENTIVE_MODELS.keys())}")
    return INCENTIVE_MODELS[model_name]

def calculate_rewards(
    state: SimulationState,
    incentives: IncentiveParameters,
    epoch: int,
    model_name: str = "linear"
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate reward scores and ADA rewards for all DReps based on their performance.
    
    Args:
        state: The current simulation state
        incentives: Incentive parameters
        epoch: Current epoch
        model_name: Name of the incentive model to use
        
    Returns:
        Tuple of (scores, rewards) dictionaries mapping DRep IDs to values
    """
    # Get the incentive model
    model_class = get_incentive_model(model_name)
    model = model_class()
    
    # Calculate rewards using the model
    return model.calculate_rewards(state, incentives, epoch) 