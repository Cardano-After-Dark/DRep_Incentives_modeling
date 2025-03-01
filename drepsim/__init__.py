"""
DRep Incentivization Simulation Framework
=========================================

A framework for simulating and analyzing incentive structures for 
Delegated Representatives (DReps) in blockchain governance ecosystems.

This package combines Monte Carlo methods with agent-based modeling to evaluate
different incentive mechanisms and their impact on governance outcomes.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from drepsim.core.models import (
    GovernanceAction, DRepVote, PeerEvaluation, 
    DRepProfile, DRepState, IncentiveParameters, 
    SimulationParameters, SimulationState
)
from drepsim.core.simulation import (
    initialize_simulation, run_epoch, run_simulation
) 