"""
Configuration management for the DRep simulation framework.

This module provides functions for loading, validating, and saving
configuration files.
"""

import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("drepsim.config")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the configuration file is not valid JSON
    """
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.debug(f"Loaded configuration with {len(config)} keys")
        return config
    
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save a configuration to a file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    logger.info(f"Saving configuration to {config_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.debug(f"Saved configuration with {len(config)} keys")


def validate_simulation_config(config: Dict[str, Any]) -> bool:
    """
    Validate a simulation configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if the configuration is valid, False otherwise
    """
    required_keys = [
        "total_epochs",
        "drep_count",
        "governance_actions_per_epoch",
        "total_ada_delegated",
        "delegation_change_rate",
        "veto_probability",
        "drep_type_distribution",
        "initial_delegation_distribution"
    ]
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration key: {key}")
            return False
    
    # Validate specific values
    if config["total_epochs"] <= 0:
        logger.error("total_epochs must be positive")
        return False
    
    if config["drep_count"] <= 0:
        logger.error("drep_count must be positive")
        return False
    
    if not isinstance(config["governance_actions_per_epoch"], list) or len(config["governance_actions_per_epoch"]) != 2:
        logger.error("governance_actions_per_epoch must be a list of two integers")
        return False
    
    if config["total_ada_delegated"] <= 0:
        logger.error("total_ada_delegated must be positive")
        return False
    
    if not 0 <= config["delegation_change_rate"] <= 1:
        logger.error("delegation_change_rate must be between 0 and 1")
        return False
    
    if not 0 <= config["veto_probability"] <= 1:
        logger.error("veto_probability must be between 0 and 1")
        return False
    
    if not isinstance(config["drep_type_distribution"], dict):
        logger.error("drep_type_distribution must be a dictionary")
        return False
    
    type_sum = sum(config["drep_type_distribution"].values())
    if abs(type_sum - 1.0) > 0.001:
        logger.error(f"drep_type_distribution values must sum to 1.0 (got {type_sum})")
        return False
    
    return True


def validate_incentive_config(config: Dict[str, Any]) -> bool:
    """
    Validate an incentive configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if the configuration is valid, False otherwise
    """
    weight_keys = [
        "w1_delegation",
        "w2_participation",
        "w3_successful_votes",
        "w4_veto_penalty",
        "w5_decentralization",
        "w6_peer_evaluation",
        "w7_community_engagement"
    ]
    
    # Check weight keys
    for key in weight_keys:
        if key not in config:
            logger.error(f"Missing required weight parameter: {key}")
            return False
        
        if not 0 <= config[key] <= 1:
            logger.error(f"{key} must be between 0 and 1")
            return False
    
    # Check that weights sum to approximately 1
    weight_sum = sum(config[key] for key in weight_keys)
    if abs(weight_sum - 1.0) > 0.001:
        logger.error(f"Weight parameters must sum to 1.0 (got {weight_sum})")
        return False
    
    # Check other required parameters
    if "total_reward_per_epoch" not in config:
        logger.error("Missing required parameter: total_reward_per_epoch")
        return False
    
    if config["total_reward_per_epoch"] < 0:
        logger.error("total_reward_per_epoch must be non-negative")
        return False
    
    if "veto_penalty_factor" not in config:
        logger.error("Missing required parameter: veto_penalty_factor")
        return False
    
    if config["veto_penalty_factor"] < 0:
        logger.error("veto_penalty_factor must be non-negative")
        return False
    
    if "min_participation_threshold" not in config:
        logger.error("Missing required parameter: min_participation_threshold")
        return False
    
    if not 0 <= config["min_participation_threshold"] <= 1:
        logger.error("min_participation_threshold must be between 0 and 1")
        return False
    
    return True


def get_default_simulation_config() -> Dict[str, Any]:
    """
    Get the default simulation configuration.
    
    Returns:
        Default simulation configuration dictionary
    """
    return {
        "total_epochs": 20,
        "drep_count": 100,
        "governance_actions_per_epoch": [5, 15],
        "total_ada_delegated": 1000000000,
        "delegation_change_rate": 0.05,
        "veto_probability": 0.05,
        "drep_type_distribution": {
            "Professional": 0.2,
            "Hobbyist": 0.4,
            "Passive": 0.3,
            "Strategic": 0.1
        },
        "initial_delegation_distribution": "pareto",
        "delegation_distribution_params": {"alpha": 1.5},
        "peer_evaluation_sample_size": 5,
        "random_seed": 42
    }


def get_default_incentive_config() -> Dict[str, Any]:
    """
    Get the default incentive configuration.
    
    Returns:
        Default incentive configuration dictionary
    """
    return {
        "w1_delegation": 0.2,
        "w2_participation": 0.3,
        "w3_successful_votes": 0.2,
        "w4_veto_penalty": 0.05,
        "w5_decentralization": 0.1,
        "w6_peer_evaluation": 0.1,
        "w7_community_engagement": 0.05,
        "total_reward_per_epoch": 100000,
        "veto_penalty_factor": 0.5,
        "min_participation_threshold": 0.1
    } 