"""
Configuration file: Single Source of Truth for all constants and reward profiles
"""

import numpy as np

# =================== PHYSICS CONSTANTS ===================
NUM_UES = 100
NUM_ANTENNAS = 64
NUM_BEAMS = NUM_ANTENNAS
ROAD_LENGTH = 500.0
BS_POSITION = ROAD_LENGTH / 2
CARRIER_FREQ = 28e9
TX_POWER_DBM = 38
NOISE_POWER_DBM = -80
TIMESTEP_DURATION = 0.01
PATH_LOSS_EXPONENT = 3.0
BLOCKAGE_ATTENUATION_DB = 12.0
VELOCITY_NORMALIZATION_FACTOR = 50.0

# =================== TRAINING CONSTANTS ===================
NUM_TIMESTEPS = 20000       # Production setting
EVAL_TIMESTEPS = 1000       # Production setting
BUFFER_SIZE = 100000        # Increased for H100 memory (better sample diversity)

# --- CRITICAL FIXES FOR H100 STABILITY ---
BATCH_SIZE = 512            # Reduced from 4096. 512 is the "Sweet Spot" for H100 in RL.
                            # It saturates the CUDA cores enough without destroying convergence.
HIDDEN_SIZE = 512           # Increased from 256. 
                            # Uses H100 compute power to learn more complex features 
                            # without overfitting like 768/1024 might.
LEARNING_RATE = 0.0003      # Reduced from 0.002. 
                            # Essential for stability given the heavy switch_penalty (-40).
GAMMA = 0.99                # Standard for long-term stability
SNR_THRESHOLD = 6.0
TARGET_UPDATE_FREQ = 100    # Increased frequency to stabilize the moving target
STATE_SIZE = 8
BASE_SEED = 42

# =================== BLOCKAGE SCENARIOS ===================
BLOCKAGE_SCENARIOS = [
    {"P_BB": 0.25, "P_UB": 0.08, "name": "realistic"},
]

# =================== REWARD PROFILES ===================
# REVIEWER REQUIREMENT: REWARD PROFILES including Vanilla Baseline
REWARD_PROFILES = {
    "high_stability": {
        "name": "High-Stability",
        "snr_weight": 0.20,
        "stability_weight": 0.75,
        "fairness_weight": 0.05,
        "switch_penalty": 5.0,
        "snr_bonus_threshold": 12.0,
        "epsilon_start": 0.95,
        "epsilon_min": 0.08,
        "epsilon_decay": 0.9998,
        "max_beam_change": 8,
        "description": "Maximum stability - minimal switching"
    },
    "balanced": {
        "name": "Balanced",
        "snr_weight": 0.40,           # Reduced from 0.55 to reduce greediness
        "stability_weight": 2.5,       # Aggressive increase from 1.0 for superior stability
        "fairness_weight": 0.15,       # Slight increase to maintain coverage
        "switch_penalty": 40.0,        # Heavy penalty (from 15.0) - agent learns to avoid switching
        "snr_bonus_threshold": 8.0,   # Only reward if SNR is really good
        "epsilon_start": 0.7,          # Reduced initial exploration for faster convergence to stability
        "epsilon_min": 0.05,
        "epsilon_decay": 0.9997,
        "max_beam_change": 8,          # Physical constraint on angle change (prevents sudden jumps)
        "description": "Optimized for Superior Stability over MAB"
    },
    # REVIEWER 2 REQUEST: VANILLA BASELINE
    "vanilla": {
        "name": "Vanilla-Throughput",
        "snr_weight": 1.0,
        "stability_weight": 0.0,      # Zero stability focus
        "fairness_weight": 0.0,
        "switch_penalty": 0.0,        # No penalty
        "snr_bonus_threshold": 0.0,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.999,
        "max_beam_change": NUM_BEAMS,  # No constraints
        "description": "Baseline: Pure SNR Maximization"
    },
    "high_coverage": {
        "name": "High-Coverage",
        "snr_weight": 0.85,
        "stability_weight": 0.05,
        "fairness_weight": 0.10,
        "switch_penalty": 0.5,
        "snr_bonus_threshold": 12.0,
        "epsilon_start": 0.98,
        "epsilon_min": 0.02,
        "epsilon_decay": 0.9999,
        "max_beam_change": NUM_BEAMS,
        "description": "Maximum coverage - aggressive switching"
    }
}

