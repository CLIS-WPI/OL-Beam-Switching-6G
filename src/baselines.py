"""
Baseline algorithms: Greedy and Multi-Armed Bandit (MAB-UCB)
"""

import numpy as np
from src.config import NUM_BEAMS
from src.environment import compute_snr_from_channel


class GreedyBaseline:
    """
    Greedy baseline: Always selects the beam with highest instantaneous SNR.
    """
    def select_beam(self, h_channel, ue_idx):
        """
        Select beam for a UE based on greedy SNR maximization.
        
        Args:
            h_channel: Channel matrix (NUM_UES x NUM_ANTENNAS)
            ue_idx: Index of the UE
        
        Returns:
            Selected beam index
        """
        best_beam = 0
        best_snr = -float('inf')
        for beam_idx in range(NUM_BEAMS):
            snr = compute_snr_from_channel(h_channel, beam_idx, ue_idx, 0)
            if snr > best_snr:
                best_snr = snr
                best_beam = beam_idx
        return best_beam


class MAB_UCB:
    """
    Multi-Armed Bandit with Upper Confidence Bound (UCB) algorithm.
    """
    def __init__(self, num_ues, num_beams=NUM_BEAMS, exploration_factor=2.0):
        self.num_ues = num_ues
        self.num_beams = num_beams
        self.exploration_factor = exploration_factor
        self.counts = np.zeros((num_ues, num_beams))
        self.values = np.zeros((num_ues, num_beams))
        self.total_counts = np.zeros(num_ues)
        
    def select_beam(self, ue_idx, t):
        """
        Select beam using UCB algorithm.
        
        Args:
            ue_idx: Index of the UE
            t: Current timestep
        
        Returns:
            Selected beam index
        """
        unexplored = np.where(self.counts[ue_idx, :] == 0)[0]
        if len(unexplored) > 0:
            return np.random.choice(unexplored)
        
        total_count = self.total_counts[ue_idx]
        ucb_values = np.zeros(self.num_beams)
        
        for beam_idx in range(self.num_beams):
            count = self.counts[ue_idx, beam_idx]
            if count == 0:
                ucb_values[beam_idx] = float('inf')
            else:
                mean_reward = self.values[ue_idx, beam_idx] / count
                exploration_bonus = np.sqrt(
                    self.exploration_factor * np.log(max(1, total_count)) / count
                )
                ucb_values[beam_idx] = mean_reward + exploration_bonus
        
        return np.argmax(ucb_values)
    
    def update(self, ue_idx, beam_idx, reward):
        """
        Update UCB statistics after receiving reward.
        
        Args:
            ue_idx: Index of the UE
            beam_idx: Selected beam index
            reward: Observed reward
        """
        self.counts[ue_idx, beam_idx] += 1
        self.values[ue_idx, beam_idx] += reward
        self.total_counts[ue_idx] += 1

