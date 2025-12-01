"""
Core data structures: ReplayBuffer and StateHistory with Ablation support
"""

import numpy as np
from collections import deque
from src.config import VELOCITY_NORMALIZATION_FACTOR


class PredictiveStateHistory:
    """
    Maintains historical state information for enhanced feature extraction.
    
    REVIEWER 1 & 3 REQUEST: Supports ablation mode to disable history features
    for testing the importance of GRU/memory components.
    """
    def __init__(self, num_ues, window=20):
        self.window = window
        self.num_ues = num_ues
        self.snr_history = deque(maxlen=window)
        self.beam_history = deque(maxlen=window)
        self.blockage_history = deque(maxlen=window)
        self.position_history = deque(maxlen=window)
        self.velocity_history = deque(maxlen=window)
        
    def update(self, snr, beams, blockage, positions, velocities):
        """Update history with new observations"""
        self.snr_history.append(snr)
        self.beam_history.append(beams)
        self.blockage_history.append(blockage)
        self.position_history.append(positions)
        self.velocity_history.append(velocities)
    
    def get_enhanced_features(self, ue_idx, ablation_mode=False):
        """
        Extract enhanced features from history for a specific UE.
        
        Args:
            ue_idx: Index of the UE
            ablation_mode: If True, return zeros to simulate "No Memory/GRU" access
                          This is used for ablation studies to prove the value of history features.
        
        Returns:
            numpy array of enhanced features
        """
        # REVIEWER 1 & 3 REQUEST: ABLATION STUDY
        if ablation_mode:
            # Return zeros to simulate "No Memory/GRU" access
            return np.zeros(5, dtype=np.float32)
        
        features = []
        
        # Current blockage status
        if len(self.blockage_history) > 0:
            features.append(float(self.blockage_history[-1][ue_idx]))
        else:
            features.append(0.0)
        
        # Recent blockage average (last 5 timesteps)
        if len(self.blockage_history) >= 5:
            recent_blocks = [b[ue_idx] for b in list(self.blockage_history)[-5:]]
            features.append(np.mean(recent_blocks))
        else:
            features.append(0.0)
        
        # SNR trend (last 3 timesteps)
        if len(self.snr_history) >= 3:
            snr_list = list(self.snr_history)
            recent_snr = [s[ue_idx] for s in snr_list[-3:]]
            snr_trend = (recent_snr[-1] - recent_snr[0]) / 20.0
            features.append(np.clip(snr_trend, -1, 1))
        else:
            features.append(0.0)
        
        # Beam persistence (how long current beam has been used)
        if len(self.beam_history) >= 2:
            beam_list = list(self.beam_history)
            same_beam_count = 0
            current_beam = beam_list[-1][ue_idx]
            for b in reversed(beam_list[:-1]):
                if b[ue_idx] == current_beam:
                    same_beam_count += 1
                else:
                    break
            features.append(same_beam_count / self.window)
        else:
            features.append(0.0)
        
        # Normalized velocity
        if len(self.velocity_history) >= 2:
            vel_list = list(self.velocity_history)
            current_vel = vel_list[-1][ue_idx] if len(vel_list[-1]) > ue_idx else 0
            features.append(current_vel / VELOCITY_NORMALIZATION_FACTOR)
        else:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for DQN training.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done, priority=None):
        """Add experience to buffer"""
        if priority is None:
            priority = max(self.priorities, default=1.0) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        """Sample batch with prioritized sampling"""
        if len(self.buffer) < batch_size:
            return None
        
        priorities = np.array(self.priorities)
        
        if not np.all(np.isfinite(priorities)):
            priorities = np.ones_like(priorities)
        
        if np.sum(priorities) < 1e-10:
            priorities = np.ones_like(priorities)
        
        probs = priorities / priorities.sum()
        
        if not np.all(np.isfinite(probs)):
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        weights = (len(self.buffer) * probs[indices]) ** -0.5
        weights /= weights.max()
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = abs(td_error) + 1e-6
    
    def __len__(self):
        return len(self.buffer)

