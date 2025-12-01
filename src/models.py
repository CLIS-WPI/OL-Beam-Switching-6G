"""
Neural Network Models: PyTorch DQN implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import NUM_BEAMS


class MobilityAwareDQN(nn.Module):
    """
    Dueling DQN architecture with BatchNorm and Dropout for 6G beam switching.
    """
    def __init__(self, input_size, num_beams=NUM_BEAMS, hidden_size=512):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        
        # Dueling architecture
        self.value_stream = nn.Linear(hidden_size // 2, 1)
        self.advantage_stream = nn.Linear(hidden_size // 2, num_beams)
        
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = F.relu(self.fc1(x))
        if batch_size > 1:
            x = self.bn1(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        if batch_size > 1:
            x = self.bn2(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        if batch_size > 1:
            x = self.bn3(x)
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values

