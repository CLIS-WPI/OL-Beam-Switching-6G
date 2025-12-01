"""
Environment: Channel, Mobility, and Beamforming Physics
"""

import numpy as np
import tensorflow as tf
from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading
from src.config import (
    NUM_UES, NUM_ANTENNAS, NUM_BEAMS, ROAD_LENGTH, BS_POSITION,
    CARRIER_FREQ, TX_POWER_DBM, NOISE_POWER_DBM, TIMESTEP_DURATION,
    BLOCKAGE_ATTENUATION_DB
)

# Configure GPU for TensorFlow (Sionna) to avoid conflicts with PyTorch
# Only configure if not already configured (to avoid conflicts)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Already configured, skip
                pass
except Exception:
    # GPU configuration failed, continue without GPU
    pass


# Global codebook (generated once)
CODEBOOK = None


def generate_codebook(num_antennas=NUM_ANTENNAS, num_beams=NUM_BEAMS):
    """Generate beamforming codebook"""
    angles = np.linspace(-np.pi/3, np.pi/3, num_beams)
    codebook = np.zeros((num_antennas, num_beams), dtype=complex)
    antenna_indices = np.arange(num_antennas)
    for i, theta in enumerate(angles):
        steering_vector = np.exp(1j * np.pi * antenna_indices * np.sin(theta))
        codebook[:, i] = steering_vector / np.sqrt(num_antennas)
    return codebook


def get_codebook():
    """Get or generate codebook (singleton pattern)"""
    global CODEBOOK
    if CODEBOOK is None:
        CODEBOOK = generate_codebook()
    return CODEBOOK


def compute_path_loss_3gpp(distances, freq_hz=CARRIER_FREQ):
    """
    Compute 3GPP path loss model.
    Returns: (path_loss_linear, path_loss_db)
    """
    min_distance = 1.0
    distances = np.maximum(distances, min_distance)
    
    d_bp = 4 * 25 * 1.5 * freq_hz / 3e8
    
    pl_db = np.where(
        distances < d_bp,
        32.4 + 21 * np.log10(distances) + 20 * np.log10(freq_hz / 1e9),
        32.4 + 40 * np.log10(distances) + 20 * np.log10(freq_hz / 1e9) - 9.5
    )
    
    path_loss_linear = 10 ** (-pl_db / 10)
    return path_loss_linear, pl_db


def get_ue_mobility_category(ue_idx):
    """Get mobility category for UE (0=slow, 1=medium, 2=fast)"""
    if ue_idx % 3 == 0:
        return 0
    elif ue_idx % 3 == 1:
        return 1
    else:
        return 2


def update_positions(t, velocity_multiplier=1.0):
    """
    Update UE positions based on mobility model.
    Returns: (positions, velocities)
    """
    positions = np.zeros(NUM_UES)
    velocities = np.zeros(NUM_UES)
    
    for i in range(NUM_UES):
        mobility_cat = get_ue_mobility_category(i)
        
        if mobility_cat == 0:  # Slow
            freq = 0.001 + (i % 10) * 0.0005
            movement_range = 50
            base_speed = 1.4
        elif mobility_cat == 1:  # Medium
            freq = 0.005 + (i % 10) * 0.001
            movement_range = 150
            base_speed = 8.3
        else:  # Fast
            freq = 0.01 + (i % 10) * 0.002
            movement_range = 250
            base_speed = 20.0
        
        positions[i] = BS_POSITION + movement_range * np.sin(
            freq * t * TIMESTEP_DURATION * velocity_multiplier + i * np.pi / NUM_UES
        )
        positions[i] = np.clip(positions[i], 10, ROAD_LENGTH - 10)
        
        velocities[i] = base_speed * np.cos(
            freq * t * TIMESTEP_DURATION * velocity_multiplier + i * np.pi / NUM_UES
        )
    
    return positions, velocities


class SionnaChannelGenerator:
    """
    NVIDIA Sionna-based Channel Generator using Rayleigh Block Fading.
    
    This class replaces the manual NumPy-based Rician fading implementation
    to match the methodology claimed in the accepted paper.
    """
    def __init__(self, num_ues, num_antennas, carrier_freq):
        """
        Initializes the Sionna Rayleigh Block Fading Model.
        
        Matches the paper description: Rayleigh Block Fading.
        
        Args:
            num_ues: Number of user equipments (receivers)
            num_antennas: Number of BS antennas (transmit antennas)
            carrier_freq: Carrier frequency (for path loss calculation)
        """
        self.num_ues = num_ues
        self.num_antennas = num_antennas
        self.carrier_freq = carrier_freq
        
        # Configure the Rayleigh Block Fading model from Sionna
        # This creates an i.i.d. Rayleigh fading channel
        # num_rx: number of receivers (UEs)
        # num_rx_ant: number of antennas per receiver (each UE has 1 antenna)
        # num_tx: number of transmitters (1 BS)
        # num_tx_ant: number of antennas per transmitter (BS antennas)
        self.rayleigh_model = RayleighBlockFading(
            num_rx=num_ues,        # All UEs as separate receivers
            num_rx_ant=1,           # Single antenna per UE
            num_tx=1,               # Single BS
            num_tx_ant=num_antennas # BS antennas
        )
        
        print("✅ NVIDIA Sionna: Rayleigh Block Fading Model Initialized")

    def generate_channel(self, positions_np):
        """
        Generates channel coefficients h using Sionna.
        
        Args:
            positions_np: Numpy array of user positions (used for pathloss).
            
        Returns:
            h_combined: Numpy array (NUM_UES, NUM_ANTENNAS) - Complex channel
        """
        # Generate channel coefficients
        # batch_size = 1 (single batch for all UEs)
        # num_time_steps = 1 (single time step for block fading)
        # Output shape: [1, NUM_UES, 1, 1, NUM_ANTENNAS, 1, 1]
        #   [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        h_tensor, _ = self.rayleigh_model(batch_size=1, num_time_steps=1)
        
        # Convert TensorFlow tensor to NumPy array
        h_np = h_tensor.numpy()
        
        # Extract and reshape from [1, NUM_UES, 1, 1, NUM_ANTENNAS, 1, 1] to [NUM_UES, NUM_ANTENNAS]
        # Extract: batch=0, all UEs, rx_ant=0, tx=0, all tx_ants, path=0, time=0
        h_fading = h_np[0, :, 0, 0, :, 0, 0]  # Shape: [NUM_UES, NUM_ANTENNAS]
        
        # Handle edge case if squeeze removes too many dims
        if h_fading.ndim == 1:
            h_fading = h_fading.reshape(self.num_ues, self.num_antennas)

        # Calculate Path Loss (Keep existing manual logic - 3GPP model)
        distances = np.abs(positions_np - BS_POSITION)
        path_loss_linear, _ = compute_path_loss_3gpp(distances, self.carrier_freq)
        
        # Combine Path Loss and Fading
        # h_effective = sqrt(PL) * h_fading
        h_combined = h_fading * np.sqrt(path_loss_linear[:, np.newaxis])
        
        return h_combined.astype(np.complex128)


# ================= EXPORT INSTANCE =================
# Initialize the generator (singleton pattern)
_channel_gen = None

def _get_channel_generator():
    """Get or create the Sionna channel generator instance"""
    global _channel_gen
    if _channel_gen is None:
        _channel_gen = SionnaChannelGenerator(NUM_UES, NUM_ANTENNAS, CARRIER_FREQ)
    return _channel_gen


def generate_fast_channel(positions):
    """
    Wrapper function to match the signature expected by main.py.
    
    This function maintains backward compatibility while using Sionna internally.
    
    Args:
        positions: Array of UE positions (NUM_UES,)
    
    Returns:
        h_channel: Channel matrix (NUM_UES x NUM_ANTENNAS complex array)
                  Shape is maintained for compatibility with existing DRL agent.
    """
    channel_gen = _get_channel_generator()
    return channel_gen.generate_channel(positions)


def compute_snr_from_channel(h_channel, beam_idx, ue_idx, extra_attenuation_db=0.0):
    """
    Compute SNR for a specific UE and beam.
    
    Args:
        h_channel: Channel matrix (NUM_UES x NUM_ANTENNAS)
        beam_idx: Selected beam index
        ue_idx: UE index
        extra_attenuation_db: Additional attenuation (e.g., from blockage)
    
    Returns:
        SNR in dB
    """
    codebook = get_codebook()
    h_ue = h_channel[ue_idx, :].astype(complex)
    beam = codebook[:, beam_idx].astype(complex)
    
    effective_signal = np.abs(np.dot(np.conj(h_ue), beam))**2
    
    if not np.isfinite(effective_signal) or effective_signal < 1e-20:
        effective_signal = 1e-20
    
    noise_power_linear = 10**((NOISE_POWER_DBM - TX_POWER_DBM) / 10)
    snr_linear = effective_signal / (noise_power_linear + 1e-15)
    snr_db = 10 * np.log10(snr_linear + 1e-15) - extra_attenuation_db
    snr_db = np.clip(snr_db, -30, 60)
    
    if not np.isfinite(snr_db):
        snr_db = -30.0
    
    return snr_db


def compute_relative_angles(positions):
    """Compute relative angles from BS to UEs"""
    y_distance = 20.0
    x_distance = positions - BS_POSITION
    return np.arctan2(x_distance, y_distance)


def dynamic_channel_conditions(t):
    """
    Simulate dynamic channel conditions over time.
    Returns: (k_factor_offset, additional_blockage_prob)
    """
    if t % 1000 < 200:
        return 3.0, 0.05
    elif t % 1000 > 800:
        return -2.0, -0.05
    else:
        return 0.0, 0.0

