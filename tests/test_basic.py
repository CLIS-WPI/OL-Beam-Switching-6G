#!/usr/bin/env python3
"""
Basic unit tests for core functionality
"""

import pytest
import numpy as np
import torch


def test_numpy_basic():
    """Test basic NumPy operations"""
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.sum() == 15
    assert arr.mean() == 3.0


def test_torch_basic():
    """Test basic PyTorch operations"""
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert tensor.sum().item() == 15.0
    assert tensor.mean().item() == 3.0


def test_torch_device():
    """Test PyTorch device availability"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device is not None
    # Test that we can create a tensor on the device
    tensor = torch.tensor([1.0, 2.0], device=device)
    assert tensor.device == device


def test_parameters():
    """Test that basic parameters are valid"""
    NUM_BEAMS = 64
    NUM_UES = 100
    ROAD_LENGTH = 500.0
    
    assert NUM_BEAMS > 0
    assert NUM_UES > 0
    assert ROAD_LENGTH > 0
    assert NUM_BEAMS <= 128  # Reasonable upper limit


@pytest.mark.parametrize("snr_threshold", [5.0, 6.0, 10.0, 15.0])
def test_snr_threshold_valid(snr_threshold):
    """Test that SNR threshold values are valid"""
    assert snr_threshold > 0
    assert snr_threshold < 100  # Reasonable upper limit

