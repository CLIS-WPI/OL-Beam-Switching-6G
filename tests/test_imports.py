#!/usr/bin/env python3
"""
Test that all required dependencies can be imported
"""

import pytest


def test_import_torch():
    """Test PyTorch import"""
    import torch
    assert torch.__version__ is not None


def test_import_tensorflow():
    """Test TensorFlow import"""
    import tensorflow as tf
    assert tf.__version__ is not None


def test_import_numpy():
    """Test NumPy import"""
    import numpy as np
    assert np.__version__ is not None


def test_import_scipy():
    """Test SciPy import"""
    import scipy
    assert scipy.__version__ is not None


def test_import_sionna():
    """Test Sionna import"""
    try:
        import sionna
        assert sionna.__version__ is not None
    except ImportError:
        pytest.skip("Sionna not available in test environment")


def test_import_matplotlib():
    """Test Matplotlib import"""
    import matplotlib
    assert matplotlib.__version__ is not None


def test_main_module_import():
    """Test that main-final.py can be imported (syntax check)"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Just check syntax, don't actually import to avoid execution
    import py_compile
    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main-final.py')
    py_compile.compile(main_path, doraise=True)

