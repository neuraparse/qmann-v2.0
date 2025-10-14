#!/usr/bin/env python3
"""
Simple test script to verify QMANN imports work correctly.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_basic_imports():
    """Test basic QMANN imports."""
    # Test core config import
    from qmann.core.config import QMANNConfig

    # Create config instance
    config = QMANNConfig()
    assert config is not None

    # Test core base import
    from qmann.core.base import QMANNBase

    assert QMANNBase is not None

    # Test exceptions
    from qmann.core.exceptions import QMANNError

    assert QMANNError is not None


def test_optional_imports():
    """Test optional component imports."""
    # Test quantum components (requires qiskit)
    try:
        from qmann.quantum.qmatrix import QMatrix

        assert QMatrix is not None
    except ImportError:
        pytest.skip("Quantum components not available")

    # Test classical components (requires torch)
    try:
        from qmann.classical.lstm import ClassicalLSTM

        assert ClassicalLSTM is not None
    except ImportError:
        pytest.skip("Classical components not available")

    # Test utilities
    try:
        from qmann.utils.backend import QuantumBackend

        assert QuantumBackend is not None
    except ImportError:
        pytest.skip("Utilities not available")


def test_config_functionality():
    """Test configuration functionality."""
    from qmann.core.config import QMANNConfig

    # Test default config
    config = QMANNConfig()
    assert config is not None
    assert hasattr(config, "quantum")
    assert hasattr(config, "classical")
    assert hasattr(config, "hybrid")

    # Test config serialization
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)

    # Test config from dict
    new_config = QMANNConfig.from_dict(config_dict)
    assert new_config is not None
