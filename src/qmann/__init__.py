"""
QMANN: Quantum Memory-Augmented Neural Networks

A cutting-edge implementation leveraging 2025 quantum computing technologies
to create hybrid quantum-classical neural networks with quantum memory.

Copyright 2025 QMANN Research Team
Licensed under the Apache License, Version 2.0
"""

__version__ = "2.0.0"
__author__ = "QMANN Research Team"
__email__ = "qmann@research.org"

# Core imports
from .core import QMANNConfig, QMANNBase
from .quantum import QMatrix, QuantumMemory, AmplitudeAmplification
from .classical import ClassicalLSTM  # AttentionMechanism TODO: Implement
from .hybrid import QuantumLSTM, HybridTrainer
from .utils import QuantumBackend, ErrorMitigation, Visualization, Benchmarks

# Application imports
from .applications import (
    HealthcarePredictor,
    IndustrialMaintenance,
    AutonomousCoordination,
)

__all__ = [
    # Core
    "QMANNConfig",
    "QMANNBase",
    # Quantum components
    "QMatrix",
    "QuantumMemory",
    "AmplitudeAmplification",
    # Classical components
    "ClassicalLSTM",
    # "AttentionMechanism",  # TODO: Implement
    # Hybrid components
    "QuantumLSTM",
    "HybridTrainer",
    # Utilities
    "QuantumBackend",
    "ErrorMitigation",
    "Visualization",
    "Benchmarks",
    # Applications
    "HealthcarePredictor",
    "IndustrialMaintenance",
    "AutonomousCoordination",
]

# Version compatibility check
import sys

if sys.version_info < (3, 10):
    raise RuntimeError("QMANN requires Python 3.10 or higher")

# Quantum framework compatibility check
try:
    import qiskit

    if tuple(map(int, qiskit.__version__.split(".")[:2])) < (2, 1):
        raise ImportError("QMANN requires Qiskit 2.1 or higher")
except ImportError:
    raise ImportError(
        "Qiskit is required for QMANN. Install with: pip install qiskit>=2.1.0"
    )

# Optional GPU support check
try:
    import torch

    if torch.cuda.is_available():
        print(f"QMANN: CUDA GPU support detected - {torch.cuda.get_device_name()}")
    else:
        print(
            "QMANN: Running in CPU mode. For GPU acceleration, ensure CUDA is properly installed."
        )
except ImportError:
    print("QMANN: PyTorch not found. Some features may be limited.")

# IBM Quantum access check
try:
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        service = QiskitRuntimeService()
        backends = service.backends()
        print(
            f"QMANN: IBM Quantum access confirmed - {len(backends)} backends available"
        )
    except Exception:
        print(
            "QMANN: IBM Quantum access not configured. Use QiskitRuntimeService.save_account() to set up."
        )
except ImportError:
    print(
        "QMANN: IBM Quantum Runtime not available. Install with: pip install qiskit-ibm-runtime"
    )


def get_version():
    """Get the current version of QMANN."""
    return __version__


def get_config():
    """Get default QMANN configuration."""
    return QMANNConfig()


def list_backends():
    """List available quantum backends."""
    try:
        from .utils import QuantumBackend

        return QuantumBackend.list_available()
    except Exception as e:
        print(f"Error listing backends: {e}")
        return []


def run_diagnostics():
    """Run system diagnostics for QMANN setup."""
    print("QMANN System Diagnostics")
    print("=" * 40)

    # Python version
    print(f"Python version: {sys.version}")

    # Qiskit version
    try:
        import qiskit

        print(f"Qiskit version: {qiskit.__version__}")
    except ImportError:
        print("Qiskit: NOT INSTALLED")

    # PyTorch version and GPU
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")

    # IBM Quantum access
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        backends = service.backends()
        print(f"IBM Quantum backends: {len(backends)}")
    except Exception:
        print("IBM Quantum: NOT CONFIGURED")

    print("=" * 40)
