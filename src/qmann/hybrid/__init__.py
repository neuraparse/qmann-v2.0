"""
Hybrid Quantum-Classical Components

This module implements the hybrid architecture that seamlessly
integrates quantum memory operations with classical neural networks.
"""

from .quantum_lstm import QuantumLSTM
from .trainer import HybridTrainer
from .training_protocols import (
    QuantumParameterShift,
    HybridOptimizer,
    NISQAwareTraining,
    AdvancedTrainingProtocols
)

__all__ = [
    "QuantumLSTM",
    "HybridTrainer",
    "QuantumParameterShift",
    "HybridOptimizer",
    "NISQAwareTraining",
    "AdvancedTrainingProtocols",
]
