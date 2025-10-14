"""
Quantum Computing Components for QMANN

This module implements quantum memory, circuits, and algorithms
using the latest Qiskit 2.1+ features and 2025 NISQ optimizations.
"""

from .qmatrix import QMatrix
from .memory import QuantumMemory
from .circuits import AmplitudeAmplification, QuantumEncoder, QuantumDecoder
# from .algorithms import GroverSearch, QuantumAttention  # TODO: Implement
# from .error_mitigation import ErrorMitigation, ZeroNoiseExtrapolation  # TODO: Implement
from .advanced_techniques_2025 import (
    QuantumTechnique2025,
    QuantumAdvantageMetrics,
    MultiHeadQuantumAttention,
    AdaptiveVariationalAnsatz,
    QuantumMemoryConsolidation,
    QuantumLSTM2025,
    QAOAWarmStart2025,
    GroverDynamicsOptimization2025
)

# Import 2025 quantum transformer components
from .quantum_transformer_2025 import (
    QuantumTransformerConfig,
    QuantumAttentionHead2025,
    QuantumFeedForward2025,
    QuantumTransformerLayer2025
)

# Import 2025 error mitigation techniques
from .error_mitigation_2025 import (
    ErrorMitigationTechnique2025,
    ErrorMitigationConfig2025,
    CircuitNoiseResilientVirtualDistillation,
    LearningBasedErrorMitigation2025,
    AdaptiveErrorCorrection2025
)

__all__ = [
    "QMatrix",
    "QuantumMemory",
    "AmplitudeAmplification",
    "QuantumEncoder",
    "QuantumDecoder",
    # "GroverSearch",  # TODO: Implement
    # "QuantumAttention",  # TODO: Implement
    # "ErrorMitigation",  # TODO: Implement
    # "ZeroNoiseExtrapolation",  # TODO: Implement
    "QuantumTechnique2025",
    "QuantumAdvantageMetrics",
    "MultiHeadQuantumAttention",
    "AdaptiveVariationalAnsatz",
    "QuantumMemoryConsolidation",
    "QuantumLSTM2025",
    "QAOAWarmStart2025",
    "GroverDynamicsOptimization2025",

    # 2025 Quantum Transformer
    "QuantumTransformerConfig",
    "QuantumAttentionHead2025",
    "QuantumFeedForward2025",
    "QuantumTransformerLayer2025",

    # 2025 Error Mitigation
    "ErrorMitigationTechnique2025",
    "ErrorMitigationConfig2025",
    "CircuitNoiseResilientVirtualDistillation",
    "LearningBasedErrorMitigation2025",
    "AdaptiveErrorCorrection2025",
]
