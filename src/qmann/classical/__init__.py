"""
Classical Neural Network Components

This module provides classical neural network components that integrate
with quantum memory systems in the QMANN architecture.
"""

from .lstm import ClassicalLSTM
# from .attention import AttentionMechanism, MultiHeadAttention  # TODO: Implement
# from .memory import ClassicalMemory  # TODO: Implement
# from .layers import MemoryAugmentedLayer  # TODO: Implement

__all__ = [
    "ClassicalLSTM",
    # "AttentionMechanism",  # TODO: Implement
    # "MultiHeadAttention",  # TODO: Implement
    # "ClassicalMemory",  # TODO: Implement
    # "MemoryAugmentedLayer",  # TODO: Implement
]
