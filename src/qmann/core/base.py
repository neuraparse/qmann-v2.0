"""
QMANN Base Classes

Foundational classes for quantum-classical hybrid neural networks.
"""

import abc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from .config import QMANNConfig
from .exceptions import QMANNError, ConfigurationError


class QMANNBase(abc.ABC):
    """Abstract base class for all QMANN components."""

    def __init__(self, config: QMANNConfig, name: str = None):
        """
        Initialize QMANN base component.

        Args:
            config: QMANN configuration object
            name: Optional component name for logging
        """
        self.config = config
        self.name = name or self.__class__.__name__
        self.logger = self._setup_logger()

        # Performance tracking (2025 feature)
        self.metrics = {
            "energy_consumption": 0.0,
            "quantum_advantage_score": 0.0,
            "execution_time": 0.0,
            "memory_usage": 0.0,
        }

        # State tracking
        self._initialized = False
        self._training_mode = False

        self.logger.info(
            f"Initialized {self.name} with config: {type(config).__name__}"
        )

    def _setup_logger(self) -> logging.Logger:
        """Set up component-specific logger."""
        logger = logging.getLogger(f"qmann.{self.name}")
        logger.setLevel(getattr(logging, self.config.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the component. Must be implemented by subclasses."""
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass. Must be implemented by subclasses."""
        pass

    def train(self) -> None:
        """Set component to training mode."""
        self._training_mode = True
        self.logger.debug(f"{self.name} set to training mode")

    def eval(self) -> None:
        """Set component to evaluation mode."""
        self._training_mode = False
        self.logger.debug(f"{self.name} set to evaluation mode")

    def is_training(self) -> bool:
        """Check if component is in training mode."""
        return self._training_mode

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        for key in self.metrics:
            self.metrics[key] = 0.0
        self.logger.debug(f"{self.name} metrics reset")

    def save_state(self, filepath: str) -> None:
        """Save component state to file."""
        state = {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "training_mode": self._training_mode,
            "initialized": self._initialized,
        }
        torch.save(state, filepath)
        self.logger.info(f"{self.name} state saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load component state from file."""
        state = torch.load(filepath)
        self.metrics = state.get("metrics", self.metrics)
        self._training_mode = state.get("training_mode", False)
        self._initialized = state.get("initialized", False)
        self.logger.info(f"{self.name} state loaded from {filepath}")

    def validate_config(self) -> None:
        """Validate configuration for this component."""
        if not isinstance(self.config, QMANNConfig):
            raise ConfigurationError(
                f"Invalid config type: expected QMANNConfig, got {type(self.config)}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', initialized={self._initialized})"


class QuantumComponent(QMANNBase):
    """Base class for quantum computing components."""

    def __init__(self, config: QMANNConfig, name: str = None):
        super().__init__(config, name)
        self.quantum_config = config.quantum
        self.backend = None
        self.circuit_cache = {}

        # Quantum-specific metrics
        self.metrics.update(
            {
                "gate_count": 0,
                "circuit_depth": 0,
                "fidelity": 0.0,
                "decoherence_rate": 0.0,
            }
        )

    @abc.abstractmethod
    def build_circuit(self, *args, **kwargs):
        """Build quantum circuit. Must be implemented by subclasses."""
        pass

    def get_backend(self):
        """Get quantum backend for execution."""
        if self.backend is None:
            from ..utils import QuantumBackend

            self.backend = QuantumBackend.get_backend(
                self.quantum_config.backend_name,
                use_hardware=self.quantum_config.use_hardware,
            )
        return self.backend

    def execute_circuit(self, circuit, shots: int = None):
        """Execute quantum circuit on backend."""
        shots = shots or self.quantum_config.shots
        backend = self.get_backend()

        start_time = time.time()
        job = backend.run(circuit, shots=shots)
        result = job.result()
        execution_time = time.time() - start_time

        # Update metrics
        self.metrics["execution_time"] += execution_time
        self.metrics["gate_count"] += circuit.size()
        self.metrics["circuit_depth"] = max(
            self.metrics["circuit_depth"], circuit.depth()
        )

        return result


class ClassicalComponent(QMANNBase, nn.Module):
    """Base class for classical neural network components."""

    def __init__(self, config: QMANNConfig, name: str = None):
        QMANNBase.__init__(self, config, name)
        nn.Module.__init__(self)

        self.classical_config = config.classical
        self.device = self._get_device()

        # Classical-specific metrics
        self.metrics.update(
            {
                "parameters": 0,
                "flops": 0,
                "gpu_memory": 0.0,
            }
        )

    def _get_device(self) -> torch.device:
        """Determine the appropriate device for computation."""
        device_str = self.classical_config.device

        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_str)

        self.logger.info(f"{self.name} using device: {device}")
        return device

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to appropriate device."""
        return tensor.to(self.device)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize(self) -> None:
        """Initialize classical component."""
        self.to(self.device)
        self.metrics["parameters"] = self.count_parameters()
        self._initialized = True
        self.logger.info(
            f"{self.name} initialized with {self.metrics['parameters']} parameters"
        )


class HybridComponent(QMANNBase):
    """Base class for hybrid quantum-classical components."""

    def __init__(self, config: QMANNConfig, name: str = None):
        super().__init__(config, name)
        self.hybrid_config = config.hybrid

        # Hybrid-specific metrics
        self.metrics.update(
            {
                "quantum_ratio": 0.0,
                "classical_ratio": 0.0,
                "sync_operations": 0,
                "gradient_norm": 0.0,
            }
        )

        # Component tracking
        self.quantum_components = []
        self.classical_components = []

    def add_quantum_component(self, component: QuantumComponent) -> None:
        """Add quantum component to hybrid system."""
        self.quantum_components.append(component)
        self.logger.debug(f"Added quantum component: {component.name}")

    def add_classical_component(self, component: ClassicalComponent) -> None:
        """Add classical component to hybrid system."""
        self.classical_components.append(component)
        self.logger.debug(f"Added classical component: {component.name}")

    def sync_components(self) -> None:
        """Synchronize quantum and classical components."""
        self.metrics["sync_operations"] += 1
        self.logger.debug(
            f"Synchronized components (operation #{self.metrics['sync_operations']})"
        )

    @abc.abstractmethod
    def quantum_classical_interface(self, quantum_output, classical_input):
        """Interface between quantum and classical components."""
        pass
