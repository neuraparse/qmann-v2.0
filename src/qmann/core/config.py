"""
QMANN Configuration Management

Provides comprehensive configuration for quantum-classical hybrid systems
with 2025 hardware specifications and optimization settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import json
import os
from pathlib import Path


@dataclass
class QuantumConfig:
    """Configuration for quantum computing components."""

    # Hardware specifications (2025 NISQ era)
    max_qubits: int = 127  # IBM Quantum System Two capability
    gate_fidelity: float = 0.999  # Current best gate fidelities
    measurement_fidelity: float = 0.995
    coherence_time_t1: float = 100e-6  # 100 microseconds
    coherence_time_t2: float = 50e-6  # 50 microseconds

    # Backend configuration
    backend_name: str = "ibm_quantum"
    simulator_name: str = "qasm_simulator"
    use_hardware: bool = False
    shots: int = 8192
    optimization_level: int = 3

    # Circuit design
    max_circuit_depth: int = 100
    two_qubit_gate_limit: int = 50
    enable_pulse_optimization: bool = True

    # Error mitigation (2025 state-of-the-art)
    enable_error_mitigation: bool = True
    mitigation_methods: List[str] = field(
        default_factory=lambda: [
            "zero_noise_extrapolation",
            "probabilistic_error_cancellation",
            "measurement_error_mitigation",
        ]
    )
    zne_noise_factors: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])

    # Quantum memory specific
    memory_qubits: int = 16
    ancilla_qubits: int = 8
    enable_decoherence_protection: bool = True


@dataclass
class ClassicalConfig:
    """Configuration for classical neural network components."""

    # Model architecture
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"

    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10

    # Optimization
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    # Hardware acceleration
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    compile_model: bool = True  # PyTorch 2.0+ compilation


@dataclass
class HybridConfig:
    """Configuration for quantum-classical hybrid training."""

    # Training coordination
    quantum_classical_ratio: float = 0.3  # 30% quantum, 70% classical
    alternating_training: bool = True
    sync_frequency: int = 10  # Sync every 10 steps
    coordination_strategy: str = "alternating"  # alternating, parallel, sequential

    # Gradient handling
    quantum_lr_scale: float = 0.1  # Scale quantum learning rates
    parameter_shift_rule: str = "two_point"  # two_point, four_point
    finite_diff_step: float = 1e-7

    # Stability measures
    gradient_clipping: bool = True
    adaptive_learning_rates: bool = True
    numerical_stability_eps: float = 1e-8

    # Memory management
    quantum_memory_size: int = 64
    classical_memory_size: int = 1024
    memory_consolidation_freq: int = 100


@dataclass
class ApplicationConfig:
    """Configuration for specific application domains."""

    # Healthcare application
    healthcare_enabled: bool = False
    prediction_horizon_days: int = 14
    sensitivity_threshold: float = 0.85
    specificity_threshold: float = 0.90

    # Industrial maintenance
    industrial_enabled: bool = False
    sensor_sampling_rate: float = 1000.0  # Hz
    prediction_window_hours: int = 48
    downtime_reduction_target: float = 0.30

    # Autonomous systems
    autonomous_enabled: bool = False
    num_agents: int = 4
    coordination_memory_size: int = 128
    communication_range: float = 10.0


@dataclass
class QMANNConfig:
    """Main QMANN configuration class."""

    # Sub-configurations
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    application: ApplicationConfig = field(default_factory=ApplicationConfig)

    # Global settings
    random_seed: int = 42
    log_level: str = "INFO"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 100

    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Experimental features (2025)
    enable_quantum_advantage_tracking: bool = True
    enable_energy_monitoring: bool = True
    enable_interpretability_analysis: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._create_directories()

    def _validate_config(self):
        """Validate configuration parameters."""
        # Quantum validation
        if (
            self.quantum.memory_qubits + self.quantum.ancilla_qubits
            > self.quantum.max_qubits
        ):
            raise ValueError("Total qubits exceed maximum available qubits")

        if not 0 < self.quantum.gate_fidelity <= 1:
            raise ValueError("Gate fidelity must be between 0 and 1")

        # Classical validation
        if self.classical.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.classical.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # Hybrid validation
        if not 0 <= self.hybrid.quantum_classical_ratio <= 1:
            raise ValueError("Quantum-classical ratio must be between 0 and 1")

    def _create_directories(self):
        """Create necessary directories."""
        for dir_path in [
            self.data_dir,
            self.output_dir,
            self.checkpoint_dir,
            self.log_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def save(self, filepath: Union[str, Path]):
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "QMANNConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "quantum": self.quantum.__dict__,
            "classical": self.classical.__dict__,
            "hybrid": self.hybrid.__dict__,
            "application": self.application.__dict__,
            "random_seed": self.random_seed,
            "log_level": self.log_level,
            "save_checkpoints": self.save_checkpoints,
            "checkpoint_frequency": self.checkpoint_frequency,
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "enable_quantum_advantage_tracking": self.enable_quantum_advantage_tracking,
            "enable_energy_monitoring": self.enable_energy_monitoring,
            "enable_interpretability_analysis": self.enable_interpretability_analysis,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QMANNConfig":
        """Create configuration from dictionary."""
        quantum_config = QuantumConfig(**config_dict.get("quantum", {}))
        classical_config = ClassicalConfig(**config_dict.get("classical", {}))
        hybrid_config = HybridConfig(**config_dict.get("hybrid", {}))
        application_config = ApplicationConfig(**config_dict.get("application", {}))

        return cls(
            quantum=quantum_config,
            classical=classical_config,
            hybrid=hybrid_config,
            application=application_config,
            **{
                k: v
                for k, v in config_dict.items()
                if k not in ["quantum", "classical", "hybrid", "application"]
            },
        )

    def get_hardware_requirements(self) -> Dict[str, Any]:
        """Get hardware requirements summary."""
        return {
            "quantum_qubits_required": self.quantum.memory_qubits
            + self.quantum.ancilla_qubits,
            "quantum_gate_fidelity_required": self.quantum.gate_fidelity,
            "classical_gpu_recommended": self.classical.device in ["auto", "cuda"],
            "memory_gb_estimated": self._estimate_memory_requirements(),
            "compute_hours_estimated": self._estimate_compute_time(),
        }

    def _estimate_memory_requirements(self) -> float:
        """Estimate memory requirements in GB."""
        # Quantum state simulation memory
        quantum_memory = (
            2**self.quantum.memory_qubits * 16 / (1024**3)
        )  # Complex128 in GB

        # Classical model memory
        classical_memory = (
            self.classical.hidden_size * self.classical.num_layers * 4 / (1024**3)
        )

        return quantum_memory + classical_memory + 2.0  # 2GB buffer

    def _estimate_compute_time(self) -> float:
        """Estimate compute time in hours."""
        # Very rough estimation based on problem complexity
        base_time = self.classical.max_epochs * 0.1  # 0.1 hour per epoch base
        quantum_overhead = self.quantum.shots / 1000 * 0.01  # Shot overhead
        return base_time + quantum_overhead


def get_default_config() -> QMANNConfig:
    """Get default QMANN configuration optimized for 2025 hardware."""
    return QMANNConfig()


def get_config_for_application(app_type: str) -> QMANNConfig:
    """Get configuration optimized for specific applications."""
    config = get_default_config()

    if app_type == "healthcare":
        config.application.healthcare_enabled = True
        config.quantum.memory_qubits = 20  # Larger memory for patient data
        config.classical.hidden_size = 512
        config.hybrid.quantum_classical_ratio = 0.4

    elif app_type == "industrial":
        config.application.industrial_enabled = True
        config.quantum.shots = 4096  # Faster inference for real-time
        config.classical.batch_size = 64
        config.hybrid.sync_frequency = 5

    elif app_type == "autonomous":
        config.application.autonomous_enabled = True
        config.quantum.memory_qubits = 24  # Multi-agent coordination
        config.hybrid.memory_consolidation_freq = 50

    return config
