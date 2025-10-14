"""
QMANN Exception Classes

Custom exceptions for quantum-classical hybrid neural networks.
"""


class QMANNError(Exception):
    """Base exception class for QMANN framework."""

    def __init__(self, message: str, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class QuantumError(QMANNError):
    """Exceptions related to quantum computing operations."""

    def __init__(
        self,
        message: str,
        backend: str | None = None,
        circuit_depth: int | None = None,
    ) -> None:
        super().__init__(message, "QUANTUM_ERROR")
        self.backend = backend
        self.circuit_depth = circuit_depth


class CircuitError(QuantumError):
    """Exceptions related to quantum circuit construction or execution."""

    pass


class BackendError(QuantumError):
    """Exceptions related to quantum backend access or configuration."""

    pass


class CoherenceError(QuantumError):
    """Exceptions related to quantum decoherence and noise."""

    pass


class TrainingError(QMANNError):
    """Exceptions related to hybrid training processes."""

    def __init__(
        self,
        message: str,
        epoch: int | None = None,
        loss_value: float | None = None,
    ) -> None:
        super().__init__(message, "TRAINING_ERROR")
        self.epoch = epoch
        self.loss_value = loss_value


class ConvergenceError(TrainingError):
    """Exceptions related to training convergence issues."""

    pass


class GradientError(TrainingError):
    """Exceptions related to gradient computation in hybrid systems."""

    pass


class MemoryError(QMANNError):
    """Exceptions related to quantum memory operations."""

    def __init__(
        self,
        message: str,
        memory_size: int | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(message, "MEMORY_ERROR")
        self.memory_size = memory_size
        self.operation = operation


class ConfigurationError(QMANNError):
    """Exceptions related to configuration validation."""

    def __init__(self, message: str, parameter: str | None = None) -> None:
        super().__init__(message, "CONFIG_ERROR")
        self.parameter = parameter


class HardwareError(QMANNError):
    """Exceptions related to hardware requirements or limitations."""

    def __init__(
        self,
        message: str,
        required_qubits: int | None = None,
        available_qubits: int | None = None,
    ) -> None:
        super().__init__(message, "HARDWARE_ERROR")
        self.required_qubits = required_qubits
        self.available_qubits = available_qubits


class ApplicationError(QMANNError):
    """Exceptions related to specific application domains."""

    def __init__(self, message: str, application_type: str | None = None) -> None:
        super().__init__(message, "APPLICATION_ERROR")
        self.application_type = application_type


class VisualizationError(QMANNError):
    """Error in visualization operations."""

    def __init__(self, message: str, plot_type: str | None = None) -> None:
        super().__init__(message, "VISUALIZATION_ERROR")
        self.plot_type = plot_type


class BenchmarkError(QMANNError):
    """Error in benchmark operations."""

    def __init__(self, message: str, benchmark_type: str | None = None) -> None:
        super().__init__(message, "BENCHMARK_ERROR")
        self.benchmark_type = benchmark_type
