"""
Quantum Circuits for QMANN

Specialized quantum circuits for memory operations, encoding/decoding,
and amplitude amplification using Qiskit 2.1+ features.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import (
    RealAmplitudes,
    EfficientSU2,
    TwoLocal,
    GroverOperator,
    PhaseOracle,
)
from qiskit.quantum_info import Statevector, Operator
from qiskit_algorithms import AmplificationProblem, Grover

from ..core.base import QuantumComponent
from ..core.exceptions import QuantumError, CircuitError


class AmplitudeAmplification(QuantumComponent):
    """
    Amplitude amplification implementation for quantum search.

    Provides O(√N) speedup for content-addressable memory recall
    using Grover-inspired amplitude amplification.
    """

    def __init__(self, config, num_qubits: int, name: str = "AmplitudeAmplification"):
        super().__init__(config, name)
        self.num_qubits = num_qubits
        self.max_iterations = int(np.pi / 4 * np.sqrt(2**num_qubits))

        # Circuit components
        self.oracle = None
        self.diffuser = None
        self.amplification_circuit = None

    def initialize(self) -> None:
        """Initialize amplitude amplification components."""
        try:
            self._build_diffuser()
            self._initialized = True
            self.logger.info(
                f"AmplitudeAmplification initialized for {self.num_qubits} qubits"
            )
        except Exception as e:
            raise QuantumError(
                f"AmplitudeAmplification initialization failed: {str(e)}"
            )

    def build_circuit(
        self, target_states: List[str], num_iterations: Optional[int] = None
    ) -> QuantumCircuit:
        """
        Build amplitude amplification circuit for specific target states.

        Args:
            target_states: List of binary strings representing target states
            num_iterations: Number of amplification iterations (auto-calculated if None)

        Returns:
            Quantum circuit implementing amplitude amplification
        """
        if not self._initialized:
            self.initialize()

        # Calculate optimal number of iterations
        if num_iterations is None:
            num_targets = len(target_states)
            if num_targets > 0:
                success_probability = num_targets / (2**self.num_qubits)
                num_iterations = max(
                    1, int(np.pi / (4 * np.arcsin(np.sqrt(success_probability))))
                )
            else:
                num_iterations = 1

        num_iterations = min(num_iterations, self.max_iterations)

        try:
            # Build oracle for target states
            oracle = self._build_oracle(target_states)

            # Create amplification circuit
            circuit = QuantumCircuit(self.num_qubits)

            # Initialize superposition
            circuit.h(range(self.num_qubits))

            # Apply Grover iterations
            for _ in range(num_iterations):
                # Apply oracle
                circuit.compose(oracle, inplace=True)

                # Apply diffuser
                circuit.compose(self.diffuser, inplace=True)

            self.amplification_circuit = circuit

            # Update metrics
            self.metrics["gate_count"] = circuit.size()
            self.metrics["circuit_depth"] = circuit.depth()

            self.logger.debug(
                f"Built amplification circuit: {num_iterations} iterations, "
                f"{len(target_states)} targets, depth {circuit.depth()}"
            )

            return circuit

        except Exception as e:
            raise CircuitError(f"Failed to build amplification circuit: {str(e)}")

    def _build_oracle(self, target_states: List[str]) -> QuantumCircuit:
        """Build oracle circuit that marks target states."""
        oracle = QuantumCircuit(self.num_qubits)

        for target_state in target_states:
            if len(target_state) != self.num_qubits:
                raise CircuitError(
                    f"Target state length ({len(target_state)}) != num_qubits ({self.num_qubits})"
                )

            # Apply X gates to qubits that should be 0 in target state
            for i, bit in enumerate(target_state):
                if bit == "0":
                    oracle.x(i)

            # Apply multi-controlled Z gate
            if self.num_qubits == 1:
                oracle.z(0)
            elif self.num_qubits == 2:
                oracle.cz(0, 1)
            else:
                # Use multi-controlled Z gate for more qubits
                oracle.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
                oracle.z(self.num_qubits - 1)
                oracle.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)

            # Undo X gates
            for i, bit in enumerate(target_state):
                if bit == "0":
                    oracle.x(i)

        return oracle

    def _build_diffuser(self) -> None:
        """Build diffuser circuit for amplitude amplification."""
        self.diffuser = QuantumCircuit(self.num_qubits)

        # Apply Hadamard gates
        self.diffuser.h(range(self.num_qubits))

        # Apply X gates
        self.diffuser.x(range(self.num_qubits))

        # Apply multi-controlled Z gate
        if self.num_qubits == 1:
            self.diffuser.z(0)
        elif self.num_qubits == 2:
            self.diffuser.cz(0, 1)
        else:
            self.diffuser.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
            self.diffuser.z(self.num_qubits - 1)
            self.diffuser.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)

        # Undo X gates
        self.diffuser.x(range(self.num_qubits))

        # Undo Hadamard gates
        self.diffuser.h(range(self.num_qubits))

    def estimate_success_probability(
        self, target_states: List[str], num_iterations: int
    ) -> float:
        """Estimate success probability for given target states and iterations."""
        num_targets = len(target_states)
        total_states = 2**self.num_qubits

        if num_targets == 0:
            return 0.0

        # Theoretical success probability for Grover's algorithm
        theta = np.arcsin(np.sqrt(num_targets / total_states))
        success_prob = np.sin((2 * num_iterations + 1) * theta) ** 2

        return min(1.0, success_prob)

    def forward(
        self, target_states: List[str], num_iterations: Optional[int] = None
    ) -> QuantumCircuit:
        """Forward pass for neural network integration."""
        return self.build_circuit(target_states, num_iterations)


class QuantumEncoder(QuantumComponent):
    """
    Quantum encoder for converting classical data to quantum states.

    Uses variational quantum circuits optimized for NISQ devices.
    """

    def __init__(
        self,
        config,
        input_dim: int,
        num_qubits: int,
        encoding_layers: int = 2,
        name: str = "QuantumEncoder",
    ):
        super().__init__(config, name)
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.encoding_layers = encoding_layers

        # Validate dimensions
        if input_dim > 2**num_qubits:
            raise CircuitError(
                f"Input dimension ({input_dim}) exceeds quantum capacity (2^{num_qubits})"
            )

        # Encoding circuit
        self.encoding_circuit = None
        self.parameters = None

    def initialize(self) -> None:
        """Initialize quantum encoder."""
        try:
            self._build_encoding_circuit()
            self._initialize_parameters()
            self._initialized = True
            self.logger.info(
                f"QuantumEncoder initialized: {self.input_dim}→{self.num_qubits} qubits"
            )
        except Exception as e:
            raise QuantumError(f"QuantumEncoder initialization failed: {str(e)}")

    def _build_encoding_circuit(self) -> None:
        """Build variational encoding circuit."""
        # Use hardware-efficient ansatz
        self.encoding_circuit = EfficientSU2(
            num_qubits=self.num_qubits,
            reps=self.encoding_layers,
            entanglement="linear",
            insert_barriers=True,
        )

        # Add classical register for measurements
        classical_reg = ClassicalRegister(self.num_qubits)
        self.encoding_circuit.add_register(classical_reg)

    def _initialize_parameters(self) -> None:
        """Initialize encoding parameters."""
        num_params = self.encoding_circuit.num_parameters
        # Small random initialization to avoid barren plateaus
        self.parameters = np.random.normal(0, 0.1, num_params)
        self.logger.debug(f"Initialized {num_params} encoding parameters")

    def build_circuit(self, classical_data: np.ndarray) -> QuantumCircuit:
        """
        Build encoding circuit for specific classical data.

        Args:
            classical_data: Classical data vector to encode

        Returns:
            Quantum circuit encoding the classical data
        """
        if not self._initialized:
            self.initialize()

        if len(classical_data) > self.input_dim:
            raise CircuitError(
                f"Data dimension ({len(classical_data)}) exceeds encoder capacity ({self.input_dim})"
            )

        # Normalize and pad data
        normalized_data = classical_data / (np.linalg.norm(classical_data) + 1e-8)
        padded_data = np.zeros(2**self.num_qubits)
        padded_data[: len(normalized_data)] = normalized_data

        # Create encoding circuit
        circuit = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Initialize with amplitude encoding
        try:
            circuit.initialize(padded_data, range(self.num_qubits))
        except Exception:
            # Fallback: use basis encoding
            self._apply_basis_encoding(circuit, classical_data)

        # Apply variational encoding
        param_dict = dict(zip(self.encoding_circuit.parameters, self.parameters))
        bound_encoding = self.encoding_circuit.assign_parameters(param_dict)
        circuit.compose(bound_encoding, inplace=True)

        return circuit

    def _apply_basis_encoding(self, circuit: QuantumCircuit, data: np.ndarray) -> None:
        """Apply basis encoding as fallback method."""
        # Simple basis encoding: encode data as rotation angles
        for i, value in enumerate(data[: self.num_qubits]):
            angle = value * np.pi  # Scale to [0, π]
            circuit.ry(angle, i)

    def update_parameters(self, new_parameters: np.ndarray) -> None:
        """Update encoding parameters."""
        if len(new_parameters) != len(self.parameters):
            raise CircuitError(
                f"Parameter count mismatch: expected {len(self.parameters)}, got {len(new_parameters)}"
            )

        self.parameters = new_parameters.copy()
        self.logger.debug("Updated encoding parameters")

    def forward(self, classical_data: np.ndarray) -> QuantumCircuit:
        """Forward pass for neural network integration."""
        return self.build_circuit(classical_data)


class QuantumDecoder(QuantumComponent):
    """
    Quantum decoder for converting quantum states back to classical data.

    Implements measurement strategies optimized for information extraction.
    """

    def __init__(
        self,
        config,
        num_qubits: int,
        output_dim: int,
        measurement_strategy: str = "computational",
        name: str = "QuantumDecoder",
    ):
        super().__init__(config, name)
        self.num_qubits = num_qubits
        self.output_dim = output_dim
        self.measurement_strategy = measurement_strategy

        # Supported measurement strategies
        self.supported_strategies = ["computational", "pauli", "tomography"]
        if measurement_strategy not in self.supported_strategies:
            raise CircuitError(
                f"Unsupported measurement strategy: {measurement_strategy}"
            )

    def initialize(self) -> None:
        """Initialize quantum decoder."""
        self._initialized = True
        self.logger.info(
            f"QuantumDecoder initialized: {self.num_qubits} qubits→{self.output_dim}"
        )

    def build_circuit(self, quantum_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Build decoding circuit with measurements.

        Args:
            quantum_circuit: Input quantum circuit to decode

        Returns:
            Circuit with appropriate measurements added
        """
        if not self._initialized:
            self.initialize()

        # Create decoding circuit
        decode_circuit = quantum_circuit.copy()

        # Add measurements based on strategy
        if self.measurement_strategy == "computational":
            decode_circuit.measure_all()
        elif self.measurement_strategy == "pauli":
            self._add_pauli_measurements(decode_circuit)
        elif self.measurement_strategy == "tomography":
            self._add_tomography_measurements(decode_circuit)

        return decode_circuit

    def _add_pauli_measurements(self, circuit: QuantumCircuit) -> None:
        """Add Pauli measurements for enhanced information extraction."""
        # Measure in X, Y, Z bases for each qubit
        for qubit in range(min(self.num_qubits, 3)):  # Limit for demonstration
            # X measurement
            circuit.h(qubit)
            circuit.measure(qubit, qubit)

    def _add_tomography_measurements(self, circuit: QuantumCircuit) -> None:
        """Add measurements for quantum state tomography."""
        # Simplified tomography - in practice would require multiple circuits
        circuit.measure_all()

    def decode_measurements(self, measurement_results: dict) -> np.ndarray:
        """
        Decode measurement results to classical data.

        Args:
            measurement_results: Dictionary of measurement outcomes

        Returns:
            Classical data vector
        """
        if self.measurement_strategy == "computational":
            return self._decode_computational(measurement_results)
        elif self.measurement_strategy == "pauli":
            return self._decode_pauli(measurement_results)
        elif self.measurement_strategy == "tomography":
            return self._decode_tomography(measurement_results)

    def _decode_computational(self, results: dict) -> np.ndarray:
        """Decode computational basis measurements."""
        # Extract bit string probabilities
        total_shots = sum(results.values())
        probabilities = np.zeros(2**self.num_qubits)

        for bitstring, count in results.items():
            index = int(bitstring, 2)
            probabilities[index] = count / total_shots

        # Return first output_dim elements
        return probabilities[: self.output_dim]

    def _decode_pauli(self, results: dict) -> np.ndarray:
        """Decode Pauli measurements."""
        # Simplified Pauli decoding
        return self._decode_computational(results)

    def _decode_tomography(self, results: dict) -> np.ndarray:
        """Decode tomography measurements."""
        # Simplified tomography decoding
        return self._decode_computational(results)

    def forward(self, quantum_circuit: QuantumCircuit) -> QuantumCircuit:
        """Forward pass for neural network integration."""
        return self.build_circuit(quantum_circuit)
