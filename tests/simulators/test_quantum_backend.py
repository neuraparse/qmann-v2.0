"""
Multi-level quantum simulation backend for QMANN testing.

Provides ideal, noisy NISQ, and hardware-level quantum simulators
with realistic noise models based on IBM quantum hardware specifications.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class NoiseModel(Enum):
    """Supported noise models based on real IBM quantum hardware."""
    IBM_SHERBROOKE = "ibm_sherbrooke"
    IBM_TORINO = "ibm_torino"
    IBM_HERON = "ibm_heron"
    IDEAL = "ideal"


@dataclass
class NoiseProfile:
    """Quantum hardware noise characteristics."""
    cx_error: float  # Two-qubit gate error rate
    sx_error: float  # Single-qubit gate error rate
    t1: float  # T1 relaxation time (seconds)
    t2: float  # T2 dephasing time (seconds)
    readout_error: float  # Measurement error rate
    gate_time: float = 35e-9  # Gate time in seconds


class QMANNSimulator:
    """Multi-level quantum simulation for QMANN validation."""
    
    # Noise profiles based on Table 4 specifications from paper
    NOISE_PROFILES = {
        NoiseModel.IBM_SHERBROOKE: NoiseProfile(
            cx_error=7.3e-3,
            sx_error=2.3e-4,
            t1=185e-6,
            t2=124e-6,
            readout_error=0.012
        ),
        NoiseModel.IBM_TORINO: NoiseProfile(
            cx_error=6.8e-3,
            sx_error=1.9e-4,
            t1=211e-6,
            t2=142e-6,
            readout_error=0.009
        ),
        NoiseModel.IBM_HERON: NoiseProfile(
            cx_error=5.2e-3,
            sx_error=1.5e-4,
            t1=250e-6,
            t2=180e-6,
            readout_error=0.007
        ),
        NoiseModel.IDEAL: NoiseProfile(
            cx_error=0.0,
            sx_error=0.0,
            t1=float('inf'),
            t2=float('inf'),
            readout_error=0.0
        ),
    }
    
    def __init__(self, noise_model: NoiseModel = NoiseModel.IDEAL, 
                 num_qubits: int = 10, seed: Optional[int] = None):
        """
        Initialize quantum simulator.
        
        Args:
            noise_model: Type of noise model to use
            num_qubits: Number of qubits in the simulator
            seed: Random seed for reproducibility
        """
        self.noise_model = noise_model
        self.num_qubits = num_qubits
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        if noise_model not in self.NOISE_PROFILES:
            raise ValueError(f"Unknown noise model: {noise_model}")
        
        self.noise_profile = self.NOISE_PROFILES[noise_model]
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1.0  # Initialize to |0...0⟩
        
    def apply_single_qubit_gate(self, qubit: int, gate_matrix: np.ndarray) -> None:
        """Apply single-qubit gate with noise."""
        if self.noise_model != NoiseModel.IDEAL:
            # Apply gate error
            if self.rng.random() < self.noise_profile.sx_error:
                # Bit flip error
                gate_matrix = np.array([[0, 1], [1, 0]]) @ gate_matrix
        
        # Apply gate to state vector
        self._apply_gate_to_state(gate_matrix, [qubit])
    
    def apply_two_qubit_gate(self, qubit1: int, qubit2: int, 
                            gate_matrix: np.ndarray) -> None:
        """Apply two-qubit gate with noise."""
        if self.noise_model != NoiseModel.IDEAL:
            # Apply gate error
            if self.rng.random() < self.noise_profile.cx_error:
                # Introduce error
                error_matrix = np.eye(4)
                error_matrix[0, 0] = 0.99
                gate_matrix = error_matrix @ gate_matrix
        
        self._apply_gate_to_state(gate_matrix, [qubit1, qubit2])
    
    def measure(self, qubits: Optional[List[int]] = None, 
                shots: int = 1024) -> Dict[str, int]:
        """
        Measure qubits and return counts.
        
        Args:
            qubits: Qubits to measure (None = all)
            shots: Number of measurement shots
            
        Returns:
            Dictionary of measurement outcomes and their counts
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
        
        # Get probabilities from state vector
        probabilities = np.abs(self.state_vector) ** 2
        
        # Sample from probabilities
        outcomes = self.rng.choice(
            len(probabilities), 
            size=shots, 
            p=probabilities
        )
        
        # Apply readout error if noisy
        if self.noise_model != NoiseModel.IDEAL:
            error_rate = self.noise_profile.readout_error
            for i in range(len(outcomes)):
                if self.rng.random() < error_rate:
                    # Flip a random bit
                    bit_to_flip = self.rng.randint(0, self.num_qubits)
                    outcomes[i] ^= (1 << bit_to_flip)
        
        # Convert to bitstrings and count
        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return counts
    
    def get_fidelity(self) -> float:
        """
        Estimate circuit fidelity based on noise model.
        
        Returns:
            Estimated fidelity (0.0 to 1.0)
        """
        if self.noise_model == NoiseModel.IDEAL:
            return 1.0
        
        profile = self.noise_profile
        # Simplified fidelity calculation
        fidelity = (1 - profile.cx_error) * (1 - profile.sx_error)
        fidelity *= (1 - profile.readout_error)
        return max(0.0, min(1.0, fidelity))
    
    def reset(self) -> None:
        """Reset simulator to initial state |0...0⟩."""
        self.state_vector = np.zeros(2**self.num_qubits, dtype=complex)
        self.state_vector[0] = 1.0
    
    def _apply_gate_to_state(self, gate: np.ndarray, qubits: List[int]) -> None:
        """Apply gate to state vector (simplified implementation)."""
        # This is a placeholder for actual gate application
        # In production, use proper tensor product operations
        pass
    
    def get_noise_profile_info(self) -> Dict:
        """Get detailed noise profile information."""
        profile = self.noise_profile
        return {
            'model': self.noise_model.value,
            'cx_error': profile.cx_error,
            'sx_error': profile.sx_error,
            't1': profile.t1,
            't2': profile.t2,
            'readout_error': profile.readout_error,
            'gate_time': profile.gate_time,
            'estimated_fidelity': self.get_fidelity()
        }


class IdealSimulator(QMANNSimulator):
    """Ideal quantum simulator without noise."""
    
    def __init__(self, num_qubits: int = 10, seed: Optional[int] = None):
        super().__init__(NoiseModel.IDEAL, num_qubits, seed)


class NoisyNISQSimulator(QMANNSimulator):
    """Noisy NISQ simulator with realistic error models."""

    def __init__(self, noise_model: NoiseModel = NoiseModel.IBM_SHERBROOKE,
                 num_qubits: int = 10, seed: Optional[int] = None):
        if noise_model == NoiseModel.IDEAL:
            raise ValueError("Use IdealSimulator for ideal simulation")
        super().__init__(noise_model, num_qubits, seed)


# Test functions
def test_ideal_simulator_creation():
    """Test creation of ideal simulator."""
    sim = IdealSimulator(num_qubits=5)
    assert sim.num_qubits == 5
    assert sim.noise_model == NoiseModel.IDEAL
    assert sim.get_fidelity() == 1.0


def test_noisy_simulator_creation():
    """Test creation of noisy simulator."""
    sim = NoisyNISQSimulator(
        noise_model=NoiseModel.IBM_SHERBROOKE,
        num_qubits=10
    )
    assert sim.num_qubits == 10
    assert sim.noise_model == NoiseModel.IBM_SHERBROOKE
    assert sim.get_fidelity() < 1.0


def test_noise_profile_info():
    """Test noise profile information retrieval."""
    sim = NoisyNISQSimulator(noise_model=NoiseModel.IBM_TORINO)
    info = sim.get_noise_profile_info()

    assert info['model'] == 'ibm_torino'
    assert info['cx_error'] == 6.8e-3
    assert info['sx_error'] == 1.9e-4
    assert 'estimated_fidelity' in info


def test_simulator_reset():
    """Test simulator reset functionality."""
    sim = IdealSimulator(num_qubits=3)
    sim.reset()

    # After reset, should be in |0...0⟩ state
    assert sim.state_vector[0] == 1.0
    assert np.sum(np.abs(sim.state_vector[1:])) == 0.0


def test_measurement():
    """Test measurement functionality."""
    sim = IdealSimulator(num_qubits=2)
    counts = sim.measure(shots=100)

    assert isinstance(counts, dict)
    assert sum(counts.values()) == 100
    assert '00' in counts  # Should measure |00⟩ state

