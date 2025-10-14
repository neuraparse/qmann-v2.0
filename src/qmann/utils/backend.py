"""
Quantum Backend Management

Handles quantum backend selection, configuration, and access
for IBM Quantum and other providers using 2025 APIs.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import warnings

# Qiskit 2.1+ imports
# from qiskit import IBMQ  # Deprecated in Qiskit 2.0+
from qiskit.providers import Backend
# from qiskit.providers.fake_provider import FakeProvider  # Deprecated
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
# from qiskit_ibm_provider import IBMProvider  # Deprecated in Qiskit 2.0+

from ..core.exceptions import BackendError, HardwareError


class QuantumBackend:
    """
    Quantum backend management for QMANN.
    
    Provides unified interface for accessing quantum hardware and simulators
    with automatic fallback and error handling.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Backend cache
        self._backends = {}
        self._current_backend = None
        
        # Service instances
        self._runtime_service = None
        self._ibm_provider = None
        
        # Hardware specifications cache
        self._hardware_specs = {}
    
    @classmethod
    def list_available(cls) -> List[Dict[str, Any]]:
        """List all available quantum backends."""
        backends = []
        
        try:
            # IBM Quantum backends
            service = QiskitRuntimeService()
            ibm_backends = service.backends()
            
            for backend in ibm_backends:
                config = backend.configuration()
                backends.append({
                    'name': backend.name,
                    'provider': 'IBM Quantum',
                    'type': 'hardware' if not config.simulator else 'simulator',
                    'qubits': getattr(config, 'n_qubits', 0),
                    'status': 'available',
                    'queue_length': getattr(backend.status(), 'pending_jobs', 0)
                })
                
        except Exception as e:
            logging.warning(f"Could not access IBM Quantum backends: {e}")
        
        # Add local simulators
        backends.extend([
            {
                'name': 'aer_simulator',
                'provider': 'Qiskit Aer',
                'type': 'simulator',
                'qubits': 32,  # Default Aer limit
                'status': 'available',
                'queue_length': 0
            },
            {
                'name': 'statevector_simulator',
                'provider': 'Qiskit Aer',
                'type': 'simulator',
                'qubits': 20,  # Memory limited
                'status': 'available',
                'queue_length': 0
            }
        ])
        
        return backends
    
    @classmethod
    def get_backend(
        cls, 
        backend_name: str = "aer_simulator", 
        use_hardware: bool = False,
        **kwargs
    ) -> Backend:
        """
        Get quantum backend by name with automatic configuration.
        
        Args:
            backend_name: Name of the backend
            use_hardware: Whether to prefer hardware over simulators
            **kwargs: Additional backend configuration
            
        Returns:
            Configured quantum backend
        """
        instance = cls()
        return instance._get_backend_instance(backend_name, use_hardware, **kwargs)
    
    def _get_backend_instance(
        self, 
        backend_name: str, 
        use_hardware: bool = False,
        **kwargs
    ) -> Backend:
        """Internal method to get backend instance."""
        
        # Check cache first
        cache_key = f"{backend_name}_{use_hardware}"
        if cache_key in self._backends:
            return self._backends[cache_key]
        
        backend = None
        
        try:
            if use_hardware and backend_name != "aer_simulator":
                # Try to get hardware backend
                backend = self._get_hardware_backend(backend_name, **kwargs)
            else:
                # Get simulator backend
                backend = self._get_simulator_backend(backend_name, **kwargs)
                
        except Exception as e:
            self.logger.warning(f"Failed to get backend {backend_name}: {e}")
            # Fallback to Aer simulator
            backend = self._get_fallback_backend()
        
        if backend is None:
            raise BackendError(f"Could not initialize any backend")
        
        # Cache the backend
        self._backends[cache_key] = backend
        self._current_backend = backend
        
        self.logger.info(f"Using quantum backend: {backend.name}")
        return backend
    
    def _get_hardware_backend(self, backend_name: str, **kwargs) -> Backend:
        """Get hardware backend from IBM Quantum."""
        try:
            # Initialize IBM Quantum Runtime Service
            if self._runtime_service is None:
                self._runtime_service = QiskitRuntimeService()
            
            # Get specific backend
            if backend_name == "ibm_quantum":
                # Get best available backend
                backends = self._runtime_service.backends(
                    operational=True,
                    simulator=False
                )
                if not backends:
                    raise BackendError("No operational hardware backends available")
                
                # Select backend with most qubits and lowest queue
                best_backend = min(
                    backends,
                    key=lambda b: (getattr(b.status(), 'pending_jobs', float('inf')), -getattr(b.configuration(), 'n_qubits', 0))
                )
                return best_backend
            else:
                # Get specific named backend
                return self._runtime_service.backend(backend_name)
                
        except Exception as e:
            raise BackendError(f"Failed to access IBM Quantum backend {backend_name}: {e}")
    
    def _get_simulator_backend(self, backend_name: str, **kwargs) -> Backend:
        """Get simulator backend."""
        if backend_name == "aer_simulator":
            return AerSimulator()
        elif backend_name == "statevector_simulator":
            return AerSimulator(method='statevector')
        elif backend_name == "qasm_simulator":
            return AerSimulator(method='qasm')
        else:
            # Try to get fake backend for testing (deprecated in Qiskit 2.0+)
            # try:
            #     fake_provider = FakeProvider()
            #     return fake_provider.get_backend(backend_name)
            # except Exception:
            raise BackendError(f"Unknown simulator backend: {backend_name}")
    
    def _get_fallback_backend(self) -> Backend:
        """Get fallback backend when all else fails."""
        try:
            return AerSimulator()
        except Exception as e:
            raise BackendError(f"Even fallback backend failed: {e}")
    
    def get_backend_specs(self, backend: Backend) -> Dict[str, Any]:
        """Get detailed specifications for a backend."""
        backend_name = backend.name
        
        if backend_name in self._hardware_specs:
            return self._hardware_specs[backend_name]
        
        try:
            config = backend.configuration()
            
            specs = {
                'name': backend_name,
                'num_qubits': getattr(config, 'n_qubits', 0),
                'simulator': getattr(config, 'simulator', True),
                'local': getattr(config, 'local', True),
                'coupling_map': getattr(config, 'coupling_map', None),
                'basis_gates': getattr(config, 'basis_gates', []),
                'gate_times': getattr(config, 'gate_times', {}),
                'gate_errors': getattr(config, 'gate_errors', {}),
                'readout_errors': getattr(config, 'readout_error', []),
                'max_shots': getattr(config, 'max_shots', 8192),
                'max_experiments': getattr(config, 'max_experiments', 1),
            }
            
            # Add status information if available
            try:
                status = backend.status()
                specs.update({
                    'operational': getattr(status, 'operational', True),
                    'pending_jobs': getattr(status, 'pending_jobs', 0),
                    'status_msg': getattr(status, 'status_msg', 'Available'),
                })
            except Exception:
                pass
            
            # Cache the specs
            self._hardware_specs[backend_name] = specs
            
            return specs
            
        except Exception as e:
            self.logger.warning(f"Could not get specs for backend {backend_name}: {e}")
            return {'name': backend_name, 'error': str(e)}
    
    def validate_circuit_compatibility(self, circuit, backend: Backend) -> Dict[str, Any]:
        """
        Validate if a circuit is compatible with the backend.
        
        Args:
            circuit: Quantum circuit to validate
            backend: Target backend
            
        Returns:
            Validation results with compatibility info and suggestions
        """
        specs = self.get_backend_specs(backend)
        issues = []
        suggestions = []
        
        # Check qubit count
        if circuit.num_qubits > specs.get('num_qubits', 0):
            issues.append(f"Circuit requires {circuit.num_qubits} qubits, backend has {specs.get('num_qubits', 0)}")
            suggestions.append("Reduce circuit size or use a larger backend")
        
        # Check basis gates
        circuit_gates = set(instr.operation.name for instr in circuit.data)
        backend_gates = set(specs.get('basis_gates', []))
        
        if backend_gates and not circuit_gates.issubset(backend_gates):
            unsupported = circuit_gates - backend_gates
            issues.append(f"Unsupported gates: {unsupported}")
            suggestions.append("Transpile circuit to backend basis gates")
        
        # Check circuit depth
        if circuit.depth() > 1000:  # Arbitrary threshold
            issues.append(f"Circuit depth ({circuit.depth()}) may be too deep for NISQ devices")
            suggestions.append("Reduce circuit depth or use error mitigation")
        
        # Check coupling map compatibility
        coupling_map = specs.get('coupling_map')
        if coupling_map and circuit.num_qubits > 1:
            # Simplified check - in practice would analyze all two-qubit gates
            suggestions.append("Consider circuit layout optimization for coupling map")
        
        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'suggestions': suggestions,
            'backend_specs': specs,
            'circuit_stats': {
                'num_qubits': circuit.num_qubits,
                'depth': circuit.depth(),
                'size': circuit.size(),
                'gates': list(circuit_gates)
            }
        }
    
    def get_optimal_backend(
        self, 
        circuit, 
        prefer_hardware: bool = False,
        min_qubits: Optional[int] = None
    ) -> Backend:
        """
        Get optimal backend for a given circuit.
        
        Args:
            circuit: Quantum circuit to execute
            prefer_hardware: Whether to prefer hardware over simulators
            min_qubits: Minimum number of qubits required
            
        Returns:
            Optimal backend for the circuit
        """
        available_backends = self.list_available()
        
        # Filter by requirements
        suitable_backends = []
        required_qubits = max(circuit.num_qubits, min_qubits or 0)
        
        for backend_info in available_backends:
            if backend_info['qubits'] >= required_qubits:
                if backend_info['status'] == 'available':
                    suitable_backends.append(backend_info)
        
        if not suitable_backends:
            raise HardwareError(
                f"No suitable backends found for {required_qubits} qubits",
                required_qubits=required_qubits,
                available_qubits=max(b['qubits'] for b in available_backends) if available_backends else 0
            )
        
        # Sort by preference
        def backend_score(backend_info):
            score = 0
            
            # Prefer hardware if requested
            if prefer_hardware and backend_info['type'] == 'hardware':
                score += 1000
            elif not prefer_hardware and backend_info['type'] == 'simulator':
                score += 1000
            
            # Prefer fewer queued jobs
            score -= backend_info['queue_length'] * 10
            
            # Prefer more qubits (but not too many more)
            qubit_excess = backend_info['qubits'] - required_qubits
            if qubit_excess < 10:
                score += qubit_excess
            else:
                score -= qubit_excess  # Penalty for excessive qubits
            
            return score
        
        # Select best backend
        best_backend_info = max(suitable_backends, key=backend_score)
        
        # Get the actual backend
        return self.get_backend(
            best_backend_info['name'],
            use_hardware=(best_backend_info['type'] == 'hardware')
        )
    
    def get_current_backend(self) -> Optional[Backend]:
        """Get currently active backend."""
        return self._current_backend
    
    def clear_cache(self) -> None:
        """Clear backend cache."""
        self._backends.clear()
        self._hardware_specs.clear()
        self.logger.debug("Backend cache cleared")
