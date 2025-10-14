"""
Multi-Provider Quantum Backend Manager (2025)

Unified interface for accessing quantum hardware from multiple providers:
- IBM Quantum (127-qubit superconducting)
- IonQ (trapped-ion quantum computers)
- AWS Braket (Rigetti, IonQ, D-Wave via AWS)
- Rigetti Quantum Cloud Services (superconducting)

Provides automatic provider selection, cost optimization, and failover.

Author: QMANN Development Team
Date: October 2025
Version: 2.2.0
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Qiskit core
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_aer import AerSimulator

# IBM Quantum
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Batch, Options

# IonQ
try:
    from qiskit_ionq import IonQProvider
    IONQ_AVAILABLE = True
except ImportError:
    IONQ_AVAILABLE = False
    warnings.warn("IonQ provider not available. Install with: pip install qiskit-ionq")

# AWS Braket
try:
    from qiskit_braket_provider import AWSBraketProvider
    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False
    warnings.warn("AWS Braket provider not available. Install with: pip install qiskit-braket-provider")

# Rigetti
try:
    from qiskit_rigetti import RigettiQCSProvider
    RIGETTI_AVAILABLE = True
except ImportError:
    RIGETTI_AVAILABLE = False
    warnings.warn("Rigetti provider not available. Install with: pip install qiskit-rigetti")

from ..core.exceptions import BackendError, HardwareError

logger = logging.getLogger(__name__)


class QuantumProvider(Enum):
    """Supported quantum computing providers."""
    IBM_QUANTUM = "ibm_quantum"
    IONQ = "ionq"
    AWS_BRAKET = "aws_braket"
    RIGETTI = "rigetti"
    SIMULATOR = "simulator"


@dataclass
class ProviderConfig:
    """Configuration for quantum provider access."""
    
    # IBM Quantum
    ibm_token: Optional[str] = None
    ibm_channel: str = "ibm_quantum"
    ibm_instance: Optional[str] = None
    
    # IonQ
    ionq_api_key: Optional[str] = None
    ionq_url: str = "https://api.ionq.co/v0.3"
    
    # AWS Braket
    aws_region: str = "us-east-1"
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None
    aws_s3_bucket: Optional[str] = None
    
    # Rigetti
    rigetti_qcs_url: Optional[str] = None
    rigetti_api_key: Optional[str] = None
    
    # Provider preferences
    preferred_provider: QuantumProvider = QuantumProvider.IBM_QUANTUM
    enable_auto_fallback: bool = True
    cost_optimization: bool = True
    
    @classmethod
    def from_environment(cls) -> 'ProviderConfig':
        """Load configuration from environment variables."""
        return cls(
            ibm_token=os.getenv('QISKIT_IBM_TOKEN'),
            ibm_channel=os.getenv('QISKIT_IBM_CHANNEL', 'ibm_quantum'),
            ibm_instance=os.getenv('QISKIT_IBM_INSTANCE'),
            ionq_api_key=os.getenv('IONQ_API_KEY'),
            aws_region=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_s3_bucket=os.getenv('AWS_S3_BUCKET'),
            rigetti_api_key=os.getenv('RIGETTI_API_KEY'),
        )


@dataclass
class BackendInfo:
    """Information about a quantum backend."""
    name: str
    provider: QuantumProvider
    num_qubits: int
    is_simulator: bool
    is_available: bool
    queue_length: int = 0
    cost_per_shot: float = 0.0
    gate_fidelity: float = 0.0
    coherence_time: float = 0.0
    topology: str = "all-to-all"
    backend_version: str = "unknown"
    
    def score(self, prefer_hardware: bool = True, cost_weight: float = 0.3) -> float:
        """Calculate backend selection score."""
        score = 0.0
        
        # Hardware preference
        if prefer_hardware and not self.is_simulator:
            score += 50.0
        elif not prefer_hardware and self.is_simulator:
            score += 50.0
        
        # Availability
        if self.is_available:
            score += 20.0
        
        # Queue length (lower is better)
        score -= min(self.queue_length * 0.5, 20.0)
        
        # Cost (lower is better)
        if self.cost_per_shot > 0:
            score -= min(self.cost_per_shot * cost_weight * 10, 15.0)
        
        # Quality metrics
        score += self.gate_fidelity * 10.0
        score += min(self.coherence_time / 100e-6, 5.0)  # Normalize to 100μs
        
        return max(score, 0.0)


class MultiProviderBackendManager:
    """
    Unified quantum backend manager for multiple providers.
    
    Provides intelligent backend selection, automatic failover,
    and cost optimization across quantum computing providers.
    """
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config or ProviderConfig.from_environment()
        self.logger = logging.getLogger(__name__)
        
        # Provider instances
        self._ibm_service: Optional[QiskitRuntimeService] = None
        self._ionq_provider: Optional[Any] = None
        self._braket_provider: Optional[Any] = None
        self._rigetti_provider: Optional[Any] = None
        
        # Backend cache
        self._backends_cache: Dict[str, Backend] = {}
        self._backend_info_cache: List[BackendInfo] = []
        self._cache_timestamp: float = 0.0
        self._cache_ttl: float = 300.0  # 5 minutes
        
        # Statistics
        self.stats = {
            'backends_accessed': 0,
            'jobs_submitted': 0,
            'total_shots': 0,
            'total_cost': 0.0,
            'provider_usage': {provider.value: 0 for provider in QuantumProvider}
        }
        
        self.logger.info("Initialized Multi-Provider Backend Manager")
    
    def initialize_providers(self) -> Dict[QuantumProvider, bool]:
        """Initialize all available quantum providers."""
        status = {}
        
        # IBM Quantum
        try:
            if self.config.ibm_token:
                QiskitRuntimeService.save_account(
                    channel=self.config.ibm_channel,
                    token=self.config.ibm_token,
                    instance=self.config.ibm_instance,
                    overwrite=True
                )
            self._ibm_service = QiskitRuntimeService()
            status[QuantumProvider.IBM_QUANTUM] = True
            self.logger.info("[OK] IBM Quantum initialized")
        except Exception as e:
            status[QuantumProvider.IBM_QUANTUM] = False
            self.logger.warning(f"[X] IBM Quantum initialization failed: {e}")
        
        # IonQ
        if IONQ_AVAILABLE and self.config.ionq_api_key:
            try:
                self._ionq_provider = IonQProvider(self.config.ionq_api_key)
                status[QuantumProvider.IONQ] = True
                self.logger.info("[OK] IonQ initialized")
            except Exception as e:
                status[QuantumProvider.IONQ] = False
                self.logger.warning(f"[X] IonQ initialization failed: {e}")
        else:
            status[QuantumProvider.IONQ] = False

        # AWS Braket
        if BRAKET_AVAILABLE:
            try:
                self._braket_provider = AWSBraketProvider()
                status[QuantumProvider.AWS_BRAKET] = True
                self.logger.info("[OK] AWS Braket initialized")
            except Exception as e:
                status[QuantumProvider.AWS_BRAKET] = False
                self.logger.warning(f"[X] AWS Braket initialization failed: {e}")
        else:
            status[QuantumProvider.AWS_BRAKET] = False

        # Rigetti
        if RIGETTI_AVAILABLE and self.config.rigetti_api_key:
            try:
                self._rigetti_provider = RigettiQCSProvider()
                status[QuantumProvider.RIGETTI] = True
                self.logger.info("[OK] Rigetti initialized")
            except Exception as e:
                status[QuantumProvider.RIGETTI] = False
                self.logger.warning(f"[X] Rigetti initialization failed: {e}")
        else:
            status[QuantumProvider.RIGETTI] = False

        # Simulator always available
        status[QuantumProvider.SIMULATOR] = True
        self.logger.info("[OK] Local simulators available")
        
        return status
    
    def list_all_backends(self, refresh: bool = False) -> List[BackendInfo]:
        """List all available backends from all providers."""
        import time
        
        # Check cache
        if not refresh and self._backend_info_cache:
            if time.time() - self._cache_timestamp < self._cache_ttl:
                return self._backend_info_cache
        
        backends = []
        
        # IBM Quantum backends
        backends.extend(self._list_ibm_backends())
        
        # IonQ backends
        backends.extend(self._list_ionq_backends())
        
        # AWS Braket backends
        backends.extend(self._list_braket_backends())
        
        # Rigetti backends
        backends.extend(self._list_rigetti_backends())
        
        # Local simulators
        backends.extend(self._list_simulator_backends())
        
        # Update cache
        self._backend_info_cache = backends
        self._cache_timestamp = time.time()
        
        self.logger.info(f"Found {len(backends)} total backends across all providers")
        return backends
    
    def _list_ibm_backends(self) -> List[BackendInfo]:
        """List IBM Quantum backends."""
        backends = []
        
        if not self._ibm_service:
            return backends
        
        try:
            ibm_backends = self._ibm_service.backends()
            
            for backend in ibm_backends:
                try:
                    config = backend.configuration()
                    status = backend.status()
                    
                    backends.append(BackendInfo(
                        name=backend.name,
                        provider=QuantumProvider.IBM_QUANTUM,
                        num_qubits=getattr(config, 'n_qubits', 0),
                        is_simulator=getattr(config, 'simulator', False),
                        is_available=status.operational and not status.status_msg == 'offline',
                        queue_length=getattr(status, 'pending_jobs', 0),
                        cost_per_shot=0.0,  # IBM Quantum pricing varies
                        gate_fidelity=0.999,  # Approximate
                        coherence_time=100e-6,  # Approximate
                        topology="heavy-hex" if config.n_qubits > 50 else "linear",
                        backend_version=getattr(config, 'backend_version', 'unknown')
                    ))
                except Exception as e:
                    self.logger.debug(f"Could not get info for IBM backend: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Could not list IBM backends: {e}")
        
        return backends
    
    def _list_ionq_backends(self) -> List[BackendInfo]:
        """List IonQ backends."""
        backends = []
        
        if not self._ionq_provider:
            return backends
        
        try:
            ionq_backends = self._ionq_provider.backends()
            
            for backend in ionq_backends:
                backends.append(BackendInfo(
                    name=backend.name(),
                    provider=QuantumProvider.IONQ,
                    num_qubits=backend.configuration().n_qubits,
                    is_simulator='simulator' in backend.name().lower(),
                    is_available=backend.status().operational,
                    queue_length=0,  # IonQ doesn't expose queue
                    cost_per_shot=0.00030,  # Approximate IonQ pricing
                    gate_fidelity=0.9999,  # IonQ trapped-ion high fidelity
                    coherence_time=1000e-6,  # Trapped ions have long coherence
                    topology="all-to-all",  # IonQ advantage
                    backend_version="aria-1" if "aria" in backend.name().lower() else "harmony"
                ))
        except Exception as e:
            self.logger.warning(f"Could not list IonQ backends: {e}")
        
        return backends
    
    def _list_braket_backends(self) -> List[BackendInfo]:
        """List AWS Braket backends."""
        backends = []
        
        if not self._braket_provider:
            return backends
        
        try:
            braket_backends = self._braket_provider.backends()
            
            for backend in braket_backends:
                backends.append(BackendInfo(
                    name=backend.name,
                    provider=QuantumProvider.AWS_BRAKET,
                    num_qubits=backend.num_qubits,
                    is_simulator='sv1' in backend.name or 'tn1' in backend.name,
                    is_available=True,  # AWS Braket backends generally available
                    queue_length=0,
                    cost_per_shot=0.00030 if 'ionq' in backend.name.lower() else 0.00035,
                    gate_fidelity=0.999,
                    coherence_time=100e-6,
                    topology="varies",
                    backend_version="braket"
                ))
        except Exception as e:
            self.logger.warning(f"Could not list AWS Braket backends: {e}")
        
        return backends
    
    def _list_rigetti_backends(self) -> List[BackendInfo]:
        """List Rigetti backends."""
        backends = []
        
        if not self._rigetti_provider:
            return backends
        
        try:
            rigetti_backends = self._rigetti_provider.backends()
            
            for backend in rigetti_backends:
                backends.append(BackendInfo(
                    name=backend.name(),
                    provider=QuantumProvider.RIGETTI,
                    num_qubits=backend.configuration().n_qubits,
                    is_simulator='qvm' in backend.name().lower(),
                    is_available=backend.status().operational,
                    queue_length=0,
                    cost_per_shot=0.00035,  # Approximate Rigetti pricing
                    gate_fidelity=0.998,
                    coherence_time=50e-6,
                    topology="square-lattice",
                    backend_version="aspen-m"
                ))
        except Exception as e:
            self.logger.warning(f"Could not list Rigetti backends: {e}")
        
        return backends
    
    def _list_simulator_backends(self) -> List[BackendInfo]:
        """List local simulator backends."""
        return [
            BackendInfo(
                name="aer_simulator",
                provider=QuantumProvider.SIMULATOR,
                num_qubits=32,
                is_simulator=True,
                is_available=True,
                queue_length=0,
                cost_per_shot=0.0,
                gate_fidelity=1.0,
                coherence_time=float('inf'),
                topology="all-to-all",
                backend_version="qiskit-aer"
            ),
            BackendInfo(
                name="statevector_simulator",
                provider=QuantumProvider.SIMULATOR,
                num_qubits=20,
                is_simulator=True,
                is_available=True,
                queue_length=0,
                cost_per_shot=0.0,
                gate_fidelity=1.0,
                coherence_time=float('inf'),
                topology="all-to-all",
                backend_version="qiskit-aer"
            )
        ]

    def get_best_backend(
        self,
        num_qubits: int = 1,
        prefer_hardware: bool = False,
        max_cost_per_shot: float = float('inf'),
        provider_preference: Optional[QuantumProvider] = None
    ) -> Tuple[Backend, BackendInfo]:
        """
        Get the best available backend based on requirements.

        Args:
            num_qubits: Minimum number of qubits required
            prefer_hardware: Prefer hardware over simulators
            max_cost_per_shot: Maximum acceptable cost per shot
            provider_preference: Preferred provider (optional)

        Returns:
            Tuple of (Backend instance, BackendInfo)
        """
        # Get all backends
        all_backends = self.list_all_backends()

        # Filter by requirements
        candidates = [
            b for b in all_backends
            if b.num_qubits >= num_qubits
            and b.is_available
            and b.cost_per_shot <= max_cost_per_shot
        ]

        if not candidates:
            # Fallback to simulator
            self.logger.warning("No suitable backends found, falling back to simulator")
            return self._get_simulator_backend(), self._list_simulator_backends()[0]

        # Apply provider preference
        if provider_preference:
            preferred = [b for b in candidates if b.provider == provider_preference]
            if preferred:
                candidates = preferred

        # Score and sort
        candidates.sort(key=lambda b: b.score(prefer_hardware, cost_weight=0.3), reverse=True)

        # Get best backend
        best_info = candidates[0]
        best_backend = self._get_backend_instance(best_info)

        self.logger.info(f"Selected backend: {best_info.name} ({best_info.provider.value})")
        self.stats['backends_accessed'] += 1
        self.stats['provider_usage'][best_info.provider.value] += 1

        return best_backend, best_info

    def _get_backend_instance(self, info: BackendInfo) -> Backend:
        """Get actual backend instance from BackendInfo."""
        cache_key = f"{info.provider.value}:{info.name}"

        # Check cache
        if cache_key in self._backends_cache:
            return self._backends_cache[cache_key]

        # Get backend based on provider
        if info.provider == QuantumProvider.IBM_QUANTUM:
            backend = self._ibm_service.backend(info.name)
        elif info.provider == QuantumProvider.IONQ:
            backend = self._ionq_provider.get_backend(info.name)
        elif info.provider == QuantumProvider.AWS_BRAKET:
            backend = self._braket_provider.get_backend(info.name)
        elif info.provider == QuantumProvider.RIGETTI:
            backend = self._rigetti_provider.get_backend(info.name)
        elif info.provider == QuantumProvider.SIMULATOR:
            backend = self._get_simulator_backend(info.name)
        else:
            raise BackendError(f"Unknown provider: {info.provider}")

        # Cache and return
        self._backends_cache[cache_key] = backend
        return backend

    def _get_simulator_backend(self, name: str = "aer_simulator") -> Backend:
        """Get local simulator backend."""
        if name == "statevector_simulator":
            return AerSimulator(method='statevector')
        else:
            return AerSimulator()

    def execute_with_session(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        backend_info: BackendInfo,
        shots: int = 8192,
        optimization_level: int = 3
    ) -> Any:
        """
        Execute circuits with session-based optimization.

        Args:
            circuits: Quantum circuit(s) to execute
            backend_info: Backend information
            shots: Number of measurement shots
            optimization_level: Transpilation optimization level

        Returns:
            Execution results
        """
        backend = self._get_backend_instance(backend_info)

        # Use session for IBM Quantum
        if backend_info.provider == QuantumProvider.IBM_QUANTUM:
            with Session(service=self._ibm_service, backend=backend) as session:
                from qiskit_ibm_runtime import SamplerV2
                sampler = SamplerV2(session=session)

                # Prepare circuits
                if not isinstance(circuits, list):
                    circuits = [circuits]

                # Execute
                job = sampler.run(circuits, shots=shots)
                result = job.result()

                # Update stats
                self.stats['jobs_submitted'] += 1
                self.stats['total_shots'] += shots * len(circuits)
                self.stats['total_cost'] += backend_info.cost_per_shot * shots * len(circuits)

                return result
        else:
            # Direct execution for other providers
            job = backend.run(circuits, shots=shots)
            result = job.result()

            # Update stats
            self.stats['jobs_submitted'] += 1
            self.stats['total_shots'] += shots
            self.stats['total_cost'] += backend_info.cost_per_shot * shots

            return result

    def execute_batch(
        self,
        circuit_batches: List[List[QuantumCircuit]],
        backend_info: BackendInfo,
        shots: int = 8192
    ) -> List[Any]:
        """
        Execute multiple batches of circuits with cost optimization.

        Args:
            circuit_batches: List of circuit batches
            backend_info: Backend information
            shots: Number of measurement shots

        Returns:
            List of results for each batch
        """
        backend = self._get_backend_instance(backend_info)

        # Use Batch for IBM Quantum
        if backend_info.provider == QuantumProvider.IBM_QUANTUM:
            with Batch(service=self._ibm_service, backend=backend) as batch:
                from qiskit_ibm_runtime import SamplerV2
                sampler = SamplerV2(session=batch)

                results = []
                for circuits in circuit_batches:
                    job = sampler.run(circuits, shots=shots)
                    results.append(job.result())

                # Update stats
                total_circuits = sum(len(batch) for batch in circuit_batches)
                self.stats['jobs_submitted'] += len(circuit_batches)
                self.stats['total_shots'] += shots * total_circuits
                self.stats['total_cost'] += backend_info.cost_per_shot * shots * total_circuits

                return results
        else:
            # Sequential execution for other providers
            results = []
            for circuits in circuit_batches:
                job = backend.run(circuits, shots=shots)
                results.append(job.result())

            return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            **self.stats,
            'available_providers': [
                provider.value for provider, available in self.initialize_providers().items()
                if available
            ],
            'total_backends': len(self.list_all_backends()),
            'cache_size': len(self._backends_cache)
        }

    def estimate_cost(
        self,
        num_circuits: int,
        shots_per_circuit: int,
        backend_info: BackendInfo
    ) -> Dict[str, float]:
        """
        Estimate execution cost.

        Args:
            num_circuits: Number of circuits to execute
            shots_per_circuit: Shots per circuit
            backend_info: Backend information

        Returns:
            Cost estimation dictionary
        """
        total_shots = num_circuits * shots_per_circuit
        total_cost = total_shots * backend_info.cost_per_shot

        return {
            'total_shots': total_shots,
            'cost_per_shot': backend_info.cost_per_shot,
            'total_cost_usd': total_cost,
            'backend': backend_info.name,
            'provider': backend_info.provider.value
        }


# Convenience function
def get_quantum_backend(
    num_qubits: int = 1,
    prefer_hardware: bool = False,
    provider: Optional[str] = None
) -> Tuple[Backend, BackendInfo]:
    """
    Convenience function to get a quantum backend.

    Args:
        num_qubits: Minimum number of qubits required
        prefer_hardware: Prefer hardware over simulators
        provider: Preferred provider name (optional)

    Returns:
        Tuple of (Backend instance, BackendInfo)
    """
    manager = MultiProviderBackendManager()
    manager.initialize_providers()

    provider_enum = None
    if provider:
        try:
            provider_enum = QuantumProvider(provider.lower())
        except ValueError:
            pass

    return manager.get_best_backend(
        num_qubits=num_qubits,
        prefer_hardware=prefer_hardware,
        provider_preference=provider_enum
    )

