"""
Multi-Provider Quantum Computing Demo (2025)

Demonstrates unified access to multiple quantum computing providers:
- IBM Quantum (superconducting qubits)
- IonQ (trapped-ion quantum computers)
- AWS Braket (multi-provider cloud)
- Rigetti (superconducting qubits)
- Local simulators

Shows intelligent backend selection, cost optimization, and batch execution.

Author: QMANN Development Team
Date: October 2025
"""

import sys
import time
import logging
from typing import List

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# QMANN imports
from qmann.utils import (
    MultiProviderBackendManager,
    ProviderConfig,
    QuantumProvider,
    get_quantum_backend
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_bell_state() -> QuantumCircuit:
    """Create a Bell state circuit."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def create_ghz_state(num_qubits: int = 3) -> QuantumCircuit:
    """Create a GHZ state circuit."""
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def create_quantum_fourier_transform(num_qubits: int = 3) -> QuantumCircuit:
    """Create a Quantum Fourier Transform circuit."""
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # QFT implementation
    for j in range(num_qubits):
        qc.h(j)
        for k in range(j + 1, num_qubits):
            qc.cp(3.14159 / (2 ** (k - j)), k, j)
    
    # Swap qubits
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - i - 1)
    
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def demo_provider_initialization():
    """Demonstrate provider initialization."""
    print("\n" + "=" * 80)
    print("  MULTI-PROVIDER INITIALIZATION")
    print("=" * 80)
    
    # Create manager
    manager = MultiProviderBackendManager()
    
    # Initialize all providers
    print("\nInitializing quantum providers...")
    status = manager.initialize_providers()
    
    print("\nProvider Status:")
    print("-" * 80)
    for provider, available in status.items():
        status_icon = "[OK]" if available else "[X]"
        status_text = "Available" if available else "Not Available"
        print(f"  {status_icon} {provider.value:20s} - {status_text}")
    
    return manager


def demo_backend_listing(manager: MultiProviderBackendManager):
    """Demonstrate backend listing."""
    print("\n" + "=" * 80)
    print("  AVAILABLE QUANTUM BACKENDS")
    print("=" * 80)
    
    backends = manager.list_all_backends(refresh=True)
    
    print(f"\nFound {len(backends)} total backends\n")
    
    # Group by provider
    by_provider = {}
    for backend in backends:
        provider = backend.provider.value
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append(backend)
    
    for provider, provider_backends in by_provider.items():
        print(f"\n{provider.upper()}:")
        print("-" * 80)
        for backend in provider_backends:
            availability = "[OK]" if backend.is_available else "[X]"
            backend_type = "SIM" if backend.is_simulator else "HW "
            print(f"  {availability} [{backend_type}] {backend.name:30s} "
                  f"| {backend.num_qubits:2d} qubits "
                  f"| Queue: {backend.queue_length:3d} "
                  f"| Cost: ${backend.cost_per_shot:.6f}/shot")


def demo_intelligent_backend_selection(manager: MultiProviderBackendManager):
    """Demonstrate intelligent backend selection."""
    print("\n" + "=" * 80)
    print("  INTELLIGENT BACKEND SELECTION")
    print("=" * 80)
    
    # Scenario 1: Small circuit, prefer simulator
    print("\n[1] Small Circuit (2 qubits) - Prefer Simulator")
    print("-" * 80)
    backend, info = manager.get_best_backend(
        num_qubits=2,
        prefer_hardware=False
    )
    print(f"  Selected: {info.name} ({info.provider.value})")
    print(f"  Qubits: {info.num_qubits}, Cost: ${info.cost_per_shot:.6f}/shot")
    print(f"  Score: {info.score(prefer_hardware=False):.2f}")
    
    # Scenario 2: Medium circuit, prefer hardware
    print("\n[2] Medium Circuit (8 qubits) - Prefer Hardware")
    print("-" * 80)
    backend, info = manager.get_best_backend(
        num_qubits=8,
        prefer_hardware=True,
        max_cost_per_shot=0.001
    )
    print(f"  Selected: {info.name} ({info.provider.value})")
    print(f"  Qubits: {info.num_qubits}, Cost: ${info.cost_per_shot:.6f}/shot")
    print(f"  Score: {info.score(prefer_hardware=True):.2f}")
    
    # Scenario 3: Large circuit, cost-optimized
    print("\n[3] Large Circuit (20 qubits) - Cost Optimized")
    print("-" * 80)
    backend, info = manager.get_best_backend(
        num_qubits=20,
        prefer_hardware=False,
        max_cost_per_shot=0.0001
    )
    print(f"  Selected: {info.name} ({info.provider.value})")
    print(f"  Qubits: {info.num_qubits}, Cost: ${info.cost_per_shot:.6f}/shot")
    print(f"  Score: {info.score(prefer_hardware=False, cost_weight=0.5):.2f}")


def demo_circuit_execution(manager: MultiProviderBackendManager):
    """Demonstrate circuit execution."""
    print("\n" + "=" * 80)
    print("  QUANTUM CIRCUIT EXECUTION")
    print("=" * 80)
    
    # Create test circuits
    bell_circuit = create_bell_state()
    ghz_circuit = create_ghz_state(3)
    
    # Get backend
    backend, info = manager.get_best_backend(num_qubits=3, prefer_hardware=False)
    
    print(f"\nExecuting on: {info.name} ({info.provider.value})")
    print("-" * 80)
    
    # Execute Bell state
    print("\n[1] Bell State Execution")
    start_time = time.time()
    result = manager.execute_with_session([bell_circuit], info, shots=1024)
    execution_time = time.time() - start_time
    print(f"  Execution Time: {execution_time:.4f}s")
    print(f"  Shots: 1024")
    print(f"  Cost: ${info.cost_per_shot * 1024:.6f}")
    
    # Execute GHZ state
    print("\n[2] GHZ State Execution")
    start_time = time.time()
    result = manager.execute_with_session([ghz_circuit], info, shots=1024)
    execution_time = time.time() - start_time
    print(f"  Execution Time: {execution_time:.4f}s")
    print(f"  Shots: 1024")
    print(f"  Cost: ${info.cost_per_shot * 1024:.6f}")


def demo_batch_execution(manager: MultiProviderBackendManager):
    """Demonstrate batch execution for cost optimization."""
    print("\n" + "=" * 80)
    print("  BATCH EXECUTION (Cost Optimization)")
    print("=" * 80)
    
    # Create multiple circuit batches
    batches = [
        [create_bell_state() for _ in range(5)],
        [create_ghz_state(3) for _ in range(5)],
        [create_quantum_fourier_transform(3) for _ in range(5)]
    ]
    
    # Get backend
    backend, info = manager.get_best_backend(num_qubits=3, prefer_hardware=False)
    
    print(f"\nExecuting 3 batches (5 circuits each) on: {info.name}")
    print("-" * 80)
    
    # Execute batches
    start_time = time.time()
    results = manager.execute_batch(batches, info, shots=512)
    execution_time = time.time() - start_time
    
    total_circuits = sum(len(batch) for batch in batches)
    total_shots = total_circuits * 512
    total_cost = info.cost_per_shot * total_shots
    
    print(f"\n  Total Circuits: {total_circuits}")
    print(f"  Total Shots: {total_shots}")
    print(f"  Execution Time: {execution_time:.4f}s")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Avg Time/Circuit: {execution_time/total_circuits:.4f}s")


def demo_cost_estimation(manager: MultiProviderBackendManager):
    """Demonstrate cost estimation."""
    print("\n" + "=" * 80)
    print("  COST ESTIMATION")
    print("=" * 80)
    
    # Get different backends
    backends = manager.list_all_backends()
    
    # Estimate costs for a typical workload
    num_circuits = 100
    shots_per_circuit = 8192
    
    print(f"\nWorkload: {num_circuits} circuits × {shots_per_circuit} shots")
    print("-" * 80)
    
    # Show cost for different backends
    for backend_info in backends[:5]:  # Show first 5
        cost_info = manager.estimate_cost(num_circuits, shots_per_circuit, backend_info)
        
        print(f"\n{backend_info.name} ({backend_info.provider.value}):")
        print(f"  Total Shots: {cost_info['total_shots']:,}")
        print(f"  Cost/Shot: ${cost_info['cost_per_shot']:.6f}")
        print(f"  Total Cost: ${cost_info['total_cost_usd']:.2f}")


def demo_statistics(manager: MultiProviderBackendManager):
    """Demonstrate usage statistics."""
    print("\n" + "=" * 80)
    print("  USAGE STATISTICS")
    print("=" * 80)
    
    stats = manager.get_statistics()
    
    print("\nOverall Statistics:")
    print("-" * 80)
    print(f"  Backends Accessed: {stats['backends_accessed']}")
    print(f"  Jobs Submitted: {stats['jobs_submitted']}")
    print(f"  Total Shots: {stats['total_shots']:,}")
    print(f"  Total Cost: ${stats['total_cost']:.6f}")
    print(f"  Available Providers: {len(stats['available_providers'])}")
    print(f"  Total Backends: {stats['total_backends']}")
    
    print("\nProvider Usage:")
    print("-" * 80)
    for provider, count in stats['provider_usage'].items():
        if count > 0:
            print(f"  {provider:20s}: {count} times")


def main():
    """Main demo function."""
    print("\n" + "=" * 80)
    print("  QMANN MULTI-PROVIDER QUANTUM COMPUTING DEMO (2025)")
    print("  Unified Access to IBM, IonQ, AWS Braket, Rigetti & Simulators")
    print("=" * 80)
    
    try:
        # Initialize providers
        manager = demo_provider_initialization()
        
        # List available backends
        demo_backend_listing(manager)
        
        # Demonstrate intelligent backend selection
        demo_intelligent_backend_selection(manager)
        
        # Execute circuits
        demo_circuit_execution(manager)
        
        # Batch execution
        demo_batch_execution(manager)
        
        # Cost estimation
        demo_cost_estimation(manager)
        
        # Show statistics
        demo_statistics(manager)
        
        print("\n" + "=" * 80)
        print("  DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n[OK] Multi-provider quantum computing infrastructure ready")
        print("[OK] Intelligent backend selection working")
        print("[OK] Cost optimization enabled")
        print("[OK] Batch execution functional")
        print("\n")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

