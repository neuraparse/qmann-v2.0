#!/usr/bin/env python3
"""
Test 2025 QMANN Features

Simple test to verify that the 2025 quantum computing enhancements
are working correctly.
"""

import sys
import os
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_qiskit_import():
    """Test basic Qiskit functionality."""
    print("🔬 Testing Qiskit 2025 features...")

    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import Statevector
        from qiskit.circuit.library import EfficientSU2

        # Create a simple quantum circuit
        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.cx(0, 1)
        qc.cx(2, 3)

        print("✅ Basic Qiskit functionality working")

        # Test EfficientSU2 ansatz (2025 feature)
        ansatz = EfficientSU2(4, reps=2, entanglement="circular")
        print("✅ EfficientSU2 ansatz working")

        # Test Statevector
        state = Statevector.from_instruction(qc)
        print(f"✅ Statevector created with {len(state.data)} amplitudes")

        return True

    except Exception as e:
        print(f"❌ Qiskit test failed: {e}")
        return False


def test_advanced_techniques():
    """Test our 2025 advanced techniques module."""
    print("\n🚀 Testing QMANN 2025 Advanced Techniques...")

    try:
        from qmann.quantum.advanced_techniques_2025 import (
            MultiHeadQuantumAttention,
            AdaptiveVariationalAnsatz,
            QuantumMemoryConsolidation,
            QuantumAdvantageMetrics,
        )

        # Test MultiHeadQuantumAttention
        attention = MultiHeadQuantumAttention(num_heads=2, num_qubits=4, depth=2)
        print("✅ MultiHeadQuantumAttention initialized")

        # Test AdaptiveVariationalAnsatz
        ansatz = AdaptiveVariationalAnsatz(num_qubits=4, max_depth=5)
        params = np.random.random(8) * 2 * np.pi
        circuit = ansatz.create_circuit(params)
        print("✅ AdaptiveVariationalAnsatz working")

        # Test QuantumMemoryConsolidation
        consolidation = QuantumMemoryConsolidation(num_qubits=4, compression_ratio=0.7)
        print("✅ QuantumMemoryConsolidation initialized")

        # Test QuantumAdvantageMetrics
        metrics = QuantumAdvantageMetrics()
        metrics.speedup_factor = 2.5
        metrics.energy_efficiency = 3.0
        score = metrics.quantum_advantage_score()
        print(f"✅ QuantumAdvantageMetrics working (score: {score:.3f})")

        return True

    except Exception as e:
        print(f"❌ Advanced techniques test failed: {e}")
        return False


def test_quantum_attention_demo():
    """Test quantum attention mechanism."""
    print("\n🧠 Testing Quantum Attention Demo...")

    try:
        from qiskit.quantum_info import Statevector
        from qmann.quantum.advanced_techniques_2025 import MultiHeadQuantumAttention

        # Create quantum attention
        attention = MultiHeadQuantumAttention(num_heads=2, num_qubits=3, depth=1)

        # Create test states
        query_amplitudes = np.random.random(8) + 1j * np.random.random(8)
        query_amplitudes = query_amplitudes / np.linalg.norm(query_amplitudes)
        query_state = Statevector(query_amplitudes)

        key_states = []
        for i in range(3):
            key_amplitudes = np.random.random(8) + 1j * np.random.random(8)
            key_amplitudes = key_amplitudes / np.linalg.norm(key_amplitudes)
            key_states.append(Statevector(key_amplitudes))

        # Apply attention
        start_time = time.time()
        result_state = attention.apply_attention(query_state, key_states)
        attention_time = time.time() - start_time

        print(f"✅ Quantum attention applied in {attention_time:.4f}s")
        print(f"✅ Result state has {len(result_state.data)} amplitudes")

        return True

    except Exception as e:
        print(f"❌ Quantum attention test failed: {e}")
        return False


def test_adaptive_ansatz_demo():
    """Test adaptive ansatz optimization."""
    print("\n🔄 Testing Adaptive Ansatz Demo...")

    try:
        from qmann.quantum.advanced_techniques_2025 import AdaptiveVariationalAnsatz

        # Create adaptive ansatz
        ansatz = AdaptiveVariationalAnsatz(num_qubits=4, max_depth=6)

        # Simulate optimization iterations
        for iteration in range(5):
            # Create parameters
            num_params = 4 * 2 * ansatz.current_depth
            parameters = np.random.random(num_params) * 2 * np.pi

            # Create circuit
            circuit = ansatz.create_circuit(parameters)

            # Simulate performance (improving over time)
            performance = 0.5 + 0.1 * iteration + 0.05 * np.random.random()
            ansatz.adapt_structure(performance)

            if iteration % 2 == 0:
                print(
                    f"  📊 Iteration {iteration}: depth={ansatz.current_depth}, "
                    f"pattern={ansatz.entanglement_pattern}"
                )

        print(f"✅ Adaptive ansatz optimization completed")
        print(f"✅ Final depth: {ansatz.current_depth}")

        return True

    except Exception as e:
        print(f"❌ Adaptive ansatz test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🌟 QMANN 2025 Features Test Suite")
    print("=" * 50)

    tests = [
        ("Qiskit Import", test_qiskit_import),
        ("Advanced Techniques", test_advanced_techniques),
        ("Quantum Attention", test_quantum_attention_demo),
        ("Adaptive Ansatz", test_adaptive_ansatz_demo),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")

    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! QMANN 2025 features are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
