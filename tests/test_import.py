#!/usr/bin/env python3
"""
Simple test script to verify QMANN imports work correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic QMANN imports."""
    print("Testing QMANN imports...")
    
    try:
        # Test core config import
        from qmann.core.config import QMANNConfig
        print("✓ Core config imported successfully")
        
        # Create config instance
        config = QMANNConfig()
        print("✓ Config instance created successfully")
        
        # Test core base import
        from qmann.core.base import QMANNBase
        print("✓ Core base imported successfully")
        
        # Test exceptions
        from qmann.core.exceptions import QMANNError
        print("✓ Core exceptions imported successfully")
        
        print("\n✅ All core imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_optional_imports():
    """Test optional component imports."""
    print("\nTesting optional component imports...")
    
    # Test quantum components (requires qiskit)
    try:
        from qmann.quantum.qmatrix import QMatrix
        print("✓ Quantum QMatrix imported successfully")
    except ImportError as e:
        print(f"⚠️  Quantum components not available: {e}")
    
    # Test classical components (requires torch)
    try:
        from qmann.classical.lstm import ClassicalLSTM
        print("✓ Classical LSTM imported successfully")
    except ImportError as e:
        print(f"⚠️  Classical components not available: {e}")
    
    # Test utilities
    try:
        from qmann.utils.backend import QuantumBackend
        print("✓ Quantum backend utilities imported successfully")
    except ImportError as e:
        print(f"⚠️  Utilities not available: {e}")

def test_config_functionality():
    """Test configuration functionality."""
    print("\nTesting configuration functionality...")
    
    try:
        from qmann.core.config import QMANNConfig
        
        # Test default config
        config = QMANNConfig()
        print(f"✓ Default config created")
        print(f"  - Quantum max qubits: {config.quantum.max_qubits}")
        print(f"  - Classical learning rate: {config.classical.learning_rate}")
        print(f"  - Hybrid quantum ratio: {config.hybrid.quantum_classical_ratio}")
        
        # Test config serialization
        config_dict = config.to_dict()
        print("✓ Config serialization works")
        
        # Test config from dict
        new_config = QMANNConfig.from_dict(config_dict)
        print("✓ Config deserialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def main():
    """Main test function."""
    print("QMANN Import Test")
    print("=" * 50)
    
    # Test basic imports
    basic_success = test_basic_imports()
    
    if basic_success:
        # Test optional imports
        test_optional_imports()
        
        # Test config functionality
        config_success = test_config_functionality()
        
        if config_success:
            print("\n🎉 All tests passed! QMANN is ready to use.")
        else:
            print("\n⚠️  Some functionality tests failed.")
    else:
        print("\n❌ Basic imports failed. Please check your installation.")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
