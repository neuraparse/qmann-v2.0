# Contributing to QMANN

Thank you for your interest in contributing to QMANN (Quantum Memory-Augmented Neural Networks)! This document provides guidelines for contributing to this cutting-edge quantum machine learning project.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (3.11+ recommended for best performance)
- Git
- IBM Quantum account (for quantum backend access)
- CUDA-compatible GPU (optional, for classical ML acceleration)

### Development Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/qmann-research/qmann.git
   cd qmann
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,quantum]"
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Configure IBM Quantum access**
   ```bash
   # Set your IBM Quantum token
   export QISKIT_IBM_TOKEN="your_token_here"
   ```

## 🧪 Development Workflow

### Code Style and Quality

We maintain high code quality standards:

- **Code Formatting**: Black (line length: 88)
- **Linting**: Flake8 with quantum-specific rules
- **Type Checking**: MyPy with strict mode
- **Testing**: Pytest with >90% coverage requirement

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_quantum/          # Quantum components
pytest tests/test_classical/        # Classical components
pytest tests/test_hybrid/           # Hybrid integration
pytest tests/test_applications/     # Real-world applications

# Run with coverage
pytest --cov=src/qmann --cov-report=html
```

### Code Quality Checks

```bash
# Format code
black src/ tests/ examples/

# Check linting
flake8 src/ tests/ examples/

# Type checking
mypy src/qmann/

# Run all quality checks
make lint
```

## 🔬 Quantum Computing Guidelines

### Quantum Circuit Design

1. **NISQ-First Approach**
   - Design circuits for current quantum hardware limitations
   - Keep circuit depth minimal (< 100 gates for IBM Quantum)
   - Use hardware-efficient ansätze

2. **Error Mitigation**
   - Always implement error mitigation for quantum components
   - Use Mitiq library for standardized techniques
   - Document noise assumptions and mitigation strategies

3. **Quantum Resource Management**
   - Minimize quantum backend usage in tests
   - Use simulators for development and CI/CD
   - Implement proper quantum job queuing

### Example Quantum Component

```python
from qiskit import QuantumCircuit
from qmann.quantum.base import QuantumComponent

class MyQuantumLayer(QuantumComponent):
    """Example quantum component following QMANN conventions."""
    
    def __init__(self, num_qubits: int, num_layers: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
    def build_circuit(self, parameters: np.ndarray) -> QuantumCircuit:
        """Build parameterized quantum circuit."""
        circuit = QuantumCircuit(self.num_qubits)
        
        # Hardware-efficient ansatz
        for layer in range(self.num_layers):
            # Rotation gates
            for qubit in range(self.num_qubits):
                circuit.ry(parameters[layer * self.num_qubits + qubit], qubit)
            
            # Entangling gates
            for qubit in range(self.num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
        
        return circuit
```

## 🤖 Machine Learning Guidelines

### Hybrid Architecture

1. **Quantum-Classical Integration**
   - Maintain clear separation between quantum and classical components
   - Use PyTorch for classical neural networks
   - Implement proper gradient flow through quantum layers

2. **Training Protocols**
   - Support both alternating and simultaneous optimization
   - Implement parameter-shift rule for quantum gradients
   - Use NISQ-aware training strategies

3. **Memory Management**
   - Efficient quantum state management
   - Proper cleanup of quantum resources
   - Memory-conscious classical components

## 📊 Testing Guidelines

### Test Categories

1. **Unit Tests**
   - Test individual components in isolation
   - Mock quantum backends for faster execution
   - Achieve >95% code coverage

2. **Integration Tests**
   - Test quantum-classical integration
   - Use quantum simulators
   - Validate end-to-end workflows

3. **Performance Tests**
   - Benchmark against classical baselines
   - Measure quantum advantage metrics
   - Profile memory and compute usage

4. **Hardware Tests**
   - Test on real quantum devices (limited)
   - Validate error mitigation effectiveness
   - Check NISQ device compatibility

### Example Test

```python
import pytest
import torch
from qmann.hybrid import QuantumLSTM

class TestQuantumLSTM:
    """Test suite for QuantumLSTM component."""
    
    @pytest.fixture
    def model(self):
        return QuantumLSTM(
            input_size=10,
            hidden_size=32,
            quantum_qubits=8,
            use_simulator=True  # Use simulator for tests
        )
    
    def test_forward_pass(self, model):
        """Test forward pass with quantum memory."""
        batch_size, seq_len, input_size = 4, 20, 10
        x = torch.randn(batch_size, seq_len, input_size)
        
        output, hidden = model(x)
        
        assert output.shape == (batch_size, seq_len, 32)
        assert hidden[0].shape == (1, batch_size, 32)
    
    def test_quantum_memory_integration(self, model):
        """Test quantum memory functionality."""
        # Test quantum memory storage and retrieval
        memory_input = torch.randn(1, 1, 10)
        
        # Store in quantum memory
        model.store_quantum_memory(memory_input)
        
        # Retrieve from quantum memory
        retrieved = model.recall_quantum_memory(memory_input)
        
        assert retrieved is not None
        assert retrieved.shape == memory_input.shape
```

## 🌟 Contribution Types

### 1. Bug Fixes
- Fix issues in existing functionality
- Add regression tests
- Update documentation if needed

### 2. New Features
- Implement new quantum algorithms
- Add classical ML components
- Extend hybrid architectures
- Create new applications

### 3. Performance Improvements
- Optimize quantum circuits
- Improve classical components
- Enhance error mitigation
- Reduce memory usage

### 4. Documentation
- API documentation
- Tutorials and examples
- Research papers and theory
- User guides

### 5. Applications
- Healthcare applications
- Industrial use cases
- Autonomous systems
- Financial modeling

## 📝 Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Security considerations are addressed

### 2. PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Breaking change

## Quantum Components
- [ ] Uses quantum simulators for testing
- [ ] Implements proper error mitigation
- [ ] Follows NISQ-aware design principles
- [ ] Includes quantum resource cleanup

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Performance benchmarks included
- [ ] Hardware compatibility verified

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] Examples provided
- [ ] User guide updated

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### 3. Review Process

1. **Automated Checks**
   - CI/CD pipeline runs all tests
   - Code quality checks pass
   - Security scans complete

2. **Peer Review**
   - At least 2 reviewers required
   - Quantum expert review for quantum components
   - ML expert review for classical components

3. **Final Approval**
   - Maintainer approval required
   - All discussions resolved
   - Documentation complete

## 🏆 Recognition

Contributors will be recognized through:

- **Contributors List**: Listed in README.md
- **Release Notes**: Mentioned in significant releases
- **Research Papers**: Co-authorship opportunities
- **Conference Presentations**: Speaking opportunities

## 📞 Getting Help

- **Discord**: Join our quantum ML community
- **GitHub Discussions**: Ask questions and share ideas
- **Email**: qmann@research.org
- **Office Hours**: Weekly virtual meetings (Fridays 3-4 PM UTC)

## 📚 Resources

### Quantum Computing
- [Qiskit Textbook](https://qiskit.org/textbook/)
- [IBM Quantum Network](https://quantum-network.ibm.com/)
- [Quantum Machine Learning Papers](https://arxiv.org/list/quant-ph/recent)

### Machine Learning
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Quantum ML Reviews](https://quantum-journal.org/)

### NISQ Computing
- [NISQ Era Papers](https://arxiv.org/abs/1801.00862)
- [Error Mitigation Techniques](https://mitiq.readthedocs.io/)
- [Variational Quantum Algorithms](https://arxiv.org/abs/2012.09265)

Thank you for contributing to the future of quantum machine learning! 🚀

---

**Last Updated**: October 2025
**Maintainers**: QMANN Research Team
