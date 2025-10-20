# 📚 QMANN v2.0 - Complete API Documentation

**Version**: 2.0.0  
**Last Updated**: 2025-10-20  
**Status**: ✅ Complete

---

## 📖 Table of Contents

1. [Core Components](#core-components)
2. [Quantum Components](#quantum-components)
3. [Hybrid Components](#hybrid-components)
4. [Applications](#applications)
5. [Utilities](#utilities)
6. [Configuration](#configuration)

---

## 🔧 Core Components

### QMANNConfig

**Purpose**: Main configuration class for QMANN framework

```python
from qmann.core.config import QMANNConfig, get_default_config

# Get default configuration
config = get_default_config()

# Or create custom configuration
config = QMANNConfig(
    quantum=QuantumConfig(max_qubits=127),
    classical=ClassicalConfig(hidden_size=256),
    hybrid=HybridConfig(quantum_classical_ratio=0.5)
)
```

**Key Attributes**:
- `quantum`: Quantum computing configuration
- `classical`: Classical neural network configuration
- `hybrid`: Hybrid system configuration
- `application`: Application-specific configuration
- `random_seed`: Random seed for reproducibility
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

---

### QMANNBase

**Purpose**: Abstract base class for all QMANN components

```python
from qmann.core.base import QMANNBase

class CustomComponent(QMANNBase):
    def __init__(self, config):
        super().__init__(config, name="CustomComponent")
    
    def forward(self, x):
        # Implementation
        pass
```

**Key Methods**:
- `execute_circuit(circuit, shots)`: Execute quantum circuit
- `get_backend()`: Get quantum backend
- `count_parameters()`: Count trainable parameters
- `to_device(tensor)`: Move tensor to device

---

## ⚛️ Quantum Components

### QMatrix

**Purpose**: Quantum Memory Matrix with O(√N) recall

```python
from qmann.quantum.qmatrix import QMatrix

# Initialize quantum memory
q_memory = QMatrix(
    memory_size=64,
    qubit_count=16,
    encoding_method='amplitude'
)

# Write to quantum memory
content = np.random.randn(10)
q_memory.write(address=0, content=content)

# Read from quantum memory
retrieved = q_memory.read(address=0)

# Search in quantum memory
results = q_memory.search(query=content, top_k=5)
```

**Key Methods**:
- `write(address, content)`: Write to quantum memory
- `read(address)`: Read from quantum memory
- `search(query, top_k)`: Search similar content
- `update(address, content)`: Update memory content
- `clear()`: Clear all memory

**Performance**:
- Write: O(√N)
- Read: O(√N)
- Search: O(√N)
- Memory: O(√N) qubits

---

### QuantumLSTM

**Purpose**: Quantum-enhanced LSTM for sequence modeling

```python
from qmann.hybrid.quantum_lstm import QuantumLSTM

# Initialize Quantum LSTM
qlstm = QuantumLSTM(
    input_size=10,
    hidden_size=64,
    num_layers=2,
    quantum_ratio=0.5
)

# Forward pass
output, (h_n, c_n) = qlstm(input_sequence)

# Get metrics
metrics = qlstm.get_metrics()
print(f"Quantum advantage: {metrics['quantum_advantage']:.2f}×")
print(f"Energy efficiency: {metrics['energy_efficiency']:.2f}×")
```

**Key Methods**:
- `forward(x)`: Forward pass
- `get_metrics()`: Get performance metrics
- `reset_hidden_state()`: Reset hidden state
- `get_quantum_state()`: Get quantum state

---

## 🔗 Hybrid Components

### HybridTrainer

**Purpose**: Train hybrid quantum-classical models

```python
from qmann.hybrid.trainer import HybridTrainer

# Initialize trainer
trainer = HybridTrainer(
    model=qlstm,
    config=config,
    device='cuda'
)

# Train model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    learning_rate=0.001
)

# Evaluate model
metrics = trainer.evaluate(test_loader)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
print(f"Quantum advantage: {metrics['quantum_advantage']:.2f}×")
```

**Key Methods**:
- `train(train_loader, val_loader, epochs, lr)`: Train model
- `evaluate(test_loader)`: Evaluate model
- `predict(x)`: Make predictions
- `save_checkpoint(path)`: Save model
- `load_checkpoint(path)`: Load model

---

## 🏭 Applications

### HealthcarePredictor

**Purpose**: Predict postoperative complications

```python
from qmann.applications.healthcare import HealthcarePredictor

# Initialize predictor
predictor = HealthcarePredictor(config=config)

# Make predictions
predictions = predictor.predict(patient_data)
print(f"Complication risk: {predictions['risk_score']:.4f}")
print(f"Sensitivity: {predictions['sensitivity']:.4f}")
print(f"Specificity: {predictions['specificity']:.4f}")
```

**Key Methods**:
- `predict(patient_data)`: Predict complications
- `get_risk_factors()`: Get important risk factors
- `explain_prediction(patient_data)`: Explain prediction

---

### FinancePredictor

**Purpose**: Portfolio optimization and risk analysis

```python
from qmann.applications.finance import FinancePredictor

# Initialize predictor
predictor = FinancePredictor(config=config)

# Optimize portfolio
portfolio = predictor.optimize_portfolio(
    assets=asset_returns,
    constraints={'max_risk': 0.15}
)

print(f"Sharpe ratio: {portfolio['sharpe_ratio']:.4f}")
print(f"Expected return: {portfolio['expected_return']:.4f}")
```

**Key Methods**:
- `optimize_portfolio(assets, constraints)`: Optimize portfolio
- `calculate_var(returns, confidence)`: Calculate Value at Risk
- `predict_returns(features)`: Predict asset returns

---

### DrugDiscoveryPredictor

**Purpose**: Molecular property prediction

```python
from qmann.applications.drug_discovery import DrugDiscoveryPredictor

# Initialize predictor
predictor = DrugDiscoveryPredictor(config=config)

# Predict molecular properties
properties = predictor.predict_properties(molecular_features)
print(f"Drug-likeness: {properties['drug_likeness']:.4f}")
print(f"Toxicity risk: {properties['toxicity_risk']:.4f}")
```

**Key Methods**:
- `predict_properties(features)`: Predict molecular properties
- `screen_compounds(compounds)`: Screen compound library
- `rank_candidates(compounds)`: Rank candidates

---

## 🛠️ Utilities

### QuantumBackend

**Purpose**: Manage quantum hardware/simulator backends

```python
from qmann.utils.backend import QuantumBackend

# Initialize backend
backend = QuantumBackend(
    backend_name='ibm_quantum',
    use_hardware=False
)

# Execute circuit
result = backend.run(circuit, shots=8192)

# Get backend info
info = backend.get_backend_info()
print(f"Qubits: {info['num_qubits']}")
print(f"Gate fidelity: {info['gate_fidelity']:.4f}")
```

**Key Methods**:
- `run(circuit, shots)`: Execute circuit
- `get_backend_info()`: Get backend information
- `set_backend(name)`: Switch backend

---

### ErrorMitigation

**Purpose**: Mitigate quantum errors

```python
from qmann.utils.error_mitigation import ErrorMitigation

# Initialize error mitigation
em = ErrorMitigation(method='zne')

# Mitigate circuit
mitigated_result = em.mitigate(circuit, backend)

print(f"Fidelity improvement: {em.fidelity_improvement:.2f}×")
```

**Key Methods**:
- `mitigate(circuit, backend)`: Mitigate errors
- `get_fidelity_improvement()`: Get fidelity improvement

---

### Benchmarks

**Purpose**: Benchmark quantum advantage

```python
from qmann.utils.benchmarks import Benchmarks

# Run benchmarks
benchmarks = Benchmarks(config=config)
results = benchmarks.run_all()

print(f"Memory speedup: {results['memory_speedup']:.2f}×")
print(f"Energy efficiency: {results['energy_efficiency']:.2f}×")
print(f"Quantum advantage: {results['quantum_advantage']:.2f}×")
```

**Key Methods**:
- `run_all()`: Run all benchmarks
- `run_memory_benchmark()`: Memory benchmark
- `run_energy_benchmark()`: Energy benchmark

---

## ⚙️ Configuration

### QuantumConfig

```python
@dataclass
class QuantumConfig:
    max_qubits: int = 127
    gate_fidelity: float = 0.999
    measurement_fidelity: float = 0.995
    coherence_time_t1: float = 100e-6
    coherence_time_t2: float = 50e-6
    backend_name: str = "ibm_quantum"
    simulator_name: str = "qasm_simulator"
    use_hardware: bool = False
    shots: int = 8192
```

### ClassicalConfig

```python
@dataclass
class ClassicalConfig:
    input_size: int = 10
    hidden_size: int = 256
    output_size: int = 1
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "auto"
```

### HybridConfig

```python
@dataclass
class HybridConfig:
    quantum_classical_ratio: float = 0.5
    sync_frequency: int = 10
    memory_consolidation_freq: int = 100
    enable_quantum_advantage_tracking: bool = True
```

---

## 📝 Examples

### Basic Usage

```python
from qmann import QuantumLSTM, HybridTrainer
from qmann.core.config import get_default_config

# Get configuration
config = get_default_config()

# Create model
model = QuantumLSTM(
    input_size=10,
    hidden_size=64,
    num_layers=2
)

# Create trainer
trainer = HybridTrainer(model, config)

# Train
history = trainer.train(train_loader, val_loader, epochs=100)
```

### Advanced Usage

```python
# Custom configuration
config = QMANNConfig(
    quantum=QuantumConfig(max_qubits=127),
    classical=ClassicalConfig(hidden_size=512),
    hybrid=HybridConfig(quantum_classical_ratio=0.7)
)

# Create model with custom config
model = QuantumLSTM(
    input_size=20,
    hidden_size=128,
    num_layers=3,
    config=config
)

# Get metrics
metrics = model.get_metrics()
print(f"Quantum advantage: {metrics['quantum_advantage']:.2f}×")
```

---

## 🔗 Related Documentation

- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Performance Benchmarking](PERFORMANCE_BENCHMARKING.md)
- [Troubleshooting](TROUBLESHOOTING_GUIDE.md)
- [Security Best Practices](SECURITY_BEST_PRACTICES.md)

---

**For more information, visit**: https://github.com/neuraparse/qmann-v2.0


