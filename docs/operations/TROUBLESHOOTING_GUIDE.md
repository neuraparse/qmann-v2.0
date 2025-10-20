# 🔧 QMANN v2.0 - Troubleshooting Guide

**Version**: 2.0.0  
**Last Updated**: 2025-10-20  
**Status**: ✅ Complete

---

## 📋 Quick Troubleshooting Index

| Issue | Severity | Solution |
|-------|----------|----------|
| Import errors | 🔴 Critical | [Installation Issues](#installation-issues) |
| Out of memory | 🔴 Critical | [Memory Issues](#memory-issues) |
| Quantum backend unavailable | 🟠 High | [Backend Issues](#backend-issues) |
| Slow inference | 🟡 Medium | [Performance Issues](#performance-issues) |
| High error rates | 🟡 Medium | [Error Mitigation](#error-mitigation) |
| Test failures | 🟡 Medium | [Testing Issues](#testing-issues) |

---

## 🔴 Installation Issues

### Problem: ImportError: No module named 'qmann'

**Cause**: Package not installed or not in Python path

**Solution**:
```bash
# Reinstall package
pip install -e .

# Verify installation
python -c "import qmann; print(qmann.__version__)"

# Check Python path
python -c "import sys; print(sys.path)"
```

### Problem: CUDA/GPU not detected

**Cause**: CUDA not installed or PyTorch not compiled with CUDA

**Solution**:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check CUDA version
nvidia-smi
```

### Problem: Quantum library import fails

**Cause**: Qiskit or PennyLane not installed

**Solution**:
```bash
# Install quantum dependencies
pip install -e ".[quantum]"

# Verify installation
python -c "import qiskit; print(qiskit.__version__)"
python -c "import pennylane; print(pennylane.__version__)"
```

---

## 💾 Memory Issues

### Problem: RuntimeError: CUDA out of memory

**Cause**: Model too large for GPU memory

**Solution**:
```python
# Reduce batch size
config.classical.batch_size = 8  # From 32

# Reduce hidden size
config.classical.hidden_size = 128  # From 256

# Enable gradient checkpointing
trainer.enable_gradient_checkpointing()

# Use mixed precision
config.classical.use_mixed_precision = True
```

### Problem: MemoryError: Unable to allocate memory

**Cause**: Insufficient RAM for classical computation

**Solution**:
```python
# Reduce quantum memory size
config.quantum.memory_qubits = 8  # From 16

# Reduce classical memory
config.classical.hidden_size = 64

# Enable memory consolidation
config.hybrid.memory_consolidation_freq = 50

# Use data generators instead of loading all data
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

### Problem: Quantum state vector too large

**Cause**: Too many qubits for state vector simulation

**Solution**:
```python
# Use stabilizer simulator
config.quantum.simulator_name = "stabilizer_simulator"

# Reduce qubit count
config.quantum.max_qubits = 20  # From 127

# Use tensor network simulator
config.quantum.simulator_name = "tensor_network"
```

---

## 🔌 Backend Issues

### Problem: Quantum backend unavailable

**Cause**: IBM Quantum service down or no internet connection

**Solution**:
```python
# Switch to local simulator
config.quantum.use_hardware = False
config.quantum.backend_name = "qasm_simulator"

# Check backend status
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
backends = service.backends()
print([b.name for b in backends])

# Use Qiskit Aer simulator
from qiskit_aer import AerSimulator
backend = AerSimulator()
```

### Problem: Authentication failed for IBM Quantum

**Cause**: Invalid API key or credentials

**Solution**:
```bash
# Set API key
export IBM_QUANTUM_API_KEY="your-api-key"

# Or in Python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="your-api-key")

# Verify connection
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; print(QiskitRuntimeService().backends())"
```

### Problem: Circuit too deep for hardware

**Cause**: Circuit depth exceeds hardware limit

**Solution**:
```python
# Reduce circuit depth
config.quantum.max_circuit_depth = 50  # From 100

# Use circuit optimization
from qiskit.transpiler import passes
pm = passes.PassManager([passes.Optimize1qGates()])
optimized_circuit = pm.run(circuit)

# Use error mitigation to reduce depth
config.quantum.error_mitigation_method = 'zne'
```

---

## ⚡ Performance Issues

### Problem: Slow inference (> 1 second per sample)

**Cause**: Model too large or inefficient computation

**Solution**:
```python
# Use model quantization
from torch.quantization import quantize_dynamic
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Use model pruning
import torch.nn.utils.prune as prune
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# Use batch inference
predictions = model.batch_predict(data, batch_size=64)

# Use ONNX export for faster inference
import torch.onnx
torch.onnx.export(model, dummy_input, "model.onnx")
```

### Problem: Training too slow

**Cause**: Inefficient data loading or computation

**Solution**:
```python
# Use data prefetching
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)

# Use distributed training
python -m torch.distributed.launch --nproc_per_node=4 train.py

# Use gradient accumulation
trainer.gradient_accumulation_steps = 4

# Use mixed precision
config.classical.use_mixed_precision = True
```

### Problem: High latency in quantum execution

**Cause**: Queue times or network latency

**Solution**:
```python
# Use local simulator
config.quantum.use_hardware = False

# Batch quantum circuits
circuits = [circuit1, circuit2, circuit3]
results = backend.run(circuits, shots=8192)

# Use pulse optimization
config.quantum.enable_pulse_optimization = True

# Reduce shots
config.quantum.shots = 1024  # From 8192
```

---

## 🔬 Error Mitigation

### Problem: High error rates (> 10%)

**Cause**: Quantum noise or hardware issues

**Solution**:
```python
# Enable error mitigation
from qmann.utils.error_mitigation import ErrorMitigation

em = ErrorMitigation(method='zne')
mitigated_result = em.mitigate(circuit, backend)

# Use different error mitigation method
em = ErrorMitigation(method='pec')  # Probabilistic error cancellation

# Reduce circuit depth
config.quantum.max_circuit_depth = 30

# Use better calibration
backend.calibrate()
```

### Problem: Fidelity degradation over time

**Cause**: Calibration drift or hardware degradation

**Solution**:
```python
# Recalibrate hardware
backend.calibrate()

# Use dynamic decoupling
config.quantum.enable_dynamic_decoupling = True

# Reduce execution time
config.quantum.shots = 1024  # Faster execution

# Monitor fidelity
fidelity = backend.get_fidelity()
print(f"Current fidelity: {fidelity:.4f}")
```

---

## 🧪 Testing Issues

### Problem: Tests failing with random seed

**Cause**: Non-deterministic behavior

**Solution**:
```python
# Set random seed
import numpy as np
import torch
import random

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Set CUDA seed
torch.cuda.manual_seed_all(42)

# Run tests with seed
pytest tests/ --seed=42
```

### Problem: Test timeout

**Cause**: Test taking too long

**Solution**:
```bash
# Increase timeout
pytest tests/ --timeout=300

# Run only fast tests
pytest tests/ -m "not slow"

# Run specific test
pytest tests/test_specific.py::test_function
```

### Problem: Test database connection fails

**Cause**: Database not running or credentials wrong

**Solution**:
```bash
# Start database
docker-compose up -d postgres

# Check connection
psql -h localhost -U qmann -d qmann_test

# Reset database
python scripts/reset_db.py
```

---

## 📊 Monitoring & Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("qmann")
logger.setLevel(logging.DEBUG)
```

### Profile Code

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
model.train(train_loader, epochs=1)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Monitor Resources

```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Monitor CPU and memory
top

# Monitor disk
df -h

# Monitor network
iftop
```

---

## 📞 Getting Help

### Check Documentation

- [API Documentation](API_DOCUMENTATION.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Performance Benchmarking](PERFORMANCE_BENCHMARKING.md)

### Report Issues

1. Check existing issues: https://github.com/neuraparse/qmann-v2.0/issues
2. Create new issue with:
   - Error message
   - Minimal reproducible example
   - System information (OS, Python version, etc.)
   - Steps to reproduce

### Get Support

- **GitHub Discussions**: https://github.com/neuraparse/qmann-v2.0/discussions
- **Email**: support@qmann.dev
- **Documentation**: https://qmann.readthedocs.io

---

**Status**: ✅ Complete  
**Last Updated**: 2025-10-20


