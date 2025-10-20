# 🚀 QMANN v2.0 - Production Deployment Guide

**Version**: 2.0.0  
**Last Updated**: 2025-10-20  
**Status**: ✅ Production Ready

---

## 📋 Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Monitoring & Logging](#monitoring--logging)
7. [Troubleshooting](#troubleshooting)

---

## ✅ Pre-Deployment Checklist

Before deploying to production, ensure:

- [ ] All tests passing (32/32 new tests + existing tests)
- [ ] Code coverage > 85%
- [ ] Security scan completed
- [ ] Performance benchmarks acceptable
- [ ] Documentation updated
- [ ] Backup strategy in place
- [ ] Monitoring configured
- [ ] Rollback plan ready

**Verification Command**:
```bash
make test-comprehensive
make test-report-summary
make security-scan
```

---

## 🏠 Local Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/neuraparse/qmann-v2.0.git
cd qmann-v2.0
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Basic installation
pip install -e .

# Development installation
pip install -e ".[dev,quantum,gpu]"

# For documentation
pip install -e ".[docs]"
```

### 4. Verify Installation

```bash
python -c "import qmann; print(qmann.__version__)"
make test-critical
```

---

## 🐳 Docker Deployment

### 1. Build Docker Image

```bash
# Build image
docker build -t qmann:2.0.0 .

# Build with GPU support
docker build -f Dockerfile.gpu -t qmann:2.0.0-gpu .
```

### 2. Run Container

```bash
# CPU version
docker run -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  qmann:2.0.0

# GPU version
docker run -it \
  --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  qmann:2.0.0-gpu
```

### 3. Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services**:
- QMANN API (port 8000)
- PostgreSQL (port 5432)
- Redis (port 6379)
- Prometheus (port 9090)
- Grafana (port 3000)

---

## ☸️ Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl create namespace qmann
kubectl config set-context --current --namespace=qmann
```

### 2. Create ConfigMap

```bash
kubectl create configmap qmann-config \
  --from-file=configs/qmann_config.yaml
```

### 3. Deploy Application

```bash
# Create deployment
kubectl apply -f k8s/deployment.yaml

# Create service
kubectl apply -f k8s/service.yaml

# Create ingress
kubectl apply -f k8s/ingress.yaml
```

### 4. Verify Deployment

```bash
# Check pods
kubectl get pods -n qmann

# Check services
kubectl get svc -n qmann

# Check logs
kubectl logs -f deployment/qmann -n qmann
```

### 5. Scale Deployment

```bash
# Scale to 3 replicas
kubectl scale deployment qmann --replicas=3 -n qmann

# Auto-scale
kubectl autoscale deployment qmann \
  --min=2 --max=10 \
  --cpu-percent=80 -n qmann
```

---

## ☁️ Cloud Deployment

### AWS Deployment

```bash
# Using AWS ECS
aws ecs create-service \
  --cluster qmann-cluster \
  --service-name qmann-service \
  --task-definition qmann:1 \
  --desired-count 3

# Using AWS Lambda (for inference)
aws lambda create-function \
  --function-name qmann-inference \
  --runtime python3.10 \
  --handler lambda_handler.handler \
  --zip-file fileb://lambda.zip
```

### Google Cloud Deployment

```bash
# Deploy to Cloud Run
gcloud run deploy qmann \
  --image gcr.io/PROJECT_ID/qmann:2.0.0 \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2

# Deploy to GKE
gcloud container clusters create qmann-cluster
gcloud container clusters get-credentials qmann-cluster
kubectl apply -f k8s/deployment.yaml
```

### Azure Deployment

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group qmann-rg \
  --name qmann-container \
  --image qmann:2.0.0 \
  --cpu 2 --memory 4

# Deploy to AKS
az aks create --resource-group qmann-rg --name qmann-aks
az aks get-credentials --resource-group qmann-rg --name qmann-aks
kubectl apply -f k8s/deployment.yaml
```

---

## 📊 Monitoring & Logging

### 1. Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'qmann'
    static_configs:
      - targets: ['localhost:8000']
```

### 2. Grafana Dashboards

```bash
# Access Grafana
http://localhost:3000

# Default credentials
username: admin
password: admin

# Import dashboards
- QMANN Performance Dashboard
- Quantum Advantage Tracking
- Error Mitigation Metrics
```

### 3. Logging

```python
import logging
from qmann.core.config import get_default_config

config = get_default_config()
config.log_level = "INFO"

logger = logging.getLogger("qmann")
logger.info("QMANN initialized")
```

### 4. Log Aggregation

```bash
# Using ELK Stack
docker-compose -f docker-compose.elk.yml up

# Using Loki
docker-compose -f docker-compose.loki.yml up
```

---

## 🔧 Troubleshooting

### Issue: Out of Memory

**Solution**:
```python
# Reduce batch size
config.classical.batch_size = 16

# Reduce quantum memory
config.quantum.memory_qubits = 8

# Enable gradient checkpointing
trainer.enable_gradient_checkpointing()
```

### Issue: Slow Inference

**Solution**:
```python
# Use quantization
model = quantize_model(model)

# Use model pruning
model = prune_model(model, sparsity=0.3)

# Use batch inference
predictions = model.batch_predict(data, batch_size=64)
```

### Issue: Quantum Backend Unavailable

**Solution**:
```python
# Switch to simulator
config.quantum.use_hardware = False
config.quantum.backend_name = "qasm_simulator"

# Use local simulator
from qiskit_aer import AerSimulator
backend = AerSimulator()
```

### Issue: High Error Rates

**Solution**:
```python
# Enable error mitigation
from qmann.utils.error_mitigation import ErrorMitigation

em = ErrorMitigation(method='zne')
mitigated_result = em.mitigate(circuit, backend)

# Reduce circuit depth
config.quantum.max_circuit_depth = 50
```

---

## 📈 Performance Optimization

### 1. Hardware Acceleration

```bash
# Enable GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Enable mixed precision
config.classical.use_mixed_precision = True

# Enable distributed training
python -m torch.distributed.launch \
  --nproc_per_node=4 train.py
```

### 2. Quantum Optimization

```python
# Use pulse optimization
config.quantum.enable_pulse_optimization = True

# Reduce two-qubit gates
config.quantum.two_qubit_gate_limit = 30

# Use error mitigation
config.quantum.error_mitigation_method = 'zne'
```

### 3. Memory Optimization

```python
# Enable memory consolidation
config.hybrid.memory_consolidation_freq = 50

# Use gradient accumulation
trainer.gradient_accumulation_steps = 4

# Enable checkpointing
trainer.enable_gradient_checkpointing()
```

---

## 🔐 Security Considerations

### 1. API Security

```python
# Enable authentication
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(data: dict, credentials = Depends(security)):
    # Verify credentials
    return model.predict(data)
```

### 2. Data Security

```bash
# Encrypt sensitive data
openssl enc -aes-256-cbc -in data.json -out data.json.enc

# Use environment variables
export QMANN_API_KEY="your-secret-key"
```

### 3. Model Security

```bash
# Sign model
python scripts/sign_model.py model.pth

# Verify model
python scripts/verify_model.py model.pth.sig
```

---

## 📞 Support & Resources

- **Documentation**: https://qmann.readthedocs.io
- **GitHub Issues**: https://github.com/neuraparse/qmann-v2.0/issues
- **Discussions**: https://github.com/neuraparse/qmann-v2.0/discussions
- **Email**: support@qmann.dev

---

**Status**: ✅ Production Ready  
**Last Updated**: 2025-10-20


