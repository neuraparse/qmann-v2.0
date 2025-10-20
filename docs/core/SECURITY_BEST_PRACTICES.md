# 🔐 QMANN v2.0 - Security Best Practices

**Version**: 2.0.0  
**Last Updated**: 2025-10-20  
**Status**: ✅ Complete

---

## 📋 Table of Contents

1. [Installation Security](#installation-security)
2. [API Security](#api-security)
3. [Data Security](#data-security)
4. [Model Security](#model-security)
5. [Infrastructure Security](#infrastructure-security)
6. [Dependency Management](#dependency-management)
7. [Incident Response](#incident-response)

---

## 🔒 Installation Security

### 1. Verify Package Integrity

```bash
# Check package signature
pip install qmann --require-hashes

# Verify from PyPI
pip install qmann==2.0.0 --require-hashes

# Check package hash
sha256sum qmann-2.0.0-py3-none-any.whl
```

### 2. Use Virtual Environment

```bash
# Create isolated environment
python3.10 -m venv qmann_env
source qmann_env/bin/activate

# Install dependencies
pip install -e ".[dev,quantum,gpu]"

# Verify installation
python -c "import qmann; print(qmann.__version__)"
```

### 3. Dependency Audit

```bash
# Check for known vulnerabilities
pip install safety
safety check

# Check outdated packages
pip list --outdated

# Generate requirements with hashes
pip freeze --all > requirements.txt
```

---

## 🔐 API Security

### 1. Authentication & Authorization

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthCredentials
import jwt

app = FastAPI()
security = HTTPBearer()

SECRET_KEY = "your-secret-key"  # Use environment variable
ALGORITHM = "HS256"

async def verify_token(credentials: HTTPAuthCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401)
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401)
    return user_id

@app.post("/predict")
async def predict(data: dict, user_id: str = Depends(verify_token)):
    # Only authenticated users can access
    return model.predict(data)
```

### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request, data: dict):
    return model.predict(data)
```

### 3. Input Validation

```python
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    features: list
    model_version: str = "2.0.0"
    
    @validator('features')
    def validate_features(cls, v):
        if not isinstance(v, list):
            raise ValueError('features must be a list')
        if len(v) != 10:
            raise ValueError('features must have 10 elements')
        return v

@app.post("/predict")
async def predict(request: PredictionRequest):
    return model.predict(request.features)
```

### 4. HTTPS/TLS

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Run with HTTPS
uvicorn main:app --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

---

## 🔐 Data Security

### 1. Encryption at Rest

```python
from cryptography.fernet import Fernet
import os

# Generate key
key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt data
plaintext = b"sensitive data"
encrypted = cipher.encrypt(plaintext)

# Decrypt data
decrypted = cipher.decrypt(encrypted)

# Save key securely
with open('.env', 'w') as f:
    f.write(f"ENCRYPTION_KEY={key.decode()}")
```

### 2. Encryption in Transit

```python
import ssl
import requests

# Create SSL context
context = ssl.create_default_context()
context.check_hostname = True
context.verify_mode = ssl.CERT_REQUIRED

# Make secure request
response = requests.get(
    'https://api.qmann.dev/predict',
    verify=context
)
```

### 3. Data Anonymization

```python
import hashlib
import pandas as pd

def anonymize_data(df):
    """Anonymize sensitive columns"""
    sensitive_cols = ['patient_id', 'email', 'phone']
    
    for col in sensitive_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: hashlib.sha256(str(x).encode()).hexdigest()
            )
    return df

# Usage
df = pd.read_csv('patient_data.csv')
df_anon = anonymize_data(df)
```

### 4. Secure Logging

```python
import logging
import logging.handlers

# Don't log sensitive data
logger = logging.getLogger("qmann")

# Use secure handler
handler = logging.handlers.RotatingFileHandler(
    'qmann.log',
    maxBytes=10485760,  # 10MB
    backupCount=5
)

# Mask sensitive data
class SensitiveDataFilter(logging.Filter):
    def filter(self, record):
        record.msg = record.msg.replace(
            'password=', 'password=***'
        )
        return True

handler.addFilter(SensitiveDataFilter())
logger.addHandler(handler)
```

---

## 🔐 Model Security

### 1. Model Signing

```python
import hashlib
import json

def sign_model(model_path, private_key_path):
    """Sign model with private key"""
    with open(model_path, 'rb') as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Sign hash with private key
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    
    with open(private_key_path, 'rb') as f:
        private_key = f.read()
    
    signature = private_key.sign(
        model_hash.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    
    return signature

def verify_model(model_path, signature_path, public_key_path):
    """Verify model signature"""
    with open(model_path, 'rb') as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()
    
    with open(signature_path, 'rb') as f:
        signature = f.read()
    
    with open(public_key_path, 'rb') as f:
        public_key = f.read()
    
    try:
        public_key.verify(
            signature,
            model_hash.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except:
        return False
```

### 2. Model Versioning

```python
import json
from datetime import datetime

def save_model_metadata(model_path, version, checksum):
    """Save model metadata"""
    metadata = {
        "version": version,
        "checksum": checksum,
        "timestamp": datetime.now().isoformat(),
        "framework": "qmann",
        "python_version": "3.10"
    }
    
    with open(f"{model_path}.metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def verify_model_version(model_path):
    """Verify model version"""
    with open(f"{model_path}.metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return metadata['version']
```

### 3. Model Adversarial Robustness

```python
import torch
from torch.autograd import Variable

def generate_adversarial_examples(model, data, epsilon=0.03):
    """Generate adversarial examples using FGSM"""
    data = Variable(data, requires_grad=True)
    output = model(data)
    loss = torch.nn.functional.cross_entropy(output, target)
    
    model.zero_grad()
    loss.backward()
    
    # Generate adversarial example
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    
    return perturbed_data

def test_robustness(model, test_data, epsilon=0.03):
    """Test model robustness"""
    adversarial_data = generate_adversarial_examples(model, test_data, epsilon)
    
    with torch.no_grad():
        original_output = model(test_data)
        adversarial_output = model(adversarial_data)
    
    robustness_score = torch.nn.functional.cosine_similarity(
        original_output, adversarial_output
    ).mean()
    
    return robustness_score
```

---

## 🏗️ Infrastructure Security

### 1. Docker Security

```dockerfile
# Use minimal base image
FROM python:3.10-slim

# Run as non-root user
RUN useradd -m -u 1000 qmann
USER qmann

# Don't run as root
RUN chmod -R 755 /app

# Scan for vulnerabilities
# docker scan qmann:2.0.0
```

### 2. Kubernetes Security

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qmann-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsReadOnlyRootFilesystem: true
  
  containers:
  - name: qmann
    image: qmann:2.0.0
    securityContext:
      allowPrivilegeEscalation: false
      capabilities:
        drop:
        - ALL
    resources:
      limits:
        memory: "4Gi"
        cpu: "2"
```

### 3. Network Security

```bash
# Use firewall rules
sudo ufw allow 8000/tcp
sudo ufw allow 443/tcp
sudo ufw deny 22/tcp

# Use VPN for remote access
# Configure SSH key-based authentication
ssh-keygen -t ed25519 -f ~/.ssh/qmann_key
```

---

## 📦 Dependency Management

### 1. Dependency Scanning

```bash
# Check for vulnerabilities
pip install bandit
bandit -r src/

# Check dependencies
pip install pip-audit
pip-audit

# Generate SBOM
pip install cyclonedx-bom
cyclonedx-bom -o sbom.xml
```

### 2. Lock Dependencies

```bash
# Generate lock file
pip freeze > requirements.lock

# Install from lock file
pip install -r requirements.lock

# Use pip-tools
pip install pip-tools
pip-compile requirements.in
```

---

## 🚨 Incident Response

### 1. Security Incident Checklist

- [ ] Identify the incident
- [ ] Isolate affected systems
- [ ] Preserve evidence
- [ ] Notify stakeholders
- [ ] Investigate root cause
- [ ] Implement fix
- [ ] Deploy patch
- [ ] Monitor for recurrence
- [ ] Document lessons learned

### 2. Vulnerability Disclosure

```
Email: security@qmann.dev
Subject: [SECURITY] Vulnerability Report

Include:
- Vulnerability description
- Affected versions
- Proof of concept
- Suggested fix
- Your contact information
```

---

## 📞 Security Resources

- **OWASP Top 10**: https://owasp.org/www-project-top-ten/
- **CWE/SANS Top 25**: https://cwe.mitre.org/top25/
- **Security Headers**: https://securityheaders.com/
- **SSL Labs**: https://www.ssllabs.com/

---

**Status**: ✅ Complete  
**Last Updated**: 2025-10-20


