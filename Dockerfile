# QMANN - Quantum Memory-Augmented Neural Networks
# Production-ready Docker image for 2025 quantum computing stack

# Use Python 3.11 slim image for optimal performance
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r qmann && useradd -r -g qmann qmann

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY docker-requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r docker-requirements.txt

# Development stage
FROM base as development

# Copy source code first (needed for pip install -e)
COPY . .

# Install the package in editable mode with development dependencies
RUN pip install -e ".[dev,quantum]"

# Install additional development tools
RUN pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    plotly \
    dash

# Change ownership to qmann user
RUN chown -R qmann:qmann /app

# Switch to non-root user
USER qmann

# Expose ports for Jupyter and development servers
EXPOSE 8888 8050 8080

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./

# Install the package
RUN pip install .

# Create directories for quantum cache and logs
RUN mkdir -p /app/quantum_cache /app/logs && \
    chown -R qmann:qmann /app

# Switch to non-root user
USER qmann

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import qmann; print('QMANN is healthy')" || exit 1

# Default command for production
CMD ["python", "-m", "qmann.applications.healthcare"]

# Quantum simulation stage (for CI/CD and testing)
FROM base as quantum-sim

# Copy source and tests
COPY src/ ./src/
COPY tests/ ./tests/
COPY pyproject.toml ./

# Install the package with quantum dependencies
RUN pip install -e ".[quantum]" && \
    pip install \
    qiskit-aer \
    qiskit-ibm-runtime \
    cirq \
    pennylane

# Change ownership
RUN chown -R qmann:qmann /app

# Switch to non-root user
USER qmann

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]

# Multi-architecture build support
FROM base as multi-arch

# Install architecture-specific optimizations
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
        pip install torch torchvision; \
    fi

# Copy application
COPY src/ ./src/
COPY pyproject.toml README.md ./

# Install package
RUN pip install .

# Setup user
RUN chown -R qmann:qmann /app
USER qmann

# IBM Quantum optimized stage
FROM base as ibm-quantum

# Install IBM Quantum specific dependencies
RUN pip install \
    qiskit[all] \
    qiskit-ibm-runtime \
    qiskit-optimization \
    qiskit-machine-learning

# Copy application
COPY src/ ./src/
COPY examples/ ./examples/
COPY pyproject.toml ./

# Install package
RUN pip install .

# Create quantum credentials directory
RUN mkdir -p /home/qmann/.qiskit && \
    chown -R qmann:qmann /home/qmann

# Switch to non-root user
USER qmann

# Set IBM Quantum environment
ENV QISKIT_SETTINGS=/home/qmann/.qiskit/settings.conf

# Default to running examples
CMD ["python", "examples/healthcare_demo.py"]

# GPU-enabled stage for classical ML acceleration
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as gpu

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy and install application
WORKDIR /app
COPY docker-requirements.txt pyproject.toml ./
RUN pip install -r docker-requirements.txt

COPY src/ ./src/
RUN pip install .

# Create user
RUN groupadd -r qmann && useradd -r -g qmann qmann
RUN chown -R qmann:qmann /app
USER qmann

# Verify CUDA availability
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Default command
CMD ["python", "-m", "qmann.applications.industrial"]

# Minimal runtime stage
FROM python:3.11-alpine as minimal

# Install minimal system dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev

# Set working directory
WORKDIR /app

# Copy only essential files
COPY src/ ./src/
COPY pyproject.toml ./

# Install minimal dependencies
RUN pip install --no-cache-dir \
    numpy \
    torch \
    qiskit

# Install the package
RUN pip install --no-cache-dir .

# Create user
RUN addgroup -S qmann && adduser -S qmann -G qmann
RUN chown -R qmann:qmann /app
USER qmann

# Minimal command
CMD ["python", "-c", "import qmann; print('QMANN minimal runtime ready')"]
