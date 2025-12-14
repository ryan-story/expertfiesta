# GPU Training Base Image
# Optimized for DGX Spark / Blackwell / Hopper / Ampere GPUs
#
# Base image with RAPIDS + XGBoost GPU + PyTorch
# Ready for ML/AI workloads
#
# Build:  docker build -t yourusername/gpu-base .
# Push:   docker push yourusername/gpu-base

ARG RAPIDS_VERSION=25.02
ARG CUDA_VERSION=12.8
ARG PYTHON_VERSION=3.12

FROM nvcr.io/nvidia/rapidsai/base:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-py${PYTHON_VERSION}

LABEL maintainer="Your Name <your@email.com>"
LABEL description="GPU-accelerated ML base image with RAPIDS, XGBoost, PyTorch"
LABEL version="1.0.0"

# Non-interactive
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# ============================================
# Environment
# ============================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# GPU targets: Blackwell (12.1), Hopper (9.0), Ampere (8.0)
ENV TORCH_CUDA_ARCH_LIST="12.1;9.0;8.0"

# ============================================
# System packages
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Python packages
# ============================================
RUN pip install --upgrade pip setuptools wheel

# XGBoost GPU
RUN pip install --no-cache-dir xgboost>=2.1.0 \
    --extra-index-url https://pypi.nvidia.com

# ML essentials
RUN pip install --no-cache-dir \
    scikit-learn>=1.3.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    pyarrow>=14.0.0 \
    joblib>=1.3.0 \
    tqdm>=4.66.0 \
    requests>=2.31.0 \
    matplotlib>=3.7.0

# ============================================
# Verification
# ============================================
RUN python <<'EOF'
print('=' * 50)
print('GPU Base Image - Environment Check')
print('=' * 50)
import sys
print(f'Python: {sys.version.split()[0]}')

try:
    import cudf
    print(f'cuDF: {cudf.__version__}')
except: print('cuDF: N/A')

try:
    import cupy
    print(f'CuPy: {cupy.__version__}')
except: print('CuPy: N/A')

try:
    import cuml
    print(f'cuML: {cuml.__version__}')
except: print('cuML: N/A')

import xgboost
print(f'XGBoost: {xgboost.__version__}')

import sklearn
print(f'scikit-learn: {sklearn.__version__}')

print('=' * 50)
print('Ready!')
print('=' * 50)
EOF

# Default command
CMD ["/bin/bash"]
