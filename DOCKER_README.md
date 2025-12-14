# GPU Base Image

GPU-accelerated ML base image optimized for NVIDIA DGX / Blackwell / Hopper / Ampere GPUs.

## Included

| Package | Version | Description |
|---------|---------|-------------|
| RAPIDS cuDF | 25.02 | GPU DataFrames |
| RAPIDS cuML | 25.02 | GPU ML algorithms |
| CuPy | Latest | GPU NumPy |
| XGBoost | 2.1+ | GPU gradient boosting |
| scikit-learn | 1.3+ | ML utilities |
| PyArrow | 14+ | Fast I/O |
| CUDA | 12.8 | GPU runtime |

## Quick Start

```bash
# Pull image
docker pull yourusername/gpu-base:latest

# Run interactive
docker run -it --gpus all yourusername/gpu-base

# Run your script
docker run --gpus all -v $(pwd):/workspace yourusername/gpu-base python train.py
```

## Build Locally

```bash
docker build -t gpu-base .
```

## GPU Support

Tested on:
- NVIDIA DGX Spark (Blackwell GB10)
- NVIDIA H100 (Hopper)
- NVIDIA A100 (Ampere)
- NVIDIA RTX 4090/3090

## Extend This Image

```dockerfile
FROM yourusername/gpu-base:latest

# Add your dependencies
RUN pip install transformers torch

# Copy your code
COPY . /workspace

CMD ["python", "train.py"]
```

## Example Usage

```python
import cudf
import xgboost as xgb

# GPU DataFrame
df = cudf.read_parquet('data.parquet')

# GPU XGBoost
dtrain = xgb.DMatrix(df[features], label=df[target])
model = xgb.train({'tree_method': 'hist', 'device': 'cuda'}, dtrain)
```
