"""
GPU Y Pipeline: Zero-Shot Classification + Target Preparation (RAPIDS-aware)

Generates y_target.parquet on GPU.
Requires GPU X pipeline to have run first.

Usage:
  python -m dgxb_gpu.run_Y_pipeline
Or:
  python dgxb_gpu/run_Y_pipeline.py

Notes:
- Calls the GPU implementation in dgxb_gpu/etl/.
- Output is written to ./gold-gpu-traffic/y_target.parquet
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path so we can import dgxb_gpu
sys.path.insert(0, str(Path(__file__).parent.parent))

from dgxb_gpu.etl.target_engineering_gpu import (
    prepare_y_target_regression_gpu,
)  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    y_target = prepare_y_target_regression_gpu(
        sector_hour_base_path="gold-gpu-traffic/sector_hour_base.parquet",
    )

    print("\n✓ GPU Y Pipeline complete!")
    print(f"  y (target): {y_target.shape[0]:,} records")
    print("  Check ./gold-gpu-traffic/y_target.parquet")
