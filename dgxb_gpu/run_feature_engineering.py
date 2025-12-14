#!/usr/bin/env python3
"""
Combined Pipeline: Run both X and Y pipelines sequentially.

Usage:
  python -m dgxb.run_feature_engineering
  python dgxb/run_feature_engineering.py
  python -m dgxb.run_feature_engineering --h3-resolution 9 --k-ring-size 1 --use-gpu

Notes:
- For faster iteration, run run_X_pipeline.py and run_Y_pipeline.py separately.
"""

import sys
import logging
from pathlib import Path
import argparse
from typing import Any, Tuple

# Add parent directory to path so we can import dgxb_gpu
sys.path.insert(0, str(Path(__file__).parent.parent))

from dgxb_gpu.etl.feature_engineering import merge_and_save_X_features_gpu  # noqa: E402

logger = logging.getLogger("dgxb.run_feature_engineering")


def _safe_shape(obj: Any) -> Tuple[int, int]:
    """Works for pandas/cudf DataFrames."""
    try:
        return int(obj.shape[0]), int(obj.shape[1])
    except Exception:
        return 0, 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run combined X+Y feature engineering."
    )
    parser.add_argument(
        "--h3-resolution",
        type=int,
        default=9,
        help="H3 resolution (e.g., 9 ~ 0.5km cells)",
    )
    parser.add_argument(
        "--k-ring-size",
        type=int,
        default=1,
        help="Neighborhood ring size for H3 expansion",
    )
    parser.add_argument(
        "--max-spatial-km", type=float, default=50.0, help="Max spatial distance (km)"
    )
    parser.add_argument(
        "--max-time-hours",
        type=float,
        default=1.0,
        help="Max temporal distance (hours)",
    )
    parser.add_argument(
        "--spatial-weight", type=float, default=1.0, help="Weight for spatial distance"
    )
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=0.5,
        help="Weight for temporal distance",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Prefer RAPIDS (cuDF/cuML) where supported",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 70)
    logger.info("Running combined X + Y pipeline")
    logger.info("For faster iteration, run separately:")
    logger.info("  python dgxb/run_X_pipeline.py  (fast, no zero-shot)")
    logger.info("  python dgxb/run_Y_pipeline.py  (slow, zero-shot classification)")
    logger.info("=" * 70)

    logger.info("Config:")
    logger.info(f"  h3_resolution={args.h3_resolution}")
    logger.info(f"  k_ring_size={args.k_ring_size}")
    logger.info(f"  max_spatial_km={args.max_spatial_km}")
    logger.info(f"  max_time_hours={args.max_time_hours}")
    logger.info(f"  spatial_weight={args.spatial_weight}")
    logger.info(f"  temporal_weight={args.temporal_weight}")
    logger.info(f"  use_gpu={args.use_gpu}")

    # GPU pipeline: merge and save X features
    X_features = merge_and_save_X_features_gpu(
        h3_resolution=args.h3_resolution,
        k_ring_size=args.k_ring_size,
        max_spatial_km=args.max_spatial_km,
        max_time_hours=args.max_time_hours,
        spatial_weight=args.spatial_weight,
        temporal_weight=args.temporal_weight,
    )

    x_rows, x_cols = _safe_shape(X_features)

    logger.info("✓ GPU X pipeline complete!")
    logger.info(f"  X (features): {x_rows:,} records, {x_cols} features")
    logger.info("  Check ./gold-gpu-traffic/ directory for results.")
    logger.info("  Run Y pipeline separately: python dgxb_gpu/run_Y_pipeline.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
