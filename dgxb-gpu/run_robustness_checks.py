#!/usr/bin/env python3
"""
Run GPU robustness validation checks for the sector-hour regression pipeline.

RAPIDS refactor:
- Uses GPU robustness checks implemented in dgxb_gpu.validation
- Assumes cuDF / cuML compatible implementations internally
"""

import logging
from dgxb_gpu.validation.robustness_checks_gpu import run_all_robustness_checks_gpu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    print("=" * 80)
    print("GPU ROBUSTNESS VALIDATION CHECKS")
    print("=" * 80)
    print()

    results = run_all_robustness_checks_gpu()

    print()
    print("=" * 80)
    print("✓ GPU robustness checks complete!")
    print("  Results saved to: results/robustness_checks_gpu.json")
    print("=" * 80)
