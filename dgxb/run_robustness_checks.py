#!/usr/bin/env python3
"""
Run robustness validation checks for the sector-hour regression pipeline.
"""

import logging
from dgxb.validation.robustness_checks import run_all_robustness_checks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("=" * 80)
    print("ROBUSTNESS VALIDATION CHECKS")
    print("=" * 80)
    print()
    
    results = run_all_robustness_checks()
    
    print()
    print("=" * 80)
    print("✓ Robustness checks complete!")
    print(f"  Results saved to: results/robustness_checks.json")
    print("=" * 80)

