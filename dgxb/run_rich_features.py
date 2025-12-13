#!/usr/bin/env python3
"""
Rich Feature Engineering Pipeline
Enriches base X features with lags, rolling stats, spatial aggregates, and extended temporal features
Usage: python -m dgxb.run_rich_features
Or: python dgxb/run_rich_features.py
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path so we can import dgxb
sys.path.insert(0, str(Path(__file__).parent.parent))

from dgxb.etl.rich_feature_engineering import enrich_X_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    X_enriched = enrich_X_features(
        lag_windows=[1, 3, 6, 12, 24],  # hours
        rolling_windows=[3, 6, 12, 24],  # hours
        spatial_radius_km=5.0,
        time_windows=[1, 24],  # hours for spatial aggregates
    )

    print("\n✓ Rich feature engineering complete!")
    print(
        f"  Enriched X: {X_enriched.shape[0]:,} records, {X_enriched.shape[1]} features"
    )
    print("  Check ./rich-gold-cpu-traffic/ directory for results.")
