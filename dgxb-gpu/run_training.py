#!/usr/bin/env python3
"""
Run GPU training competition pipeline (RAPIDS / cuML)
Usage:
  python -m dgxb_gpu.run_training
Or:
  python dgxb_gpu/run_training.py

Notes:
- Expects GPU-ready feature parquet paths (e.g., gold-gpu-traffic/, rich-gold-gpu-traffic/)
- Outputs results to ./results/gpu_training_results.csv (from the GPU pipeline module)
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path so we can import dgxb_gpu
sys.path.insert(0, str(Path(__file__).parent.parent))

from dgxb_gpu.training.pipeline_gpu import run_training_competition_gpu  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    results_df = run_training_competition_gpu(
        base_x_path="gold-gpu-traffic/X_features.parquet",
        rich_x_path="rich-gold-gpu-traffic/X_features.parquet",
        y_path="gold-gpu-traffic/y_target.parquet",
        merged_intermediate_path="gold-gpu-traffic/merged_intermediate.parquet",
        results_dir="results",
    )

    if len(results_df) > 0:
        print("\n" + "=" * 70)
        print("GPU TRAINING COMPETITION RESULTS")
        print("=" * 70)

        # Display champion for base channel
        base_champion = results_df[
            (results_df["champion"]) & (results_df["channel"] == "base")
        ]
        if len(base_champion) > 0:
            champ = base_champion.iloc[0]
            print(f"\nChampion (Base Channel): {champ['model_name']}")
            print(f"  RMSE: {champ['rmse']:.4f}")
            print(f"  MAE: {champ['mae']:.4f}")
            print(f"  R²: {champ['r2']:.4f}")
            print(
                f"  Hotspot Precision@K: {champ.get('hotspot_precision_at_k', 0.0):.4f}"
            )
            print(f"  Hotspot Recall@K: {champ.get('hotspot_recall_at_k', 0.0):.4f}")
            print(
                f"  Staging Utility: {champ.get('staging_utility_coverage_pct', 0.0):.1f}%"
            )
            print(f"  Train Time: {champ['train_time_sec']:.2f}s")
            print(
                f"  Inference Latency (p50): {champ.get('inference_latency_p50_ms', 0.0):.2f}ms"
            )

        # Display champion for rich channel
        rich_champion = results_df[
            (results_df["champion"]) & (results_df["channel"] == "rich")
        ]
        if len(rich_champion) > 0:
            champ = rich_champion.iloc[0]
            print(f"\nChampion (Rich Channel): {champ['model_name']}")
            print(f"  RMSE: {champ['rmse']:.4f}")
            print(f"  MAE: {champ['mae']:.4f}")
            print(f"  R²: {champ['r2']:.4f}")
            print(
                f"  Hotspot Precision@K: {champ.get('hotspot_precision_at_k', 0.0):.4f}"
            )
            print(f"  Hotspot Recall@K: {champ.get('hotspot_recall_at_k', 0.0):.4f}")
            print(
                f"  Staging Utility: {champ.get('staging_utility_coverage_pct', 0.0):.1f}%"
            )
            print(f"  Train Time: {champ['train_time_sec']:.2f}s")
            print(
                f"  Inference Latency (p50): {champ.get('inference_latency_p50_ms', 0.0):.2f}ms"
            )

        print("=" * 70)
        print("\nFull results saved to: ./results/gpu_training_results.csv")
    else:
        print("No models trained successfully. Check logs for errors.")
