#!/usr/bin/env python3
"""
Run CPU training competition pipeline
Usage: python -m dgxb.run_training
Or: python dgxb/run_training.py
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path so we can import dgxb
sys.path.insert(0, str(Path(__file__).parent.parent))

from dgxb.training.pipeline import run_training_competition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    results_df = run_training_competition()

    if len(results_df) > 0:
        print("\n" + "=" * 70)
        print("TRAINING COMPETITION RESULTS")
        print("=" * 70)

        # Display champion for base channel
        base_champion = results_df[
            (results_df["champion"]) & (results_df["channel"] == "base")
        ]
        if len(base_champion) > 0:
            champ = base_champion.iloc[0]
            print(f"\nChampion (Base Channel): {champ['model_name']}")
            print(f"  Weighted F1: {champ['weighted_f1']:.4f}")
            print(f"  F1 Score: {champ['f1_score']:.4f}")
            print(f"  Precision: {champ['precision_score']:.4f}")
            print(f"  Recall: {champ['recall_score']:.4f}")
            print(f"  Precision@K: {champ['precision_at_k']:.4f}")
            print(f"  Recall@K: {champ['recall_at_k']:.4f}")
            print(f"  Staging Utility: {champ['staging_utility_coverage_pct']:.1f}%")
            print(f"  Train Time: {champ['train_time_sec']:.2f}s")
            print(
                f"  Inference Latency (p50): {champ['inference_latency_p50_ms']:.2f}ms"
            )

        # Display champion for rich channel
        rich_champion = results_df[
            (results_df["champion"]) & (results_df["channel"] == "rich")
        ]
        if len(rich_champion) > 0:
            champ = rich_champion.iloc[0]
            print(f"\nChampion (Rich Channel): {champ['model_name']}")
            print(f"  Weighted F1: {champ['weighted_f1']:.4f}")
            print(f"  F1 Score: {champ['f1_score']:.4f}")
            print(f"  Precision: {champ['precision_score']:.4f}")
            print(f"  Recall: {champ['recall_score']:.4f}")
            print(f"  Precision@K: {champ['precision_at_k']:.4f}")
            print(f"  Recall@K: {champ['recall_at_k']:.4f}")
            print(f"  Staging Utility: {champ['staging_utility_coverage_pct']:.1f}%")
            print(f"  Train Time: {champ['train_time_sec']:.2f}s")
            print(
                f"  Inference Latency (p50): {champ['inference_latency_p50_ms']:.2f}ms"
            )

        print("=" * 70)
        print("\nFull results saved to: ./results/cpu_training_results.csv")
    else:
        print("No models trained successfully. Check logs for errors.")
