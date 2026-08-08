"""
run_all.py — Master Runner
==========================
Executes all three phases in sequence and produces a final summary.

Usage:
    python run_all.py

Outputs:
    ab_test_dataset.csv       — synthetic dataset
    confusion_matrix.png      — model evaluation plot
    feature_importance.png    — top-5 feature chart
"""

import sys
import os

# ── ensure project root is on the path ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from phase1_data_generation import generate_ab_test_dataset
from phase2_model_pipeline import (
    build_pipeline,
    evaluate,
    report_feature_importance,
    cross_validate,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
)
from phase3_statistical_power import interactive_demo, ablation_study
from sklearn.model_selection import train_test_split

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         A/B Test Outcome Predictor — Full Pipeline           ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # ── Phase 1: Generate data ───────────────────────────────────────
    print("▶  Phase 1 — Synthetic Data Generation")
    print("-" * 60)
    df = generate_ab_test_dataset(n=2000)
    df.to_csv("ab_test_dataset.csv", index=False)
    print(f"   Generated {len(df):,} rows  |  Success rate: {df['is_successful'].mean():.2%}\n")

    # ── Phase 2: Train & evaluate ────────────────────────────────────
    print("▶  Phase 2 — Feature Engineering & Model Training")
    print("-" * 60)
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)
    report_feature_importance(pipeline, top_n=5)
    cross_validate(pipeline, X, y)

    # ── Phase 3: Statistical power ───────────────────────────────────
    print("\n▶  Phase 3 — Statistical Power Module")
    print("-" * 60)
    interactive_demo()
    ablation_study("ab_test_dataset.csv")

    # ── Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✓  Pipeline Complete — Final Summary")
    print("=" * 60)
    print(f"  Dataset        : 2,000 synthetic A/B tests")
    print(f"  Features       : {len(feature_cols)} (3 categorical + {len(NUMERICAL_FEATURES)} numerical)")
    print(f"  Model          : Gradient Boosting Classifier")
    print(f"  Test Accuracy  : {metrics['accuracy']:.2%}")
    print(f"  Test ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n  Artifacts:")
    print(f"    ab_test_dataset.csv   — raw dataset")
    print(f"    confusion_matrix.png  — model evaluation")
    print(f"    feature_importance.png— top-5 features")
    print()


if __name__ == "__main__":
    main()
