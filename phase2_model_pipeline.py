"""
Phase 2: Feature Engineering & Model Training Pipeline
=======================================================
Clean, modular pipeline that preprocesses the synthetic dataset,
trains a Gradient Boosting classifier, and reports evaluation metrics
plus top feature importances.

Run after phase1_data_generation.py has produced ab_test_dataset.csv.
"""

"""
Columns required by Phase 2:
Categorical: traffic_source, device_type, target_page
Numerical: baseline_conversion_rate, sample_size, runtime_days, early_lift_observed, early_p_value, statistical_power
Target: is_successful
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for file output
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────────────

def load_data(path: str = "ab_test_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────────────────────────────
# 2. Define feature sets
# ──────────────────────────────────────────────────────────────────────

CATEGORICAL_FEATURES = ["traffic_source", "device_type", "target_page"]

NUMERICAL_FEATURES = [
    "baseline_conversion_rate",
    "sample_size",
    "runtime_days",
    "early_lift_observed",
    "early_p_value",
    "statistical_power",          # engineered in Phase 3
]

TARGET = "is_successful"


# ──────────────────────────────────────────────────────────────────────
# 3. Build the sklearn Pipeline
# ──────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Returns a full sklearn Pipeline:
      ColumnTransformer  →  GradientBoostingClassifier

    Categorical columns are one-hot encoded (drop='first' avoids
    multicollinearity). Numerical columns are standard-scaled so that
    feature importances are on a comparable footing.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


# ──────────────────────────────────────────────────────────────────────
# 4. Evaluate and report metrics
# ──────────────────────────────────────────────────────────────────────

def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("  Model Evaluation — Hold-out Test Set")
    print("=" * 60)
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Failed", "Success"]))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Failed", "Success"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix — A/B Test Outcome Predictor", fontsize=11)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("  Saved → confusion_matrix.png")

    return {"accuracy": acc, "roc_auc": roc_auc}


# ──────────────────────────────────────────────────────────────────────
# 5. Feature importance
# ──────────────────────────────────────────────────────────────────────

def report_feature_importance(pipeline: Pipeline, top_n: int = 5) -> None:
    """
    Extracts feature importances from the fitted GradientBoostingClassifier
    and maps them back to human-readable feature names (including the
    one-hot encoded dummies).
    """
    ohe        = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names  = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_names  = cat_names + NUMERICAL_FEATURES

    importances = pipeline.named_steps["classifier"].feature_importances_
    feat_imp    = (
        pd.Series(importances, index=all_names)
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print(f"  Top {top_n} Most Important Features")
    print("=" * 60)
    for rank, (feat, imp) in enumerate(feat_imp.head(top_n).items(), start=1):
        bar = "█" * int(imp * 200)
        print(f"  {rank}. {feat:<38s}  {imp:.4f}  {bar}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    feat_imp.head(top_n).sort_values().plot(
        kind="barh", ax=ax, color="#4C72B0", edgecolor="white"
    )
    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_title(f"Top {top_n} Predictors of A/B Test Success")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("  Saved → feature_importance.png")


# ──────────────────────────────────────────────────────────────────────
# 6. Cross-validation
# ──────────────────────────────────────────────────────────────────────

def cross_validate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print("\n" + "=" * 60)
    print("  5-Fold Stratified Cross-Validation (ROC-AUC)")
    print("=" * 60)
    print(f"  Fold scores : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean ± Std  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    # Load
    df = load_data("ab_test_dataset.csv")

    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    print(f"[split] Class balance: {y.value_counts().to_dict()}")

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[split] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Build & train
    pipeline = build_pipeline()
    print("\n[train] Fitting Gradient Boosting pipeline …")
    pipeline.fit(X_train, y_train)
    print("[train] Done.")

    # Evaluate on hold-out set
    evaluate(pipeline, X_test, y_test)

    # Feature importance
    report_feature_importance(pipeline, top_n=5)

    # Cross-validation (uses full X, y)
    cross_validate(pipeline, X, y)


if __name__ == "__main__":
    main()
