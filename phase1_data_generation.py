"""
Phase 1: Synthetic A/B Test Dataset Generator
==============================================
Generates 2,000 realistic historical A/B test records.
The target variable `is_successful` has a mathematically
grounded relationship with the features so the model can learn.
"""

"""
Will generate these columns:
Categorical: traffic_source, device_type, target_page
Numerical: baseline_conversion_rate, sample_size, runtime_days, early_lift_observed, early_p_value, statistical_power
Target: is_successful
"""

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(67)
N = 2000


def generate_ab_test_dataset(n: int = N) -> pd.DataFrame:
    """
    Generate a synthetic dataset of historical A/B test outcomes.

    Each row represents one past experiment. The `is_successful` label
    is derived from a probabilistic model that honours real-world
    relationships (power, sample size, effect size), so the signal is
    learnable but not trivially separable.

    Parameters
    ----------
    n : int
        Number of synthetic experiments to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with contextual, design, mid-test, and outcome columns.
    """

    # ------------------------------------------------------------------
    # 1. Contextual / categorical features
    
    traffic_source = np.random.choice(
        ["Organic", "Paid", "Social"],
        size=n,
        p=[0.45, 0.35, 0.20],           # Organic is most common
    )
    device_type = np.random.choice(
        ["Mobile", "Desktop"],
        size=n,
        p=[0.60, 0.40],
    )
    target_page = np.random.choice(
        ["Checkout", "Homepage", "Product"],
        size=n,
        p=[0.30, 0.40, 0.30],
    )
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 2. Design / setup features
    
    baseline_conversion_rate = np.round(
        np.random.uniform(0.02, 0.15, size=n), 4
    )
    sample_size = np.random.randint(5_000, 50_001, size=n)
    runtime_days = np.random.randint(7, 29, size=n)   # 7 – 28 days

    # True (simulated) minimum detectable effect — what the experiment
    # was actually designed to detect.  Bigger = easier to detect.
    true_effect_size = np.random.uniform(0.005, 0.04, size=n)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 3. Statistical power — the core driver of success
    
    # compute a per-row approximation of power for a two-proportion
    # z-test at alpha = 0.05.
    alpha = 0.05
    z_alpha = stats.norm.ppf(1 - alpha)          # ≈ 1.645

    p1 = baseline_conversion_rate
    p2 = p1 + true_effect_size
    p_pool = (p1 + p2) / 2
    # Each arm gets half the total sample
    n_per_arm = sample_size / 2

    # Standard error of the difference under H1
    se = np.sqrt(2 * p_pool * (1 - p_pool) / n_per_arm)
    z_beta = (true_effect_size / se) - z_alpha   # non-centrality
    statistical_power = stats.norm.cdf(z_beta)   # 1 – β
    statistical_power = np.clip(statistical_power, 0.0, 1.0)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 4. Mid-test snapshot (Day 3) — noisy early signals
    
    # Early lift: centred on true effect with noise; sometimes negative
    noise = np.random.normal(0, 0.005, size=n)
    early_lift_observed = np.round(true_effect_size + noise, 5)

    # Early p-value: lower when power is high, but still noisy
    early_p_value = np.clip(
        np.random.beta(
            a=2 * (1 - statistical_power) + 0.5,  # higher power → lower p
            b=5,
            size=n,
        ),
        0.001,
        0.999,
    )
    early_p_value = np.round(early_p_value, 4)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 5. Target variable: is_successful
    
    # Base probability is statistical power; modulated by context.
    success_prob = statistical_power.copy()

    # Traffic source modifier
    ts_map = {"Organic": +0.05, "Paid": 0.00, "Social": -0.04}
    success_prob += np.array([ts_map[t] for t in traffic_source])

    # Device modifier
    dev_map = {"Desktop": +0.04, "Mobile": -0.03}
    success_prob += np.array([dev_map[d] for d in device_type])

    # Page modifier
    pg_map = {"Checkout": -0.05, "Product": +0.02, "Homepage": 0.00}
    success_prob += np.array([pg_map[p] for p in target_page])

    # Runtime
    success_prob += (runtime_days - 14) * 0.002

    # Early signal bonus: if early p-value is already < 0.10, bump success
    success_prob += np.where(early_p_value < 0.10, 0.08, 0.0)

    success_prob = np.clip(success_prob, 0.05, 0.95)

    is_successful = (np.random.uniform(size=n) < success_prob).astype(int)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 6. Assemble DataFrame
    
    df = pd.DataFrame(
        {
            "traffic_source": traffic_source,
            "device_type": device_type,
            "target_page": target_page,
            "baseline_conversion_rate": baseline_conversion_rate,
            "sample_size": sample_size,
            "runtime_days": runtime_days,
            "early_lift_observed": early_lift_observed,
            "early_p_value": early_p_value,
            "statistical_power": np.round(statistical_power, 4),  # engineered feature
            "is_successful": is_successful,
        }
    )

    return df
    # ------------------------------------------------------------------


if __name__ == "__main__":
    df = generate_ab_test_dataset()

    print("=" * 60)
    print("  Synthetic A/B Test Dataset — Summary")
    print("=" * 60)
    print(f"Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Success rate   : {df['is_successful'].mean():.2%}")
    print(f"\nClass balance :\n{df['is_successful'].value_counts()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values: \n{df.isnull().sum()}")

    df.to_csv("ab_test_dataset.csv", index=False)
    print("\nSaved as ab_test_dataset.csv")
