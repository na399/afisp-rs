#!/usr/bin/env python3
"""Demonstrate both the original-style and Rust-native AFISP APIs."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from afisp_rs import WorstSubsetFinder as RustWorstSubsetFinder
from afisp_rs import SubgroupPhenotyper as RustSubgroupPhenotyper
from afisp_rs import brier_loss

try:
    from afisp import WorstSubsetFinder, SubgroupPhenotyper

    HAS_AFISP = True
except ImportError:
    HAS_AFISP = False


def main():
    rng = np.random.default_rng(42)
    n = 5000
    x = rng.normal(size=(n, 5))
    shift_features = np.column_stack(
        [
            (x[:, 0] > 0.75).astype(np.uint8),
            (x[:, 1] < -0.5).astype(np.uint8),
            (x[:, 2] > 1.0).astype(np.uint8),
        ]
    )

    logit = 0.8 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * x[:, 2]
    prob = 1.0 / (1.0 + np.exp(-logit))
    y_true = rng.binomial(1, prob).astype(np.uint8)
    y_score = prob.copy()

    deflated = (shift_features[:, 0] == 1) & (shift_features[:, 1] == 1)
    y_score[deflated] = 0.5 * y_score[deflated] + 0.15
    losses = np.asarray(brier_loss(y_true, y_score), dtype=np.float64)

    print("=== afisp_rs (Rust-native API) ===")
    finder = RustWorstSubsetFinder(
        subset_fractions=[0.1, 0.2, 0.4, 1.0],
        eps=1e-8,
        random_state=42,
    )
    finder.fit_with_estimator(
        shift_features.astype(np.float64),
        losses,
        RandomForestRegressor(random_state=42),
        cv=5,
    )
    print("Worst-case loss estimates:", finder.r_hats_)

    subset_labels = finder.subset_masks()[0].astype(np.uint8)
    phen = RustSubgroupPhenotyper(
        backend="sirus",
        max_depth=2,
        min_support=50,
        bootstrap_samples=100,
        random_state=42,
        sirus_num_trees=128,
        sirus_p0=0.025,
        sirus_min_samples_leaf=20,
    )
    phen.fit(
        shift_features,
        subset_labels,
        losses,
        y_true=y_true,
        y_score=y_score,
        performance_threshold=0.70,
        feature_names=["x0_high", "x1_low", "x2_high"],
    )
    print(phen.summary_table())

    if HAS_AFISP:
        print("\n=== afisp (sklearn-compatible API) ===")
        compat_finder = WorstSubsetFinder(
            subset_fractions=[0.1, 0.2, 0.4, 1.0],
            eps=1e-8,
            random_state=42,
            conditional_loss_model=RandomForestRegressor(random_state=42),
            cv=5,
        )
        compat_finder.fit(
            shift_features.astype(np.float64),
            losses,
            feature_names=["x0_high", "x1_low", "x2_high"],
        )
        print("Worst-case loss estimates:", compat_finder.R_hats_)


if __name__ == "__main__":
    main()
