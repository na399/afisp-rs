import numpy as np

from afisp_rs import WorstSubsetFinder, SubgroupPhenotyper, brier_loss


def test_end_to_end_smoke():
    rng = np.random.default_rng(7)
    n = 200
    x = rng.integers(0, 2, size=(n, 6), dtype=np.uint8)
    y_true = rng.integers(0, 2, size=n, dtype=np.uint8)
    y_score = (0.1 + 0.8 * rng.random(n)).astype(np.float64)
    losses = np.asarray(brier_loss(y_true, y_score), dtype=np.float64)

    mu_hat = losses + 0.01 * rng.standard_normal(n)
    fold_ids = np.repeat(np.arange(5), n // 5)
    if fold_ids.shape[0] < n:
        fold_ids = np.concatenate([fold_ids, np.full(n - fold_ids.shape[0], 4)])

    finder = WorstSubsetFinder(subset_fractions=[0.2, 0.4, 1.0], eps=1e-8, random_state=7)
    finder.fit_from_mu_hat(losses, mu_hat, fold_ids=fold_ids)
    masks = finder.subset_masks()
    assert masks.shape == (3, n)

    phen = SubgroupPhenotyper(
        backend="sirus",
        max_depth=2,
        min_support=10,
        bootstrap_samples=20,
        random_state=7,
        sirus_num_trees=32,
        sirus_p0=0.05,
        sirus_min_samples_leaf=5,
    )
    phen.fit(
        x,
        masks[0].astype(np.uint8),
        losses,
        y_true=y_true,
        y_score=y_score,
        performance_threshold=0.55,
    )
    summary = phen.summary_table()
    assert summary is not None


def test_fit_binary_preserves_prepared_literals_and_exposes_literal_metadata():
    n = 160
    x_binary = np.zeros((n, 4), dtype=np.uint8)
    x_binary[:80, 0] = 1
    x_binary[40:120, 1] = 1
    x_binary[80:140, 2] = 1
    x_binary[20:70, 3] = 1

    planted = (x_binary[:, 0] == 1) & (x_binary[:, 1] == 1)
    subset_labels = planted.astype(np.uint8)
    y_true = np.arange(n) % 2
    y_score = np.where(y_true == 1, 0.9, 0.1).astype(np.float64)
    y_score[planted] = np.where(y_true[planted] == 1, 0.2, 0.8)
    samplewise_losses = np.where(planted, 0.9, 0.05).astype(np.float64)

    phen = SubgroupPhenotyper(
        backend="conjunction",
        max_depth=2,
        min_support=20,
        include_negations=False,
        min_lift=1.0,
        effect_threshold=0.0,
        bootstrap_samples=20,
        random_state=11,
    )

    phen.fit_binary(
        x_binary,
        subset_labels,
        samplewise_losses,
        y_true=y_true.astype(np.uint8),
        y_score=y_score,
        performance_threshold=0.95,
        literal_names=[
            "Sepsis",
            "Invasive ventilation",
            "Vasopressor exposure",
            "High lactate",
        ],
    )

    summary = phen.summary_table()
    assert {"literal_indices", "literal_names", "literal_signs", "literal_count"} <= set(summary.columns)
    assert summary["literal_count"].max() >= 2

    multivariate = summary.loc[summary["literal_count"].ge(2)].iloc[0]
    assert multivariate["literal_indices"] == [0, 1]
    assert multivariate["literal_names"] == ["Sepsis", "Invasive ventilation"]
    assert multivariate["literal_signs"] == [True, True]
    assert multivariate["phenotype"] == "Sepsis & Invasive ventilation"
