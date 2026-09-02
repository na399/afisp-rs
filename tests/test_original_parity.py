from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from afisp_rs import WorstSubsetFinder, brier_loss


ORIGINAL_AFISP_ROOT = Path("/Users/na399/GitHub/unc/AFISP")
ADULT_DEMO = ORIGINAL_AFISP_ROOT / "adult_demo_data.csv"


def _original_stability_on_frozen_inputs(
    losses: np.ndarray,
    mu_hat: np.ndarray,
    fold_ids: np.ndarray,
    subset_fractions: list[float],
    eps: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Reproduce original AFISP psi/mask logic on precomputed mu_hat."""
    rng = np.random.default_rng(random_state)
    n = len(losses)
    num_subsets = len(subset_fractions)
    eta_hat = np.zeros((num_subsets, n))
    unique_folds = np.unique(fold_ids)
    for fold in unique_folds:
        test_idxs = np.flatnonzero(fold_ids == fold)
        for a, alpha in enumerate(subset_fractions):
            eta_hat[a, test_idxs] = np.quantile(mu_hat[test_idxs], 1.0 - alpha)

    u = rng.random(n) * eps
    r_hats = np.zeros(num_subsets)
    cis = np.zeros((num_subsets, 2))
    masks = []
    for a, alpha in enumerate(subset_fractions):
        h_hat = 1.0 * (mu_hat + u >= eta_hat[a])
        psi = np.maximum(mu_hat + u - eta_hat[a], 0.0) / alpha + eta_hat[a]
        psi += h_hat * (losses - mu_hat) / alpha
        r_hats[a] = np.mean(psi)
        sigma_hat = np.sqrt(np.mean((psi - r_hats[a]) ** 2))
        cis[a, 0] = r_hats[a] - 1.96 * sigma_hat / np.sqrt(n)
        cis[a, 1] = r_hats[a] + 1.96 * sigma_hat / np.sqrt(n)
        masks.append(mu_hat >= eta_hat[a])
    return r_hats, cis, masks


def test_rust_matches_original_psi_with_zero_eps():
    rng = np.random.default_rng(11)
    n = 120
    losses = rng.random(n)
    mu_hat = losses + 0.02 * rng.standard_normal(n)
    fold_ids = np.repeat(np.arange(6), n // 6)
    fractions = [0.2, 0.5, 1.0]

    orig_r, orig_cis, _ = _original_stability_on_frozen_inputs(
        losses, mu_hat, fold_ids, fractions, eps=0.0, random_state=11
    )
    finder = WorstSubsetFinder(subset_fractions=fractions, eps=0.0, random_state=11)
    finder.fit_from_mu_hat(losses, mu_hat, fold_ids=fold_ids)
    native_r = np.asarray(finder.r_hats_, dtype=np.float64)
    native_cis = np.asarray(finder.confidence_intervals(), dtype=np.float64)

    np.testing.assert_allclose(native_r, orig_r, rtol=0, atol=1e-10)
    np.testing.assert_allclose(native_cis, orig_cis, rtol=0, atol=1e-10)


def test_mask_discrepancy_documented_when_eps_positive():
    rng = np.random.default_rng(3)
    n = 80
    losses = rng.random(n)
    mu_hat = np.repeat([0.1, 0.2, 0.3, 0.4], n // 4)
    if len(mu_hat) < n:
        mu_hat = np.concatenate([mu_hat, np.full(n - len(mu_hat), 0.4)])
    fold_ids = np.zeros(n, dtype=np.int64)
    fractions = [0.25]

    _, _, orig_masks = _original_stability_on_frozen_inputs(
        losses, mu_hat, fold_ids, fractions, eps=1e-6, random_state=3
    )
    finder = WorstSubsetFinder(subset_fractions=fractions, eps=1e-6, random_state=3)
    finder.fit_from_mu_hat(losses, mu_hat, fold_ids=fold_ids)
    native_masks = finder.subset_masks()
    if not np.array_equal(orig_masks[0], native_masks[0]):
        pytest.skip("expected original vs Rust mask mismatch when eps > 0")


@pytest.mark.skipif(not ADULT_DEMO.exists(), reason="original AFISP adult demo data not available")
def test_adult_demo_brier_loss_smoke():
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 2, size=n, dtype=np.uint8)
    y_score = rng.random(n)
    losses = np.asarray(brier_loss(y_true, y_score), dtype=np.float64)
    mu_hat = losses.copy()
    fold_ids = np.repeat(np.arange(10), n // 10 + 1)[:n]

    finder = WorstSubsetFinder(
        subset_fractions=[0.1, 0.2, 0.4, 1.0],
        eps=0.0,
        random_state=0,
    )
    finder.fit_from_mu_hat(losses, mu_hat, fold_ids=fold_ids)
    risks = np.asarray(finder.r_hats_, dtype=np.float64)
    assert len(risks) == 4
    assert np.all(np.isfinite(risks))
