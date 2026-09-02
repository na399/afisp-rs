from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from afisp_rs import WorstSubsetFinder


@dataclass(frozen=True)
class ReferenceWorstSubsetResult:
    risk_estimates: np.ndarray
    confidence_intervals: np.ndarray
    masks: np.ndarray


def reference_worst_subset(
    losses: np.ndarray,
    mu_hat: np.ndarray,
    fold_ids: np.ndarray,
    subset_fractions: tuple[float, ...],
    *,
    eps: float,
    random_state: int,
) -> ReferenceWorstSubsetResult:
    losses = np.asarray(losses, dtype=np.float64).reshape(-1)
    mu_hat = np.asarray(mu_hat, dtype=np.float64).reshape(-1)
    fold_ids = np.asarray(fold_ids, dtype=np.int64).reshape(-1)
    n = len(losses)
    thresholds = np.zeros((len(subset_fractions), n), dtype=np.float64)
    for fold in np.unique(fold_ids):
        indices = np.flatnonzero(fold_ids == fold)
        local = np.sort(mu_hat[indices])
        for fraction_index, fraction in enumerate(subset_fractions):
            thresholds[fraction_index, indices] = _quantile_sorted(local, 1 - fraction)

    shifted = mu_hat + _splitmix64_uniforms(n, random_state) * eps
    risks: list[float] = []
    intervals: list[tuple[float, float]] = []
    masks: list[np.ndarray] = []
    for fraction_index, fraction in enumerate(subset_fractions):
        threshold = thresholds[fraction_index]
        mask = shifted >= threshold
        psi = np.maximum(shifted - threshold, 0.0) / fraction + threshold
        psi = psi + mask * (losses - mu_hat) / fraction
        risk = float(np.mean(psi))
        sigma = float(np.sqrt(np.mean(np.square(psi - risk))))
        margin = 1.96 * sigma / np.sqrt(n)
        risks.append(risk)
        intervals.append((risk - margin, risk + margin))
        masks.append(mask)
    return ReferenceWorstSubsetResult(
        risk_estimates=np.asarray(risks, dtype=np.float64),
        confidence_intervals=np.asarray(intervals, dtype=np.float64),
        masks=np.asarray(masks, dtype=bool),
    )


def _quantile_sorted(values: np.ndarray, quantile: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = min(1.0, max(0.0, quantile)) * (len(values) - 1)
    low = int(np.floor(position))
    high = int(np.ceil(position))
    if low == high:
        return float(values[low])
    weight = position - low
    return float(values[low] * (1 - weight) + values[high] * weight)


def _splitmix64_uniforms(length: int, seed: int) -> np.ndarray:
    mask = (1 << 64) - 1
    state = int(seed) & mask
    scale = 1.0 / float(1 << 53)
    values = np.empty(length, dtype=np.float64)
    for index in range(length):
        state = (state + 0x9E3779B97F4A7C15) & mask
        current = state
        current = ((current ^ (current >> 30)) * 0xBF58476D1CE4E5B9) & mask
        current = ((current ^ (current >> 27)) * 0x94D049BB133111EB) & mask
        current = (current ^ (current >> 31)) & mask
        values[index] = float(current >> 11) * scale
    return values


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_native_matches_independent_reference(seed: int):
    rng = np.random.default_rng(seed)
    n = 256
    losses = rng.random(n)
    mu_hat = losses + 0.05 * rng.standard_normal(n)
    fold_ids = np.repeat(np.arange(8), n // 8)
    if fold_ids.shape[0] < n:
        fold_ids = np.concatenate([fold_ids, np.full(n - fold_ids.shape[0], 7)])
    fractions = (0.1, 0.2, 0.4, 1.0)
    eps = 1e-8

    reference = reference_worst_subset(
        losses,
        mu_hat,
        fold_ids,
        fractions,
        eps=eps,
        random_state=seed,
    )
    finder = WorstSubsetFinder(
        subset_fractions=list(fractions),
        eps=eps,
        random_state=seed,
    )
    finder.fit_from_mu_hat(losses, mu_hat, fold_ids=fold_ids)
    native_risks = np.asarray(finder.r_hats_, dtype=np.float64)
    native_cis = np.asarray(finder.confidence_intervals(), dtype=np.float64)
    native_masks = np.asarray(finder.subset_masks(), dtype=bool)

    np.testing.assert_allclose(native_risks, reference.risk_estimates, rtol=0, atol=1e-10)
    np.testing.assert_allclose(native_cis, reference.confidence_intervals, rtol=0, atol=1e-10)
    assert np.array_equal(native_masks, reference.masks)
