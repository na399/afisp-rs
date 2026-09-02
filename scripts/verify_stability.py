#!/usr/bin/env python3
"""Generate a stability equivalence report for the native Rust backend."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afisp_rs import WorstSubsetFinder
from tests.test_stability_reference import reference_worst_subset


@dataclass
class StabilityReport:
    status: str
    mask_mismatch_count: int
    max_risk_abs_diff: float
    max_ci_abs_diff: float
    tolerance: float


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stability_cross_validation_report.json"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = 512
    losses = rng.random(n)
    mu_hat = losses + 0.03 * rng.standard_normal(n)
    fold_ids = np.repeat(np.arange(10), n // 10)
    if fold_ids.shape[0] < n:
        fold_ids = np.concatenate([fold_ids, np.full(n - fold_ids.shape[0], 9)])
    fractions = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    eps = 1e-8

    reference = reference_worst_subset(
        losses,
        mu_hat,
        fold_ids,
        fractions,
        eps=eps,
        random_state=args.seed,
    )
    finder = WorstSubsetFinder(
        subset_fractions=list(fractions),
        eps=eps,
        random_state=args.seed,
    )
    finder.fit_from_mu_hat(losses, mu_hat, fold_ids=fold_ids)
    native_risks = np.asarray(finder.r_hats_, dtype=np.float64)
    native_cis = np.asarray(finder.confidence_intervals(), dtype=np.float64)
    native_masks = np.asarray(finder.subset_masks(), dtype=bool)

    mask_mismatch = int(np.sum(native_masks != reference.masks))
    max_risk_diff = float(np.max(np.abs(native_risks - reference.risk_estimates)))
    max_ci_diff = float(np.max(np.abs(native_cis - reference.confidence_intervals)))

    passed = (
        mask_mismatch == 0
        and max_risk_diff <= args.tolerance
        and max_ci_diff <= args.tolerance
    )
    report = StabilityReport(
        status="passed" if passed else "failed",
        mask_mismatch_count=mask_mismatch,
        max_risk_abs_diff=max_risk_diff,
        max_ci_abs_diff=max_ci_diff,
        tolerance=args.tolerance,
    )
    args.output.write_text(json.dumps(asdict(report), indent=2) + "\n")
    print(json.dumps(asdict(report), indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
