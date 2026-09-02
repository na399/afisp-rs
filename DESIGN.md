# Design notes

## Clean-room policy

Because the upstream AFISP repository did not expose a license identifier, this project avoids copying code. The implementation is a fresh design based on:

1. the AFISP paper's algorithmic description;
2. the public API shape of the original package; and
3. independent engineering choices for performance and packaging.

The R script `python/afisp/run_sirus.r` is copied from the upstream repository for optional R SIRUS compatibility when R and the `sirus` package are installed.

## Mapping to AFISP

### Stability analysis

The `WorstSubsetFinder` implements the fold-aware out-of-fold thresholding pattern used by AFISP:

- consume per-sample observed loss
- consume out-of-fold conditional loss estimates `mu_hat`
- for each subset fraction `alpha`, compute fold-local `eta_hat(alpha)`
- compute the worst-case loss estimator via the AFISP influence-function style expression
- expose masks for the worst-performing subset(s)

### Phenotype learning

Two Rust-native backends ship in `afisp_rs`:

1. `backend="sirus"` (default)
2. `backend="conjunction"` (DecisionList fallback in the compatibility layer)

The `afisp` compatibility layer additionally supports R SIRUS when available.

### Statistical filtering

Rules are filtered with the same broad sequence as the paper:

1. generate candidate interpretable phenotypes
2. evaluate subgroup performance with bootstrap AUROC
3. run one-sided tests against a tolerance threshold
4. correct with Holm–Bonferroni
5. filter by Cohen's `d` (default 0.3 in the compatibility layer)

## Fidelity notes

| Area | Original AFISP | Consolidated crate |
|------|----------------|-------------------|
| SIRUS | R `sirus` package via subprocess | Rust-native SIRUS default; R path optional in `afisp` layer |
| DecisionList | SkopeRules | Rust conjunction miner |
| `subset_masks()` | Uses `mu_hat >= eta_hat` without tie noise | Rust uses `mu_hat + U >= eta_hat` (consistent with psi/h_hat) |
| Conditional loss model | In-process EBM | Python cross-fit in `afisp` layer; Rust receives `mu_hat` only |
| SIRUS p0 CV | `cv=True` in original | Not implemented; fixed `p0` or R path without CV |

## Known gaps

- no automatic cross-validation for `p0` in the Rust backend
- no SkopeRules-identical DecisionList backend
- PyTorch surrogate losses remain Python-only in `afisp.utils`
- plotting helpers live in the `afisp` compatibility layer only
