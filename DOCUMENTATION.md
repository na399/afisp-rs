# AFISP-RS Documentation

Complete user and API reference for the consolidated [AFISP](https://github.com/unc-vba/AFISP) Rust/Python implementation.

---

## Table of contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Package layout](#package-layout)
4. [Workflow](#workflow)
5. [API: `afisp_rs`](#api-afisp_rs)
6. [API: `afisp` compatibility layer](#api-afisp-compatibility-layer)
7. [Loss and utility functions](#loss-and-utility-functions)
8. [Algorithm details](#algorithm-details)
9. [Fidelity vs original AFISP](#fidelity-vs-original-afisp)
10. [Rust library usage](#rust-library-usage)
11. [Development and testing](#development-and-testing)
12. [Publishing](#publishing)

---

## Overview

**AFISP** (Adversarial/Fairness-oriented Interpretable Subgroup Phenotyping) is a two-stage method for auditing machine learning models on structured data:

1. **Stability analysis** — find worst-performing data subsets of varying sizes, conditioned on subgroup-defining features.
2. **Subgroup phenotyping** — extract interpretable rules describing who is in those subsets.

This crate implements both stages in **Rust** (performance + determinism) with **PyO3** Python bindings and an optional sklearn-compatible **`afisp`** wrapper matching the original Python package API.

---

## Installation

### From PyPI (recommended)

```bash
pip install afisp-rs
```

| Extra | Install command | Includes |
|-------|-----------------|----------|
| *(core)* | `pip install afisp-rs` | `numpy`, Rust extension |
| `original` | `pip install "afisp-rs[original]"` | sklearn, interpret (EBM), pandas, statsmodels, matplotlib, tqdm |
| `demo` | `pip install "afisp-rs[demo]"` | `original` + PyTorch |
| `test` | `pip install "afisp-rs[test]"` | `original` + pytest |

**Requirements:** Python ≥ 3.9. Prebuilt wheels are available for common Linux, macOS, and Windows platforms.

### From source

```bash
git clone https://github.com/na399/afisp-rs.git
cd afisp-rs
pip install maturin numpy
maturin develop --release --features extension-module
```

### Rust only (crates.io)

```toml
[dependencies]
afisp_rs = { version = "0.1", default-features = false }
```

---

## Package layout

```
afisp-rs/
├── src/                 # Rust core (stability, phenotype, SIRUS, metrics)
├── python/
│   ├── afisp_rs/        # Ergonomic Python API (Rust-backed)
│   └── afisp/           # sklearn-compatible wrappers (original API shape)
├── tests/               # pytest suite
├── examples/demo.py     # End-to-end example
└── scripts/             # Stability equivalence verifier
```

| Import | Module | When to use |
|--------|--------|-------------|
| `from afisp_rs import …` | Rust-backed API | New code, explicit control, best performance |
| `from afisp import …` | Compatibility layer | Drop-in replacement for original `afisp` sklearn API |

---

## Workflow

```text
                    ┌─────────────────────────────────────┐
                    │  Trained model + test set           │
                    │  (y_true, y_score, per-sample loss) │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │  Stage 1: WorstSubsetFinder         │
                    │  Cross-fit conditional loss μ̂(x)   │
                    │  → worst-case subsets per fraction α│
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │  Select subset (e.g. max effect size) │
                    │  → binary subset_labels               │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │  Stage 2: SubgroupPhenotyper        │
                    │  SIRUS or conjunction rule mining   │
                    │  → interpretable phenotype rules    │
                    └─────────────────────────────────────┘
```

**Inputs you always need:**

| Input | Shape | Description |
|-------|-------|-------------|
| Subgroup features `X` | `(N, F)` | Metadata / covariates defining subgroups (not model features unless intentional) |
| Samplewise losses | `(N,)` | Per-sample loss on the test set (Brier, cross-entropy, surrogate AUROC, etc.) |
| `y_true`, `y_score` | `(N,)` | Labels and predictions (for phenotyping AUROC tests) |

---

## API: `afisp_rs`

### `WorstSubsetFinder`

Identifies worst-performing subsets at multiple size fractions `α`.

#### Constructor

```python
WorstSubsetFinder(
    subset_fractions=None,  # default: [0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0]
    eps=0.0,                # tie-breaking noise scale for discrete μ̂
    random_state=0,         # SplitMix64 seed for tie noise
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `fit_from_mu_hat(losses, mu_hat, fold_ids=None)` | Core fit when conditional loss estimates are precomputed |
| `fit_with_estimator(X, losses, estimator, cv=5, shuffle=False)` | sklearn KFold cross-fit → `μ̂` → stability core |
| `confidence_intervals()` | Analytical 95% CIs per subset fraction `(N, 2)` |
| `subset_masks()` | Boolean masks `(n_fractions, N)` |
| `observed_fractions()` | Actual fraction selected per mask |
| `compute_effect_sizes()` | Cohen's *d* per worst subset vs full dataset |
| `find_max_effect_size()` | `(index, max_d)` |
| `select_largest_below_threshold(metric_values, threshold, smaller_is_worse=True)` | Pick largest subset below a performance threshold |
| `subset_auc_table(y_true, y_score, n_bootstrap=100, confidence=0.95)` | Bootstrap AUROC per subset |

#### Attributes after fit

| Attribute | Description |
|-----------|-------------|
| `r_hats_` | Worst-case average loss per subset fraction |
| `mu_hat_` | Conditional loss estimates used for fit |
| `fold_ids_` | Fold assignment per sample (if cross-fit) |

#### Example

```python
from sklearn.ensemble import RandomForestRegressor
from afisp_rs import WorstSubsetFinder, brier_loss

losses = brier_loss(y_true, y_score)
finder = WorstSubsetFinder(subset_fractions=[0.1, 0.2, 0.4, 1.0], eps=1e-8, random_state=7)
finder.fit_with_estimator(X, losses, RandomForestRegressor(random_state=7), cv=10)

print(finder.r_hats_)
idx, d = finder.find_max_effect_size()
worst_mask = finder.subset_masks()[idx]
```

---

### `SubgroupPhenotyper`

Extracts interpretable rules for samples in vs. out of a worst subset.

#### Constructor

```python
SubgroupPhenotyper(
    backend="sirus",              # "sirus" or "conjunction"
    max_depth=3,
    min_support=50,
    max_candidates=5000,
    include_negations=True,
    max_literal_prevalence=0.95,
    min_lift=1.0,
    effect_threshold=0.4,         # Cohen's d filter (use 0.3 for original AFISP parity)
    bootstrap_samples=100,
    random_state=0,
    # SIRUS-specific
    sirus_num_trees=400,
    sirus_p0=0.025,
    sirus_num_rule=None,
    sirus_mtry=None,
    sirus_min_samples_leaf=5,
    sirus_sample_fraction=1.0,
    sirus_replace=True,
    numeric_quantiles=(0.25, 0.5, 0.75),
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `fit(X, subset_labels, losses, *, y_true, y_score, performance_threshold, feature_names=None)` | Auto-binarize features, mine rules, filter |
| `fit_binary(X_binary, …, literal_names=None)` | Pre-binarized `(N, F)` uint8 matrix |
| `selected_rules()` | List of phenotype rule strings |
| `summary_table()` | DataFrame (if pandas installed) or list of dicts with AUROC CIs, *p*-values, effect sizes |
| `fit_metadata()` | Backend parameters and counts |

#### Example

```python
from afisp_rs import SubgroupPhenotyper

subset_labels = worst_mask.astype(np.uint8)
phen = SubgroupPhenotyper(backend="sirus", sirus_num_trees=300, effect_threshold=0.3)
phen.fit(
    X,
    subset_labels,
    losses,
    y_true=y_true,
    y_score=y_score,
    performance_threshold=0.85,
)
print(phen.summary_table())
```

---

### `prepare_binary_feature_matrix`

Converts mixed-type features to a binary matrix for rule mining:

- Binary/boolean columns → kept as-is
- Numeric columns → threshold bins at quantiles (default 25th, 50th, 75th)
- Categorical columns → one-hot encoding (requires pandas for non-DataFrame categoricals)

```python
from afisp_rs import prepare_binary_feature_matrix

X_bin, names = prepare_binary_feature_matrix(df, numeric_quantiles=(0.25, 0.5, 0.75))
```

---

## API: `afisp` compatibility layer

Mirrors the original [AFISP Python package](https://github.com/unc-vba/AFISP) sklearn API.

```python
from afisp import WorstSubsetFinder, SubgroupPhenotyper
```

Requires `pip install "afisp-rs[original]"` for EBM defaults and plotting.

### `afisp.WorstSubsetFinder`

sklearn `BaseEstimator` wrapper.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `subset_fractions` | `[0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0]` | |
| `conditional_loss_model` | `ExplainableBoostingRegressor` | Any sklearn regressor with `fit`/`predict` |
| `cv` | `10` | KFold splits, `shuffle=False` |
| `eps` | `0.0` | |
| `verbose` | `False` | |

| Method | Maps to |
|--------|---------|
| `fit(X, losses, feature_names=None)` | EBM/estimator cross-fit → `afisp_rs` |
| `confidence_intervals()` | `cis_` array |
| `subset_masks()` | List of 1D boolean masks |
| `check_subset_sizes(plot=True)` | Diagnostic plot vs requested fractions |
| `compute_effect_sizes(plot=False)` | Cohen's *d* per fraction |
| `find_max_effect_size()` | `(index, value)` |

Fit attributes: `R_hats_`, `mu_hat_`, `cis_`, `num_samples_`.

### `afisp.SubgroupPhenotyper`

| Parameter | Default | Notes |
|-----------|---------|-------|
| `method` | `"DecisionList"` | `"SIRUS"` or `"DecisionList"` |
| `depth` | `2` | Max rule depth |
| `p0` | `0.025` | SIRUS stability threshold |
| `rule_max` | `50` | Max SIRUS rules |
| `cv` | `False` | SIRUS p0 CV (falls back to Rust SIRUS) |

**Rust-backed phenotyping** requires keyword args on `fit()`:

```python
phen.fit(
    X, subset_labels, test_loss,
    method="SIRUS",
    y_true=y_true,
    y_score=y_score,
    performance_threshold=0.85,
)
```

| `method` | Backend |
|----------|---------|
| `"SIRUS"` | R `sirus` if installed, else Rust SIRUS |
| `"DecisionList"` | Rust conjunction miner |

`generate_subgroup_table(y_test, test_preds, loss_fn=...)` returns a pandas DataFrame with Phenotype, Performance, N, Lower, Upper columns.

---

## Loss and utility functions

### `afisp_rs` (Rust-backed)

```python
from afisp_rs import cross_entropy_loss, brier_loss, zero_one_loss, roc_auc_score
```

| Function | Inputs | Output |
|----------|--------|--------|
| `cross_entropy_loss(y_true, y_pred)` | uint8 labels, float probs | per-sample CE |
| `brier_loss(y_true, y_pred)` | uint8 labels, float probs | per-sample Brier |
| `zero_one_loss(y_true, y_pred_binary)` | uint8 labels, uint8 preds | per-sample 0/1 |
| `roc_auc_score(y_true, y_score)` | uint8 labels, float scores | scalar AUROC |

### `afisp.utils` (compatibility + surrogates)

```python
from afisp.utils import (
    cross_entropy, brier, zero_one_loss, mse,
    clip_predictions, logit, cohens_d, bootstrap_ci,
    roc_auc_surrogate, torch_roc_auc_surrogate,
)
```

PyTorch surrogates require `pip install "afisp-rs[demo]"`.

---

## Algorithm details

### Stage 1: Stability analysis

For each subset fraction `α` and each cross-validation fold:

1. Compute fold-local quantile threshold `η̂ = Q_{1-α}(μ̂)` on out-of-fold samples.
2. Apply tie noise: `μ̂ + U` where `U ~ Uniform(0, ε)` via deterministic SplitMix64.
3. Subset mask: `h = 1[μ̂ + U ≥ η̂]`.
4. Influence-function estimator:

   ```
   ψ = max(μ̂ + U - η̂, 0) / α + η̂ + h · (loss - μ̂) / α
   ```

5. Risk `R̂_α = mean(ψ)`; CI from normal approximation with `σ̂`.

### Stage 2: Rule mining

**SIRUS backend (`backend="sirus"`):**

1. Build randomized shallow binary tree forest predicting worst-subset membership.
2. Extract path rules from non-root nodes.
3. Keep rules with frequency ≥ `p0` across trees.
4. Score and filter (see below).

**Conjunction backend (`backend="conjunction"`):**

1. Enumerate depth-limited literal conjunctions with bitset pruning.
2. Rank by support and lift.
3. Score and filter.

**Statistical filtering (both backends):**

1. Bootstrap AUROC CI within each rule-defined subgroup.
2. One-sided *z*-test vs `performance_threshold`.
3. Holm–Bonferroni correction (α = 0.05).
4. Cohen's *d* > `effect_threshold`.

---

## Fidelity vs original AFISP

| Component | Match quality | Notes |
|-----------|---------------|-------|
| Stability ψ / CIs | High | Validated to `1e-10` vs independent NumPy reference |
| Fold-stratified quantiles | High | |
| Tie noise (SplitMix64) | High | |
| `subset_masks()` with `eps > 0` | Differs | Original omits tie noise in masks; this crate includes it (consistent with ψ) |
| EBM cross-fit | Via `afisp` layer | Requires `interpret` |
| R SIRUS | Optional | Rust SIRUS default |
| SkopeRules DecisionList | Approximate | Conjunction miner substitute |
| SIRUS `cv=True` for p0 | Not implemented | |

See [DESIGN.md](DESIGN.md) for engineering rationale.

---

## Rust library usage

Public types (no Python required):

```rust
use afisp_rs::{WorstSubsetFinderCore, SubgroupPhenotyperCore};

let mut finder = WorstSubsetFinderCore::new(
    vec![0.1, 0.2, 0.4, 1.0],
    1e-8,
    7,
);
finder.fit_from_mu_hat(losses, mu_hat, Some(fold_ids))?;
let masks = &finder.masks;
let risks = &finder.r_hats;
```

Build with Python extension:

```bash
cargo build --features extension-module
```

---

## Development and testing

```bash
# Rust unit tests (no Python)
cargo test --no-default-features

# Full Python suite
maturin develop --release --features extension-module
pytest -q

# Stability equivalence report
python scripts/verify_stability.py
```

### Test coverage

| Test file | Purpose |
|-----------|---------|
| `tests/test_smoke.py` | End-to-end SIRUS + conjunction |
| `tests/test_stability_reference.py` | Native vs NumPy reference |
| `tests/test_original_parity.py` | Original ψ formula parity |

---

## Publishing

Maintainers: see [PUBLISHING.md](PUBLISHING.md).

| Registry | Package name | Trigger |
|----------|--------------|---------|
| PyPI | `afisp-rs` | Push tag `v*` |
| crates.io | `afisp_rs` | Push tag `v*` |

---

## References

- Original AFISP repository: https://github.com/unc-vba/AFISP
- PyPI: https://pypi.org/project/afisp-rs/
- crates.io: https://crates.io/crates/afisp_rs
- docs.rs: https://docs.rs/afisp_rs
