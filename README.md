# afisp-rs

A clean-room Rust + PyO3 implementation of the [AFISP](https://github.com/unc-vba/AFISP) workflow: stability analysis to find worst-performing data subsets, then interpretable subgroup phenotyping.

**Full documentation:** [DOCUMENTATION.md](DOCUMENTATION.md)

Two Python APIs ship in one package:

| Import | Use when |
|--------|----------|
| `from afisp import …` | You want the **original sklearn-style API** (`fit(X, losses)`, optional EBM, plotting) |
| `from afisp_rs import …` | You want the **Rust-backed API** (`fit_from_mu_hat`, `fit_with_estimator`, explicit backends) |

---

## Install from PyPI

```bash
pip install afisp-rs
```

**Original AFISP compatibility** (EBM cross-fit, pandas, plotting, statsmodels):

```bash
pip install "afisp-rs[original]"
```

**Demo extras** (includes PyTorch surrogate losses):

```bash
pip install "afisp-rs[demo]"
```

Requires **Python ≥ 3.9** and **numpy**. Prebuilt wheels are provided for common Linux, macOS, and Windows platforms. If no wheel matches your platform, pip builds from source and you will need Rust + Python dev headers.

---

## Quick start

### Rust-native API (recommended for new code)

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from afisp_rs import WorstSubsetFinder, SubgroupPhenotyper, brier_loss

y_true = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8)
y_score = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9])
losses = np.asarray(brier_loss(y_true, y_score))

X = np.random.randn(len(losses), 3)

finder = WorstSubsetFinder(subset_fractions=[0.5, 1.0], eps=1e-8, random_state=7)
finder.fit_with_estimator(X, losses, RandomForestRegressor(random_state=7), cv=3)

print("Worst-case risks:", finder.r_hats_)
print("Subset masks:", finder.subset_masks())

subset_labels = finder.subset_masks()[0].astype(np.uint8)
phen = SubgroupPhenotyper(backend="sirus", random_state=7, sirus_num_trees=64)
phen.fit(
    X,
    subset_labels,
    losses,
    y_true=y_true,
    y_score=y_score,
    performance_threshold=0.6,
)
print(phen.summary_table())
```

### Original-style sklearn API

```python
from afisp import WorstSubsetFinder, SubgroupPhenotyper

# Requires: pip install "afisp-rs[original]"
# Uses ExplainableBoostingRegressor by default for conditional-loss cross-fit.

finder = WorstSubsetFinder(subset_fractions=[0.2, 0.4, 1.0], cv=10, eps=1e-5)
finder.fit(X, losses, feature_names=["age", "sex", "score"])
print(finder.R_hats_)
print(finder.confidence_intervals())
idx, effect = finder.find_max_effect_size()
```

For Rust-backed phenotyping through the compatibility layer, pass `y_true`, `y_score`, and `performance_threshold` to `SubgroupPhenotyper.fit()`.

---

## What is implemented

### Stage 1 — Stability analysis

- Fold-aware worst-subset thresholds on out-of-fold conditional loss estimates (`mu_hat`)
- Influence-function style worst-case risk estimator with analytical 95% CIs
- Subset masks, observed fractions, Cohen's *d* effect sizes

### Stage 2 — Subgroup phenotyping

- **`backend="sirus"`** — Rust-native stable rule forest (default)
- **`backend="conjunction"`** — deterministic conjunction miner (DecisionList fallback in `afisp` layer)
- Bootstrap AUROC CIs, Holm–Bonferroni correction, Cohen's *d* filtering

See [DESIGN.md](DESIGN.md) for fidelity notes vs the upstream Python reference.

---

## Rust library (crates.io)

```toml
[dependencies]
afisp_rs = { version = "0.1", default-features = false }
```

Pure Rust usage (no Python):

```rust
use afisp_rs::WorstSubsetFinderCore;

let mut finder = WorstSubsetFinderCore::new(vec![0.2, 1.0], 0.0, 0);
finder.fit_from_mu_hat(losses, mu_hat, Some(fold_ids)).unwrap();
```

Enable the Python extension when building with maturin: `--features extension-module`.

---

## Development (from source)

```bash
git clone https://github.com/na399/afisp-rs.git
cd afisp-rs
python -m pip install maturin pytest numpy scikit-learn
maturin develop --release --features extension-module
pytest -q
cargo test --no-default-features
```

Full example: [examples/demo.py](examples/demo.py)

---

## Publishing

Maintainers: see [PUBLISHING.md](PUBLISHING.md). Releases are automated on `v*` tags to PyPI (`afisp-rs`) and crates.io (`afisp_rs`).

---

## License

Dual-licensed under MIT OR Apache-2.0.
