# Publishing

This repo publishes to **PyPI** (`afisp-rs`) and **crates.io** (`afisp_rs`) from GitHub Actions when you push a version tag.

## One-time setup

### 1. GitHub repository secrets

In GitHub: **Settings → Secrets and variables → Actions → New repository secret**

| Secret | Source |
|--------|--------|
| `PYPI_API_TOKEN` | [pypi.org](https://pypi.org) → Account settings → API tokens → scope to project `afisp-rs` |
| `CARGO_REGISTRY_TOKEN` | [crates.io](https://crates.io) → Account settings → API tokens |

Use a **project-scoped** PyPI token when possible. The workflow reads it as `MATURIN_PYPI_TOKEN` internally.

### 2. Reserve names (first release only)

- PyPI: [pypi.org/project/afisp-rs](https://pypi.org/project/afisp-rs/) — must be free or owned by you
- crates.io: `cargo publish --dry-run` locally to confirm `afisp_rs` is available

### 3. crates.io account requirements

Your crates.io account must have a **verified email address** before `cargo publish` succeeds. If the release job fails with:

> A verified email address is required to publish crates to crates.io

Visit [crates.io/settings/profile](https://crates.io/settings/profile), verify your email, then re-run the failed `publish-crate` job or push a patch tag (e.g. `v0.1.1`).

### 4. First manual publish (optional sanity check)

Before automating, you can dry-run locally:

```bash
# Rust crate (no Python extension; matches CI publish job)
cargo publish --dry-run --locked

# Python wheels (one platform)
maturin build --release --features extension-module
```

## Release workflow

Workflow file: [`.github/workflows/release.yml`](.github/workflows/release.yml)

**Trigger:** push a tag `v*` (e.g. `v0.1.0`)

**Jobs:**

1. **validate** — tag version must match `Cargo.toml` and `pyproject.toml`
2. **publish-crate** — `cargo publish --locked` to crates.io (pure Rust default; no PyO3 link)
3. **sdist + wheels** — maturin builds and uploads to PyPI:
   - Linux: `x86_64`, `aarch64` (manylinux)
   - macOS: universal2
   - Windows: amd64
   - sdist (for platforms without a wheel)

## Cut a release

```bash
# 1. Bump version in BOTH files (keep in sync)
#    Cargo.toml      version = "0.1.0"
#    pyproject.toml  version = "0.1.0"

# 2. Commit and tag
git add Cargo.toml pyproject.toml Cargo.lock
git commit -m "Release 0.1.0"
git tag v0.1.0
git push origin main
git push origin v0.1.0
```

The tag push starts the release workflow. Monitor **Actions → Release**.

## After publish

```bash
# PyPI
pip install afisp-rs
pip install "afisp-rs[original]"

# crates.io (Rust library, disable default features for pure Rust)
cargo add afisp_rs --no-default-features
```

## Notes

- **Users do not need Rust** when a wheel exists for their OS/Python version.
- If no wheel matches, `pip` builds from sdist and requires Rust + Python dev headers.
- Re-pushing the same tag will not overwrite PyPI/crates versions; bump the version for a new release.
- `cargo publish` publishes the Rust library with `default = []`. Python builds enable `extension-module` via maturin (`pyproject.toml` → `[tool.maturin] features`).
