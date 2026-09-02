from __future__ import annotations

from typing import Sequence

import numpy as np

from ._core import (
    WorstSubsetFinder as _RustWorstSubsetFinder,
    SubgroupPhenotyper as _RustSubgroupPhenotyper,
    brier_loss,
    cross_entropy_loss,
    roc_auc_score,
    zero_one_loss,
)

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


__all__ = [
    "WorstSubsetFinder",
    "SubgroupPhenotyper",
    "prepare_binary_feature_matrix",
    "cross_entropy_loss",
    "brier_loss",
    "zero_one_loss",
    "roc_auc_score",
]


def _is_binary_array(x: np.ndarray) -> bool:
    if x.size == 0:
        return True
    unique = np.unique(x[~np.isnan(x)] if np.issubdtype(x.dtype, np.floating) else x)
    return np.all(np.isin(unique, [0, 1]))


def _to_1d_float(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        arr = np.asarray(arr).reshape(-1)
    return np.ascontiguousarray(arr)


def _to_1d_u8(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.uint8)
    if arr.ndim != 1:
        arr = np.asarray(arr).reshape(-1)
    return np.ascontiguousarray(arr)


def _to_2d_numeric(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError("expected a 2D array-like object")
    return np.ascontiguousarray(arr)


def _to_2d_binary_u8(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError("expected a 2D array-like object")
    if not _is_binary_array(arr):
        raise ValueError("fit_binary expects a pre-binarized matrix containing only 0/1 values")
    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def _default_feature_names(n_features: int) -> list[str]:
    return [f"x{i}" for i in range(n_features)]


def prepare_binary_feature_matrix(
    X,
    feature_names: Sequence[str] | None = None,
    numeric_quantiles: Sequence[float] = (0.25, 0.5, 0.75),
):
    """
    Convert mixed feature input into a binary matrix suitable for the Rust
    subgroup-rule backends, including the SIRUS-style stable rule extractor.

    Rules:
    - binary / boolean columns are kept as-is
    - categorical columns are one-hot encoded
    - numeric non-binary columns are threshold-binned using quantiles
    """
    if pd is not None and isinstance(X, pd.DataFrame):
        columns: list[np.ndarray] = []
        names: list[str] = []
        for col_name in X.columns:
            s = X[col_name]
            values = s.to_numpy()
            if pd.api.types.is_bool_dtype(s) or _is_binary_array(np.asarray(values, dtype=np.float64 if np.issubdtype(np.asarray(values).dtype, np.number) else object)):
                columns.append(np.asarray(values, dtype=np.uint8).reshape(-1))
                names.append(str(col_name))
            elif pd.api.types.is_numeric_dtype(s):
                arr = np.asarray(values, dtype=np.float64)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    continue
                thresholds = np.unique(np.quantile(finite, numeric_quantiles))
                min_v = float(np.min(finite))
                max_v = float(np.max(finite))
                for thr in thresholds:
                    thr = float(thr)
                    if thr <= min_v or thr >= max_v:
                        continue
                    columns.append((arr >= thr).astype(np.uint8))
                    names.append(f"{col_name} >= {thr:g}")
            else:
                cats = pd.Categorical(s)
                for level in cats.categories:
                    columns.append((s == level).to_numpy(dtype=np.uint8))
                    names.append(f"{col_name} == {level}")
        if not columns:
            raise ValueError("no usable binary features were produced")
        return np.column_stack(columns).astype(np.uint8, copy=False), names

    arr = _to_2d_numeric(X)
    _, n_features = arr.shape
    names_in = list(feature_names) if feature_names is not None else _default_feature_names(n_features)
    if len(names_in) != n_features:
        raise ValueError("feature_names length must match X.shape[1]")

    columns: list[np.ndarray] = []
    names: list[str] = []
    for j, name in enumerate(names_in):
        col = np.asarray(arr[:, j])
        if np.issubdtype(col.dtype, np.number):
            col_float = np.asarray(col, dtype=np.float64)
            if _is_binary_array(col_float):
                columns.append(col_float.astype(np.uint8))
                names.append(str(name))
            else:
                finite = col_float[np.isfinite(col_float)]
                if finite.size == 0:
                    continue
                thresholds = np.unique(np.quantile(finite, numeric_quantiles))
                min_v = float(np.min(finite))
                max_v = float(np.max(finite))
                for thr in thresholds:
                    thr = float(thr)
                    if thr <= min_v or thr >= max_v:
                        continue
                    columns.append((col_float >= thr).astype(np.uint8))
                    names.append(f"{name} >= {thr:g}")
        else:
            if pd is None:
                raise ValueError("non-numeric ndarray input requires pandas for categorical handling")
            s = pd.Series(col)
            cats = pd.Categorical(s)
            for level in cats.categories:
                columns.append((s == level).to_numpy(dtype=np.uint8))
                names.append(f"{name} == {level}")

    if not columns:
        raise ValueError("no usable binary features were produced")
    return np.column_stack(columns).astype(np.uint8, copy=False), names


class WorstSubsetFinder:
    def __init__(self, subset_fractions=None, eps: float = 0.0, random_state: int = 0):
        self._core = _RustWorstSubsetFinder(subset_fractions, eps, random_state)
        self.random_state = int(random_state)
        self.r_hats_ = None
        self.mu_hat_ = None
        self.fold_ids_ = None

    @property
    def subset_fractions(self):
        return np.asarray(self._core.subset_fractions, dtype=float)

    def fit_from_mu_hat(self, samplewise_losses, mu_hat, fold_ids=None):
        losses = _to_1d_float(samplewise_losses)
        mu = _to_1d_float(mu_hat)
        folds = None if fold_ids is None else np.asarray(fold_ids, dtype=np.int64).reshape(-1)
        self.r_hats_ = np.asarray(self._core.fit_from_mu_hat(losses, mu, folds), dtype=float)
        self.mu_hat_ = mu
        self.fold_ids_ = folds
        return self

    def fit_with_estimator(self, X, samplewise_losses, estimator, cv: int = 5, shuffle: bool = False):
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        X_arr = np.asarray(X)
        losses = _to_1d_float(samplewise_losses)
        if X_arr.shape[0] != losses.shape[0]:
            raise ValueError("X and samplewise_losses must have the same number of rows")

        kf = KFold(n_splits=cv, shuffle=shuffle, random_state=self.random_state if shuffle else None)
        mu_hat = np.zeros(losses.shape[0], dtype=np.float64)
        fold_ids = np.zeros(losses.shape[0], dtype=np.int64)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_arr)):
            model = clone(estimator)
            model.fit(X_arr[train_idx], losses[train_idx])
            preds = np.asarray(model.predict(X_arr[test_idx]), dtype=np.float64).reshape(-1)
            if preds.shape[0] != test_idx.shape[0]:
                raise ValueError("estimator.predict returned an unexpected shape")
            mu_hat[test_idx] = preds
            fold_ids[test_idx] = fold
        return self.fit_from_mu_hat(losses, mu_hat, fold_ids=fold_ids)

    def confidence_intervals(self):
        return np.asarray(self._core.confidence_intervals(), dtype=np.float64)

    def subset_masks(self):
        return np.asarray(self._core.subset_masks(), dtype=bool)

    def observed_fractions(self):
        return np.asarray(self._core.observed_fractions(), dtype=np.float64)

    def compute_effect_sizes(self):
        return np.asarray(self._core.compute_effect_sizes(), dtype=np.float64)

    def find_max_effect_size(self):
        return self._core.find_max_effect_size()

    def select_largest_below_threshold(self, metric_values, threshold: float, smaller_is_worse: bool = True):
        metric_values = _to_1d_float(metric_values)
        return self._core.select_largest_below_threshold(metric_values, float(threshold), bool(smaller_is_worse))

    def subset_auc_table(self, y_true, y_score, n_bootstrap: int = 100, confidence: float = 0.95):
        rows = self._core.subset_auc_table(_to_1d_u8(y_true), _to_1d_float(y_score), int(n_bootstrap), float(confidence))
        if pd is not None:
            return pd.DataFrame(rows)
        return rows


class SubgroupPhenotyper:
    def __init__(
        self,
        backend: str = "sirus",
        max_depth: int = 3,
        min_support: int = 50,
        max_candidates: int = 5000,
        include_negations: bool = True,
        max_literal_prevalence: float = 0.95,
        min_lift: float = 1.0,
        effect_threshold: float = 0.4,
        bootstrap_samples: int = 100,
        random_state: int = 0,
        sirus_num_trees: int = 400,
        sirus_p0: float = 0.025,
        sirus_num_rule: int | None = None,
        sirus_mtry: int | None = None,
        sirus_min_samples_leaf: int = 5,
        sirus_sample_fraction: float = 1.0,
        sirus_replace: bool = True,
        numeric_quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    ):
        self._core = _RustSubgroupPhenotyper(
            str(backend),
            int(max_depth),
            int(min_support),
            int(max_candidates),
            bool(include_negations),
            float(max_literal_prevalence),
            float(min_lift),
            float(effect_threshold),
            int(bootstrap_samples),
            int(random_state),
            int(sirus_num_trees),
            float(sirus_p0),
            None if sirus_num_rule is None else int(sirus_num_rule),
            None if sirus_mtry is None else int(sirus_mtry),
            int(sirus_min_samples_leaf),
            float(sirus_sample_fraction),
            bool(sirus_replace),
        )
        self.backend = str(backend)
        self.numeric_quantiles = tuple(float(q) for q in numeric_quantiles)
        self.feature_names_ = None
        self.selected_rules_ = None

    def fit(
        self,
        X,
        subset_labels,
        samplewise_losses,
        *,
        y_true,
        y_score,
        performance_threshold: float,
        feature_names: Sequence[str] | None = None,
    ):
        X_bin, names = prepare_binary_feature_matrix(
            X,
            feature_names=feature_names,
            numeric_quantiles=self.numeric_quantiles,
        )
        labels = _to_1d_u8(subset_labels)
        losses = _to_1d_float(samplewise_losses)
        yt = _to_1d_u8(y_true)
        ys = _to_1d_float(y_score)
        if not (X_bin.shape[0] == labels.shape[0] == losses.shape[0] == yt.shape[0] == ys.shape[0]):
            raise ValueError("all inputs must have the same number of rows")
        self.selected_rules_ = self._core.fit(
            X_bin.astype(np.uint8, copy=False),
            labels,
            losses,
            names,
            yt,
            ys,
            float(performance_threshold),
        )
        self.feature_names_ = names
        return self

    def fit_binary(
        self,
        X_binary,
        subset_labels,
        samplewise_losses,
        *,
        y_true,
        y_score,
        performance_threshold: float,
        literal_names: Sequence[str] | None = None,
        feature_names: Sequence[str] | None = None,
    ):
        X_bin = _to_2d_binary_u8(X_binary)
        names = list(literal_names if literal_names is not None else feature_names if feature_names is not None else _default_feature_names(X_bin.shape[1]))
        if len(names) != X_bin.shape[1]:
            raise ValueError("literal_names length must match X_binary.shape[1]")
        labels = _to_1d_u8(subset_labels)
        losses = _to_1d_float(samplewise_losses)
        yt = _to_1d_u8(y_true)
        ys = _to_1d_float(y_score)
        if not (X_bin.shape[0] == labels.shape[0] == losses.shape[0] == yt.shape[0] == ys.shape[0]):
            raise ValueError("all inputs must have the same number of rows")
        self.selected_rules_ = self._core.fit(
            X_bin,
            labels,
            losses,
            names,
            yt,
            ys,
            float(performance_threshold),
        )
        self.feature_names_ = names
        return self

    def selected_rules(self):
        return list(self._core.selected_rules())

    def summary_table(self):
        rows = self._core.summary_table()
        if pd is not None:
            return pd.DataFrame(rows)
        return rows

    def fit_metadata(self):
        metadata = dict(self._core.fit_metadata())
        metadata.update(
            {
                "backend": self.backend,
                "n_selected_rules": len(self.selected_rules() if self.selected_rules_ is not None else []),
                "n_binary_features": None if self.feature_names_ is None else len(self.feature_names_),
                "numeric_quantiles": self.numeric_quantiles,
            }
        )
        if pd is not None:
            return pd.Series(metadata)
        return metadata
