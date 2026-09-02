from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from afisp_rs import WorstSubsetFinder as _RustWorstSubsetFinder


class WorstSubsetFinder(BaseEstimator):
    """Sklearn-compatible stability analysis backed by the Rust AFISP core."""

    def __init__(
        self,
        subset_fractions=None,
        conditional_loss_model=None,
        cv=10,
        verbose=False,
        eps=0.0,
        random_state=0,
    ):
        if eps < 0:
            raise RuntimeError("eps must be a float >= 0")
        if subset_fractions is None:
            subset_fractions = [0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.subset_fractions = subset_fractions
        self.conditional_loss_model = conditional_loss_model
        self.cv = cv
        self.verbose = verbose
        self.eps = eps
        self.random_state = random_state
        self.fit_called_ = False
        self._masks = None
        self._effect_sizes_computed = False
        self._core = _RustWorstSubsetFinder(
            subset_fractions=list(subset_fractions),
            eps=float(eps),
            random_state=int(random_state),
        )

    def fit(self, subgroup_feature_data, samplewise_losses, feature_names=None):
        from sklearn.base import clone

        samplewise_losses = np.asarray(samplewise_losses, dtype=np.float64).reshape(-1)
        x = np.asarray(subgroup_feature_data)
        n = x.shape[0]
        self.num_samples_ = n
        self._samplewise_losses = samplewise_losses

        mu_mdl = self.conditional_loss_model
        if mu_mdl is None:
            from interpret.glassbox import ExplainableBoostingRegressor

            mu_mdl = ExplainableBoostingRegressor(feature_names=feature_names)

        self._core.fit_with_estimator(
            x,
            samplewise_losses,
            clone(mu_mdl),
            cv=int(self.cv),
            shuffle=False,
        )
        self.mu_hat_ = np.asarray(self._core.mu_hat_, dtype=np.float64)
        self.R_hats_ = np.asarray(self._core.r_hats_, dtype=np.float64)
        cis = self._core.confidence_intervals()
        self.cis_ = np.asarray(cis, dtype=np.float64)
        self.sigma_hats_ = (self.cis_[:, 1] - self.cis_[:, 0]) / (2 * 1.96) * np.sqrt(n)
        self._masks = None
        self._effect_sizes_computed = False
        self.fit_called_ = True
        return self.R_hats_

    def confidence_intervals(self):
        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on WorstSubsetFinder object first.')
        return self.cis_

    def subset_masks(self):
        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on WorstSubsetFinder object first.')
        if self._masks is None:
            self._masks = self._core.subset_masks()
        return [self._masks[i] for i in range(len(self.subset_fractions))]

    def check_subset_sizes(self, plot=True, ax=None):
        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on WorstSubsetFinder object first.')
        observed_fractions = list(self._core.observed_fractions())
        if plot:
            import matplotlib.pyplot as plt

            if ax is None:
                ax = plt.gca()
            ax.plot(self.subset_fractions, observed_fractions)
            ax.plot([0, 1], [0, 1], "k:", label="Perfect fit")
            ax.set_xlabel("Subset Fraction")
            ax.set_ylabel("Fraction Selected by Worst-Case Mask")
            ax.legend(loc="best")
        return self.subset_fractions, observed_fractions

    def compute_effect_sizes(self, plot=False, ax=None):
        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on WorstSubsetFinder object first.')
        cds = list(self._core.compute_effect_sizes())
        if plot:
            import matplotlib.pyplot as plt

            if ax is None:
                ax = plt.gca()
            ax.plot(self.subset_fractions, cds)
            ax.set_xlabel("Subset Fraction")
            ax.set_ylabel("Cohen's d (Effect Size) of Loss")
        self._effect_sizes = cds
        self._effect_sizes_computed = True
        return self.subset_fractions, cds

    def find_max_effect_size(self):
        if not self._effect_sizes_computed:
            self.compute_effect_sizes(plot=False)
        idx, value = self._core.find_max_effect_size()
        return int(idx), float(value)
