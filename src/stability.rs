use std::collections::BTreeMap;

use crate::metrics::bootstrap_auc_ci;
use crate::prng::SplitMix64;
use crate::stats::{cohens_d, quantile_sorted, variance_population};

pub type SubsetAucRow = (f64, usize, f64, f64, f64);

#[derive(Clone, Debug)]
pub struct WorstSubsetFinderCore {
    pub subset_fractions: Vec<f64>,
    pub eps: f64,
    pub random_state: u64,
    pub fit_called: bool,
    pub samplewise_losses: Vec<f64>,
    pub mu_hat: Vec<f64>,
    pub tie_noise: Vec<f64>,
    pub eta_hat: Vec<Vec<f64>>,
    pub masks: Vec<Vec<bool>>,
    pub r_hats: Vec<f64>,
    pub sigma_hats: Vec<f64>,
    pub cis: Vec<(f64, f64)>,
    pub effect_sizes: Option<Vec<f64>>,
}

impl WorstSubsetFinderCore {
    pub fn new(subset_fractions: Vec<f64>, eps: f64, random_state: u64) -> Self {
        Self {
            subset_fractions,
            eps,
            random_state,
            fit_called: false,
            samplewise_losses: Vec::new(),
            mu_hat: Vec::new(),
            tie_noise: Vec::new(),
            eta_hat: Vec::new(),
            masks: Vec::new(),
            r_hats: Vec::new(),
            sigma_hats: Vec::new(),
            cis: Vec::new(),
            effect_sizes: None,
        }
    }

    // Complexity: O(sum_folds n_f log n_f + A * N), where A is the number of
    // subset fractions and N is the number of samples. Sorting once per fold
    // lets us reuse quantiles for all subset fractions.
    pub fn fit_from_mu_hat(
        &mut self,
        samplewise_losses: Vec<f64>,
        mu_hat: Vec<f64>,
        fold_ids: Option<Vec<usize>>,
    ) -> Result<Vec<f64>, String> {
        let n = samplewise_losses.len();
        if n == 0 {
            return Err("samplewise_losses must be non-empty".into());
        }
        if mu_hat.len() != n {
            return Err("mu_hat must have the same length as samplewise_losses".into());
        }
        if self.subset_fractions.is_empty() {
            return Err("subset_fractions must be non-empty".into());
        }
        if self
            .subset_fractions
            .iter()
            .any(|&a| !(a > 0.0 && a <= 1.0))
        {
            return Err("subset_fractions must all lie in (0, 1]".into());
        }
        let fold_ids = fold_ids.unwrap_or_else(|| vec![0usize; n]);
        if fold_ids.len() != n {
            return Err("fold_ids must have the same length as samplewise_losses".into());
        }

        self.samplewise_losses = samplewise_losses;
        self.mu_hat = mu_hat;
        self.tie_noise = if self.eps > 0.0 {
            let mut rng = SplitMix64::new(self.random_state);
            (0..n).map(|_| rng.next_f64() * self.eps).collect()
        } else {
            vec![0.0; n]
        };

        self.eta_hat = vec![vec![0.0; n]; self.subset_fractions.len()];
        let mut groups = BTreeMap::<usize, Vec<usize>>::new();
        for (idx, fold) in fold_ids.iter().copied().enumerate() {
            groups.entry(fold).or_default().push(idx);
        }

        for indices in groups.values() {
            let mut local = indices.iter().map(|&i| self.mu_hat[i]).collect::<Vec<_>>();
            local.sort_by(|a, b| a.total_cmp(b));
            for (a_idx, &alpha) in self.subset_fractions.iter().enumerate() {
                let q = quantile_sorted(&local, 1.0 - alpha);
                for &i in indices {
                    self.eta_hat[a_idx][i] = q;
                }
            }
        }

        self.r_hats.clear();
        self.sigma_hats.clear();
        self.cis.clear();
        self.masks.clear();
        for (a_idx, &alpha) in self.subset_fractions.iter().enumerate() {
            let mut psi = vec![0.0; n];
            let mut mask = vec![false; n];
            for i in 0..n {
                let shifted = self.mu_hat[i] + self.tie_noise[i];
                let eta = self.eta_hat[a_idx][i];
                let h = shifted >= eta;
                mask[i] = h;
                let mut psi_i = (shifted - eta).max(0.0) / alpha + eta;
                if h {
                    psi_i += (self.samplewise_losses[i] - self.mu_hat[i]) / alpha;
                }
                psi[i] = psi_i;
            }
            let r_hat = psi.iter().sum::<f64>() / n as f64;
            let sigma_hat = variance_population(&psi).sqrt();
            let margin = 1.96 * sigma_hat / (n as f64).sqrt();
            self.r_hats.push(r_hat);
            self.sigma_hats.push(sigma_hat);
            self.cis.push((r_hat - margin, r_hat + margin));
            self.masks.push(mask);
        }

        self.effect_sizes = None;
        self.fit_called = true;
        Ok(self.r_hats.clone())
    }

    pub fn observed_fractions(&self) -> Result<Vec<f64>, String> {
        if !self.fit_called {
            return Err("call fit_from_mu_hat first".into());
        }
        Ok(self
            .masks
            .iter()
            .map(|m| m.iter().filter(|&&b| b).count() as f64 / m.len() as f64)
            .collect())
    }

    pub fn compute_effect_sizes(&mut self) -> Result<Vec<f64>, String> {
        if !self.fit_called {
            return Err("call fit_from_mu_hat first".into());
        }
        let mut out = Vec::with_capacity(self.masks.len());
        for mask in &self.masks {
            let mut in_group = Vec::new();
            let mut out_group = Vec::new();
            for (idx, &flag) in mask.iter().enumerate() {
                if flag {
                    in_group.push(self.samplewise_losses[idx]);
                } else {
                    out_group.push(self.samplewise_losses[idx]);
                }
            }
            out.push(cohens_d(&in_group, &out_group));
        }
        self.effect_sizes = Some(out.clone());
        Ok(out)
    }

    pub fn find_max_effect_size(&mut self) -> Result<(usize, f64), String> {
        let effects = match self.effect_sizes.clone() {
            Some(v) => v,
            None => self.compute_effect_sizes()?,
        };
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (idx, &val) in effects.iter().enumerate() {
            if val > best_val {
                best_idx = idx;
                best_val = val;
            }
        }
        Ok((best_idx, best_val))
    }

    pub fn select_largest_below_threshold(
        &self,
        metric_values: &[f64],
        threshold: f64,
        smaller_is_worse: bool,
    ) -> Result<Option<(usize, f64, f64)>, String> {
        if !self.fit_called {
            return Err("call fit_from_mu_hat first".into());
        }
        if metric_values.len() != self.subset_fractions.len() {
            return Err("metric_values must have the same length as subset_fractions".into());
        }
        let mut best: Option<(usize, f64, f64)> = None;
        for (idx, (&alpha, &value)) in self
            .subset_fractions
            .iter()
            .zip(metric_values.iter())
            .enumerate()
        {
            let is_bad = if smaller_is_worse {
                value < threshold
            } else {
                value > threshold
            };
            if is_bad {
                match best {
                    None => best = Some((idx, alpha, value)),
                    Some((_, best_alpha, _)) if alpha > best_alpha => {
                        best = Some((idx, alpha, value))
                    }
                    _ => {}
                }
            }
        }
        Ok(best)
    }

    // Complexity: O(A * (N + B * M log M)) where A is the number of subset
    // fractions, B the number of bootstrap samples, and M the subgroup size.
    pub fn subset_auc_table(
        &self,
        y_true: &[u8],
        y_score: &[f64],
        n_bootstrap: usize,
        confidence: f64,
    ) -> Result<Vec<SubsetAucRow>, String> {
        if !self.fit_called {
            return Err("call fit_from_mu_hat first".into());
        }
        if y_true.len() != self.samplewise_losses.len()
            || y_score.len() != self.samplewise_losses.len()
        {
            return Err("y_true and y_score must match fitted sample count".into());
        }
        let mut out = Vec::with_capacity(self.masks.len());
        for (idx, mask) in self.masks.iter().enumerate() {
            let indices = mask
                .iter()
                .enumerate()
                .filter_map(|(i, &flag)| if flag { Some(i) } else { None })
                .collect::<Vec<_>>();
            let seed = self.random_state.wrapping_add(idx as u64).wrapping_add(17);
            if let Some((m, l, u, _)) =
                bootstrap_auc_ci(y_true, y_score, &indices, n_bootstrap, confidence, seed)
            {
                out.push((self.subset_fractions[idx], indices.len(), m, l, u));
            } else {
                out.push((
                    self.subset_fractions[idx],
                    indices.len(),
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                ));
            }
        }
        Ok(out)
    }
}
