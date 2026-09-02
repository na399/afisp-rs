use crate::prng::SplitMix64;
use crate::stats::{mean, quantile_sorted};

pub fn clip_predictions(preds: &[f64], lower: f64, upper: f64) -> Vec<f64> {
    preds.iter().map(|p| p.max(lower).min(upper)).collect()
}

pub fn cross_entropy_loss(y: &[u8], y_pred: &[f64]) -> Vec<f64> {
    let clipped = clip_predictions(y_pred, 1e-6, 1.0 - 1e-6);
    y.iter()
        .zip(clipped.iter())
        .map(|(yt, yp)| {
            let yv = *yt as f64;
            -(yv * yp.ln() + (1.0 - yv) * (1.0 - yp).ln())
        })
        .collect()
}

pub fn brier_loss(y: &[u8], y_pred: &[f64]) -> Vec<f64> {
    y.iter()
        .zip(y_pred.iter())
        .map(|(yt, yp)| {
            let d = *yt as f64 - *yp;
            d * d
        })
        .collect()
}

pub fn zero_one_loss(y: &[u8], y_pred_binary: &[u8]) -> Vec<f64> {
    y.iter()
        .zip(y_pred_binary.iter())
        .map(|(yt, yp)| if yt == yp { 0.0 } else { 1.0 })
        .collect()
}

pub fn roc_auc_score(y: &[u8], scores: &[f64]) -> Option<f64> {
    if y.len() != scores.len() || y.is_empty() {
        return None;
    }
    let n_pos = y.iter().filter(|&&v| v == 1).count();
    let n_neg = y.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }

    let mut pairs = scores
        .iter()
        .copied()
        .zip(y.iter().copied())
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut i = 0usize;
    let mut sum_ranks_pos = 0.0;
    while i < pairs.len() {
        let start = i;
        let score = pairs[i].0;
        let mut positives_in_tie = 0usize;
        while i < pairs.len() && pairs[i].0 == score {
            positives_in_tie += pairs[i].1 as usize;
            i += 1;
        }
        let end = i;
        let avg_rank = (start as f64 + 1.0 + end as f64) / 2.0;
        sum_ranks_pos += avg_rank * positives_in_tie as f64;
    }

    let n_pos_f = n_pos as f64;
    let n_neg_f = n_neg as f64;
    Some((sum_ranks_pos - n_pos_f * (n_pos_f + 1.0) / 2.0) / (n_pos_f * n_neg_f))
}

// Complexity: O(B * M log M), where B is the number of bootstrap resamples
// and M is the subgroup size in indices.
pub fn bootstrap_auc_ci(
    y: &[u8],
    scores: &[f64],
    indices: &[usize],
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> Option<(f64, f64, f64, Vec<f64>)> {
    if indices.is_empty() {
        return None;
    }
    let mut local_y = Vec::with_capacity(indices.len());
    let mut local_scores = Vec::with_capacity(indices.len());
    for &idx in indices {
        local_y.push(y[idx]);
        local_scores.push(scores[idx]);
    }
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(n_bootstrap);
    for _ in 0..n_bootstrap {
        let mut by = Vec::with_capacity(indices.len());
        let mut bs = Vec::with_capacity(indices.len());
        for _ in 0..indices.len() {
            let j = rng.gen_index(indices.len());
            by.push(local_y[j]);
            bs.push(local_scores[j]);
        }
        if let Some(auc) = roc_auc_score(&by, &bs) {
            samples.push(auc);
        }
    }
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let lower_q = (1.0 - confidence) / 2.0;
    let upper_q = 1.0 - lower_q;
    let low = quantile_sorted(&samples, lower_q);
    let high = quantile_sorted(&samples, upper_q);
    Some((mean(&samples), low, high, samples))
}
