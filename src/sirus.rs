use std::collections::HashMap;

use crate::bitset::BitMask;
use crate::prng::SplitMix64;
use crate::rules::{render_conjunction, sort_literals, Literal};

#[derive(Clone, Debug)]
pub struct SirusConfig {
    pub num_trees: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub mtry: Option<usize>,
    pub p0: Option<f64>,
    pub num_rule: Option<usize>,
    pub min_support: usize,
    pub max_candidates: usize,
    pub sample_fraction: f64,
    pub replace: bool,
    pub include_negations: bool,
    pub max_prevalence: f64,
}

#[derive(Clone, Debug)]
pub struct SirusCandidate {
    pub phenotype: String,
    pub literals: Vec<Literal>,
    pub mask: BitMask,
    pub support: usize,
    pub positives: usize,
    pub precision: f64,
    pub recall: f64,
    pub lift: f64,
    pub f1: f64,
    pub frequency: f64,
    pub output_true: f64,
    pub output_false: f64,
}

#[derive(Clone, Debug)]
struct StableRule {
    literals: Vec<Literal>,
    frequency: f64,
}

#[derive(Clone, Debug, Default)]
struct RuleCounter {
    count: usize,
}

#[derive(Clone, Debug)]
struct SplitStat {
    feature_idx: usize,
    gain: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

// Complexity: O(T * S * mtry * D + R * N * W), where T is the number of trees,
// S the average per-tree sample size, D the maximum depth, R the number of
// stable rules, N the dataset size, and W the bitset word count.
pub fn extract_sirus_candidates(
    x_binary: &[Vec<u8>],
    subset_labels: &[u8],
    feature_names: &[String],
    config: &SirusConfig,
    random_state: u64,
) -> Result<Vec<SirusCandidate>, String> {
    validate_inputs(x_binary, subset_labels, feature_names)?;
    validate_config(config)?;

    let n = x_binary.len();
    let total_pos = subset_labels.iter().map(|&v| v as usize).sum::<usize>();
    if total_pos == 0 || total_pos == n {
        return Ok(Vec::new());
    }

    let effective_p0 = config.p0.unwrap_or(0.025);
    let stable_rules =
        extract_stable_rules(x_binary, subset_labels, config, effective_p0, random_state)?;

    let base_rate = total_pos as f64 / n as f64;
    let mut out = Vec::<SirusCandidate>::new();
    for stable in stable_rules {
        let mask = mask_for_rule(x_binary, &stable.literals);
        let support = mask.count_ones();
        if support < config.min_support || support >= n {
            continue;
        }
        let prevalence = support as f64 / n as f64;
        if prevalence > config.max_prevalence {
            continue;
        }
        let positives = mask
            .indices()
            .into_iter()
            .map(|idx| subset_labels[idx] as usize)
            .sum::<usize>();
        if positives == 0 {
            continue;
        }
        let precision = positives as f64 / support as f64;
        let recall = positives as f64 / total_pos as f64;
        let lift = precision / base_rate;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let outside_support = n - support;
        let outside_pos = total_pos - positives;
        let output_false = if outside_support > 0 {
            outside_pos as f64 / outside_support as f64
        } else {
            0.0
        };
        out.push(SirusCandidate {
            phenotype: render_conjunction(feature_names, &stable.literals),
            literals: stable.literals.clone(),
            mask,
            support,
            positives,
            precision,
            recall,
            lift,
            f1,
            frequency: stable.frequency,
            output_true: precision,
            output_false,
        });
    }

    out.sort_by(|a, b| {
        b.frequency
            .total_cmp(&a.frequency)
            .then_with(|| b.lift.total_cmp(&a.lift))
            .then_with(|| b.f1.total_cmp(&a.f1))
            .then_with(|| b.support.cmp(&a.support))
            .then_with(|| a.phenotype.cmp(&b.phenotype))
    });
    if let Some(limit) = config.num_rule {
        if out.len() > limit {
            out.truncate(limit);
        }
    }
    if out.len() > config.max_candidates {
        out.truncate(config.max_candidates);
    }
    Ok(out)
}

fn validate_inputs(
    x_binary: &[Vec<u8>],
    subset_labels: &[u8],
    feature_names: &[String],
) -> Result<(), String> {
    if x_binary.is_empty() {
        return Err("x_binary must be non-empty".into());
    }
    let p = x_binary[0].len();
    if p == 0 {
        return Err("x_binary must have at least one feature".into());
    }
    if x_binary.iter().any(|row| row.len() != p) {
        return Err("x_binary must be rectangular".into());
    }
    if subset_labels.len() != x_binary.len() {
        return Err("subset_labels must have the same row count as x_binary".into());
    }
    if feature_names.len() != p {
        return Err("feature_names length must match x_binary column count".into());
    }
    Ok(())
}

fn validate_config(config: &SirusConfig) -> Result<(), String> {
    if config.num_trees == 0 {
        return Err("num_trees must be >= 1".into());
    }
    if config.max_depth == 0 {
        return Err("max_depth must be >= 1".into());
    }
    if config.min_samples_leaf == 0 {
        return Err("min_samples_leaf must be >= 1".into());
    }
    if let Some(p0) = config.p0 {
        if !(0.0 < p0 && p0 <= 1.0) {
            return Err("p0 must be in (0, 1]".into());
        }
    }
    if !(0.0 < config.sample_fraction && config.sample_fraction <= 1.0) {
        return Err("sample_fraction must be in (0, 1]".into());
    }
    if !(0.0 < config.max_prevalence && config.max_prevalence <= 1.0) {
        return Err("max_prevalence must be in (0, 1]".into());
    }
    Ok(())
}

fn mask_for_rule(x_binary: &[Vec<u8>], literals: &[Literal]) -> BitMask {
    let n = x_binary.len();
    BitMask::from_bool_iter(
        n,
        x_binary.iter().map(|row| {
            literals.iter().all(|lit| {
                let value = row[lit.feature_idx] > 0;
                if lit.positive {
                    value
                } else {
                    !value
                }
            })
        }),
    )
}

fn extract_stable_rules(
    x_binary: &[Vec<u8>],
    subset_labels: &[u8],
    config: &SirusConfig,
    p0: f64,
    random_state: u64,
) -> Result<Vec<StableRule>, String> {
    let n = x_binary.len();
    let p = x_binary[0].len();
    let resolved_mtry = resolve_mtry(config.mtry, p);
    let mut rng = SplitMix64::new(random_state);
    let mut counts = HashMap::<Vec<Literal>, RuleCounter>::new();

    for _ in 0..config.num_trees {
        let sample = draw_sample_indices(n, config.sample_fraction, config.replace, &mut rng);
        let sample_n = sample.len();
        let sample_pos = sample
            .iter()
            .map(|&idx| subset_labels[idx] as usize)
            .sum::<usize>();
        if sample_pos == 0 || sample_pos == sample_n {
            continue;
        }

        let mut path = Vec::<Literal>::new();
        let mut used_features = Vec::<usize>::new();
        let mut per_tree = HashMap::<Vec<Literal>, f64>::new();
        grow_tree(
            x_binary,
            subset_labels,
            &sample,
            sample_n,
            sample_pos,
            config.max_depth,
            0,
            config.min_samples_leaf,
            resolved_mtry,
            config.include_negations,
            &mut path,
            &mut used_features,
            &mut per_tree,
            &mut rng,
        );
        for (rule, _) in per_tree {
            let entry = counts.entry(rule).or_default();
            entry.count += 1;
        }
    }

    let mut out = counts
        .into_iter()
        .filter_map(|(mut literals, counter)| {
            let frequency = counter.count as f64 / config.num_trees as f64;
            if frequency + 1e-12 < p0 {
                return None;
            }
            sort_literals(&mut literals);
            Some(StableRule {
                literals,
                frequency,
            })
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| {
        b.frequency
            .total_cmp(&a.frequency)
            .then_with(|| a.literals.len().cmp(&b.literals.len()))
            .then_with(|| a.literals.cmp(&b.literals))
    });
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn grow_tree(
    x_binary: &[Vec<u8>],
    subset_labels: &[u8],
    node_indices: &[usize],
    sample_n: usize,
    sample_pos: usize,
    max_depth: usize,
    depth: usize,
    min_leaf_size: usize,
    mtry: usize,
    include_negations: bool,
    path: &mut Vec<Literal>,
    used_features: &mut Vec<usize>,
    per_tree: &mut HashMap<Vec<Literal>, f64>,
    rng: &mut SplitMix64,
) {
    if node_indices.is_empty() {
        return;
    }
    let node_n = node_indices.len();
    let node_pos = node_indices
        .iter()
        .map(|&idx| subset_labels[idx] as usize)
        .sum::<usize>();
    let node_rate = node_pos as f64 / node_n as f64;
    let outside_n = sample_n.saturating_sub(node_n);
    let outside_pos = sample_pos.saturating_sub(node_pos);
    let outside_rate = if outside_n > 0 {
        outside_pos as f64 / outside_n as f64
    } else {
        0.0
    };

    if !path.is_empty() && node_rate > outside_rate {
        let has_negative = path.iter().any(|lit| !lit.positive);
        if include_negations || !has_negative {
            let mut key = path.clone();
            sort_literals(&mut key);
            let score = node_rate - outside_rate;
            let entry = per_tree.entry(key).or_insert(score);
            if score > *entry {
                *entry = score;
            }
        }
    }

    if depth >= max_depth {
        return;
    }
    if node_n < min_leaf_size.saturating_mul(2).max(2) {
        return;
    }
    if node_pos == 0 || node_pos == node_n {
        return;
    }

    let available = (0..x_binary[0].len())
        .filter(|feat| !used_features.contains(feat))
        .collect::<Vec<_>>();
    if available.is_empty() {
        return;
    }
    let candidate_features = sample_feature_subset(&available, mtry, rng);
    let Some(best) = best_split(
        x_binary,
        subset_labels,
        node_indices,
        &candidate_features,
        min_leaf_size,
    ) else {
        return;
    };
    if best.gain.is_nan() || best.gain <= 0.0 {
        return;
    }

    used_features.push(best.feature_idx);

    path.push(Literal {
        feature_idx: best.feature_idx,
        positive: false,
    });
    grow_tree(
        x_binary,
        subset_labels,
        &best.left_indices,
        sample_n,
        sample_pos,
        max_depth,
        depth + 1,
        min_leaf_size,
        mtry,
        include_negations,
        path,
        used_features,
        per_tree,
        rng,
    );
    path.pop();

    path.push(Literal {
        feature_idx: best.feature_idx,
        positive: true,
    });
    grow_tree(
        x_binary,
        subset_labels,
        &best.right_indices,
        sample_n,
        sample_pos,
        max_depth,
        depth + 1,
        min_leaf_size,
        mtry,
        include_negations,
        path,
        used_features,
        per_tree,
        rng,
    );
    path.pop();

    used_features.pop();
}

fn best_split(
    x_binary: &[Vec<u8>],
    subset_labels: &[u8],
    node_indices: &[usize],
    candidate_features: &[usize],
    min_leaf_size: usize,
) -> Option<SplitStat> {
    let node_n = node_indices.len();
    let node_pos = node_indices
        .iter()
        .map(|&idx| subset_labels[idx] as usize)
        .sum::<usize>();
    let parent_impurity = gini_impurity(node_pos, node_n);
    let mut best: Option<SplitStat> = None;

    for &feat in candidate_features {
        let mut left_indices = Vec::<usize>::new();
        let mut right_indices = Vec::<usize>::new();
        let mut left_pos = 0usize;
        let mut right_pos = 0usize;

        for &idx in node_indices {
            if x_binary[idx][feat] > 0 {
                right_indices.push(idx);
                right_pos += subset_labels[idx] as usize;
            } else {
                left_indices.push(idx);
                left_pos += subset_labels[idx] as usize;
            }
        }
        if left_indices.len() < min_leaf_size || right_indices.len() < min_leaf_size {
            continue;
        }
        let gain = gini_gain(
            parent_impurity,
            left_pos,
            left_indices.len(),
            right_pos,
            right_indices.len(),
        );
        match &best {
            Some(cur) if gain <= cur.gain => {}
            _ => {
                best = Some(SplitStat {
                    feature_idx: feat,
                    gain,
                    left_indices,
                    right_indices,
                })
            }
        }
    }
    best
}

fn resolve_mtry(mtry: Option<usize>, p: usize) -> usize {
    match mtry {
        Some(v) if v > 0 => v.min(p),
        _ => ((p as f64).sqrt().round() as usize).max(1).min(p),
    }
}

fn draw_sample_indices(
    n: usize,
    sample_fraction: f64,
    replace: bool,
    rng: &mut SplitMix64,
) -> Vec<usize> {
    let sample_n = ((sample_fraction * n as f64).round() as usize)
        .max(1)
        .min(n);
    if replace {
        let mut out = Vec::with_capacity(sample_n);
        for _ in 0..sample_n {
            out.push(rng.gen_index(n));
        }
        out
    } else {
        let mut pool = (0..n).collect::<Vec<_>>();
        for i in 0..sample_n {
            let j = i + rng.gen_index(n - i);
            pool.swap(i, j);
        }
        pool.truncate(sample_n);
        pool
    }
}

fn sample_feature_subset(available: &[usize], mtry: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let take = mtry.min(available.len());
    if take == available.len() {
        return available.to_vec();
    }
    let mut pool = available.to_vec();
    for i in 0..take {
        let j = i + rng.gen_index(pool.len() - i);
        pool.swap(i, j);
    }
    pool.truncate(take);
    pool
}

fn gini_impurity(positives: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let p = positives as f64 / total as f64;
    1.0 - p * p - (1.0 - p) * (1.0 - p)
}

fn gini_gain(
    parent_impurity: f64,
    left_pos: usize,
    left_n: usize,
    right_pos: usize,
    right_n: usize,
) -> f64 {
    let total = left_n + right_n;
    if total == 0 {
        return 0.0;
    }
    let left_weight = left_n as f64 / total as f64;
    let right_weight = right_n as f64 / total as f64;
    parent_impurity
        - left_weight * gini_impurity(left_pos, left_n)
        - right_weight * gini_impurity(right_pos, right_n)
}
