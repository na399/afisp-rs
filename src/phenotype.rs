use crate::bitset::BitMask;
use crate::metrics::bootstrap_auc_ci;
use crate::rules::{render_conjunction, Literal};
use crate::sirus::{extract_sirus_candidates, SirusConfig};
use crate::stats::{cohens_d, z_test_less};

#[derive(Clone, Debug)]
struct LiteralMask {
    literal: Literal,
    mask: BitMask,
    support: usize,
}

#[derive(Clone, Debug)]
struct ConjunctionCandidate {
    literals: Vec<Literal>,
    mask: BitMask,
    support: usize,
    positives: usize,
    precision: f64,
    recall: f64,
    lift: f64,
    f1: f64,
}

struct ConjunctionSearch<'a> {
    literals: &'a [LiteralMask],
    subset_mask: &'a BitMask,
    total_subset: usize,
    base_rate: f64,
}

#[derive(Clone, Debug)]
struct PhenotypeCandidate {
    phenotype: String,
    literals: Vec<Literal>,
    mask: BitMask,
    support: usize,
    positives: usize,
    precision: f64,
    recall: f64,
    lift: f64,
    f1: f64,
    frequency: f64,
    output_true: f64,
    output_false: f64,
    backend: String,
}

#[derive(Clone, Debug)]
pub struct ScoredRule {
    pub phenotype: String,
    pub literals: Vec<Literal>,
    pub support: usize,
    pub positives: usize,
    pub precision: f64,
    pub recall: f64,
    pub lift: f64,
    pub f1: f64,
    pub frequency: f64,
    pub output_true: f64,
    pub output_false: f64,
    pub backend: String,
    pub auroc: f64,
    pub auroc_lower: f64,
    pub auroc_upper: f64,
    pub p_value: f64,
    pub effect_size: f64,
}

pub struct PhenotypeFitInput<'a> {
    pub x_binary: &'a [Vec<u8>],
    pub subset_labels: &'a [u8],
    pub samplewise_losses: &'a [f64],
    pub feature_names: Vec<String>,
    pub y_true: &'a [u8],
    pub y_score: &'a [f64],
    pub performance_threshold: f64,
}

#[derive(Clone, Debug)]
pub struct SubgroupPhenotyperCore {
    pub backend: String,
    pub max_depth: usize,
    pub min_support: usize,
    pub max_candidates: usize,
    pub include_negations: bool,
    pub max_literal_prevalence: f64,
    pub min_lift: f64,
    pub effect_threshold: f64,
    pub bootstrap_samples: usize,
    pub random_state: u64,
    pub sirus_num_trees: usize,
    pub sirus_p0: Option<f64>,
    pub sirus_num_rule: Option<usize>,
    pub sirus_mtry: Option<usize>,
    pub sirus_min_samples_leaf: usize,
    pub sirus_sample_fraction: f64,
    pub sirus_replace: bool,
    pub feature_names: Vec<String>,
    pub fit_called: bool,
    pub selected_rules: Vec<ScoredRule>,
}

impl SubgroupPhenotyperCore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: String,
        max_depth: usize,
        min_support: usize,
        max_candidates: usize,
        include_negations: bool,
        max_literal_prevalence: f64,
        min_lift: f64,
        effect_threshold: f64,
        bootstrap_samples: usize,
        random_state: u64,
        sirus_num_trees: usize,
        sirus_p0: Option<f64>,
        sirus_num_rule: Option<usize>,
        sirus_mtry: Option<usize>,
        sirus_min_samples_leaf: usize,
        sirus_sample_fraction: f64,
        sirus_replace: bool,
    ) -> Self {
        Self {
            backend,
            max_depth,
            min_support,
            max_candidates,
            include_negations,
            max_literal_prevalence,
            min_lift,
            effect_threshold,
            bootstrap_samples,
            random_state,
            sirus_num_trees,
            sirus_p0,
            sirus_num_rule,
            sirus_mtry,
            sirus_min_samples_leaf,
            sirus_sample_fraction,
            sirus_replace,
            feature_names: Vec::new(),
            fit_called: false,
            selected_rules: Vec::new(),
        }
    }

    // Complexity: candidate generation is either
    // - conjunction mode: O(C * W) bitset work for C explored conjunctions and
    //   W = ceil(N/64) machine words per bitset; or
    // - SIRUS mode: O(T * N * mtry * depth) for forest growth plus O(R * N)
    //   for rule scoring and support-based de-duplication.
    // Final subgroup scoring adds O(R * (N + B * M log M)) for B bootstrap
    // samples and subgroup size M.
    pub fn fit(&mut self, input: PhenotypeFitInput<'_>) -> Result<Vec<String>, String> {
        let PhenotypeFitInput {
            x_binary,
            subset_labels,
            samplewise_losses,
            feature_names,
            y_true,
            y_score,
            performance_threshold,
        } = input;
        let n = x_binary.len();
        if n == 0 {
            return Err("x_binary must be non-empty".into());
        }
        if subset_labels.len() != n
            || samplewise_losses.len() != n
            || y_true.len() != n
            || y_score.len() != n
        {
            return Err("all inputs must have the same row count".into());
        }
        let p = x_binary[0].len();
        if feature_names.len() != p {
            return Err("feature_names length must match x_binary column count".into());
        }
        if x_binary.iter().any(|row| row.len() != p) {
            return Err("x_binary must be rectangular".into());
        }
        self.feature_names = feature_names;

        let backend = self.backend.to_lowercase();
        let candidates = match backend.as_str() {
            "sirus" => self.generate_sirus_candidates(x_binary, subset_labels)?,
            "conjunction" | "conjunctions" | "miner" => {
                self.generate_conjunction_candidates(x_binary, subset_labels)?
            }
            other => {
                return Err(format!(
                    "unknown backend '{}' expected one of 'sirus' or 'conjunction'",
                    other
                ))
            }
        };

        if candidates.is_empty() {
            self.selected_rules.clear();
            self.fit_called = true;
            return Ok(Vec::new());
        }

        let mut scored = self.score_candidates(
            candidates,
            samplewise_losses,
            y_true,
            y_score,
            performance_threshold,
        );
        scored.sort_by(|a, b| a.p_value.total_cmp(&b.p_value));
        let significant = holm_bonferroni(scored, 0.05);
        let mut selected = significant
            .into_iter()
            .filter(|r| r.effect_size >= self.effect_threshold)
            .collect::<Vec<_>>();
        selected.sort_by(|a, b| {
            a.auroc
                .total_cmp(&b.auroc)
                .then_with(|| a.p_value.total_cmp(&b.p_value))
                .then_with(|| b.effect_size.total_cmp(&a.effect_size))
                .then_with(|| b.frequency.total_cmp(&a.frequency))
        });
        self.selected_rules = selected;
        self.fit_called = true;
        Ok(self
            .selected_rules
            .iter()
            .map(|r| r.phenotype.clone())
            .collect())
    }

    fn generate_sirus_candidates(
        &self,
        x_binary: &[Vec<u8>],
        subset_labels: &[u8],
    ) -> Result<Vec<PhenotypeCandidate>, String> {
        let config = SirusConfig {
            num_trees: self.sirus_num_trees.max(1),
            max_depth: self.max_depth.max(1),
            min_samples_leaf: self.sirus_min_samples_leaf.max(1),
            mtry: self.sirus_mtry,
            p0: self.sirus_p0,
            num_rule: self.sirus_num_rule,
            min_support: self.min_support,
            max_candidates: self.max_candidates,
            sample_fraction: self.sirus_sample_fraction,
            replace: self.sirus_replace,
            include_negations: self.include_negations,
            max_prevalence: self.max_literal_prevalence,
        };
        let raw = extract_sirus_candidates(
            x_binary,
            subset_labels,
            &self.feature_names,
            &config,
            self.random_state,
        )?;
        Ok(raw
            .into_iter()
            .filter(|cand| cand.lift >= self.min_lift)
            .map(|cand| PhenotypeCandidate {
                phenotype: cand.phenotype,
                literals: cand.literals,
                mask: cand.mask,
                support: cand.support,
                positives: cand.positives,
                precision: cand.precision,
                recall: cand.recall,
                lift: cand.lift,
                f1: cand.f1,
                frequency: cand.frequency,
                output_true: cand.output_true,
                output_false: cand.output_false,
                backend: "sirus".to_string(),
            })
            .collect())
    }

    fn generate_conjunction_candidates(
        &self,
        x_binary: &[Vec<u8>],
        subset_labels: &[u8],
    ) -> Result<Vec<PhenotypeCandidate>, String> {
        let n = x_binary.len();
        let subset_mask = BitMask::from_bool_iter(n, subset_labels.iter().map(|&v| v > 0));
        let total_subset = subset_mask.count_ones();
        if total_subset == 0 {
            return Ok(Vec::new());
        }
        let base_rate = total_subset as f64 / n as f64;

        let literals = self.build_literals(x_binary)?;
        let mut candidates = Vec::<ConjunctionCandidate>::new();
        let mut prefix = Vec::<usize>::new();
        let search = ConjunctionSearch {
            literals: &literals,
            subset_mask: &subset_mask,
            total_subset,
            base_rate,
        };
        self.enumerate_conjunctions(&search, None, 0, &mut prefix, &mut candidates);
        candidates.sort_by(|a, b| {
            b.lift
                .total_cmp(&a.lift)
                .then_with(|| b.f1.total_cmp(&a.f1))
                .then_with(|| b.support.cmp(&a.support))
                .then_with(|| a.literals.len().cmp(&b.literals.len()))
        });
        if candidates.len() > self.max_candidates {
            candidates.truncate(self.max_candidates);
        }
        Ok(candidates
            .into_iter()
            .map(|cand| {
                let outside_support = n - cand.support;
                let outside_pos = total_subset - cand.positives;
                let output_false = if outside_support > 0 {
                    outside_pos as f64 / outside_support as f64
                } else {
                    0.0
                };
                PhenotypeCandidate {
                    phenotype: render_conjunction(&self.feature_names, &cand.literals),
                    literals: cand.literals,
                    mask: cand.mask,
                    support: cand.support,
                    positives: cand.positives,
                    precision: cand.precision,
                    recall: cand.recall,
                    lift: cand.lift,
                    f1: cand.f1,
                    frequency: 1.0,
                    output_true: cand.precision,
                    output_false,
                    backend: "conjunction".to_string(),
                }
            })
            .collect())
    }

    fn build_literals(&self, x_binary: &[Vec<u8>]) -> Result<Vec<LiteralMask>, String> {
        let n = x_binary.len();
        let p = x_binary[0].len();
        let mut out = Vec::new();
        for feat in 0..p {
            let positive_mask =
                BitMask::from_bool_iter(n, x_binary.iter().map(|row| row[feat] > 0));
            let pos_support = positive_mask.count_ones();
            let pos_prev = pos_support as f64 / n as f64;
            if pos_support >= self.min_support && pos_prev <= self.max_literal_prevalence {
                out.push(LiteralMask {
                    literal: Literal {
                        feature_idx: feat,
                        positive: true,
                    },
                    mask: positive_mask.clone(),
                    support: pos_support,
                });
            }
            if self.include_negations {
                let negative_mask = positive_mask.not();
                let neg_support = negative_mask.count_ones();
                let neg_prev = neg_support as f64 / n as f64;
                if neg_support >= self.min_support && neg_prev <= self.max_literal_prevalence {
                    out.push(LiteralMask {
                        literal: Literal {
                            feature_idx: feat,
                            positive: false,
                        },
                        mask: negative_mask,
                        support: neg_support,
                    });
                }
            }
        }
        out.sort_by(|a, b| a.support.cmp(&b.support));
        Ok(out)
    }

    fn enumerate_conjunctions(
        &self,
        search: &ConjunctionSearch<'_>,
        current_mask: Option<BitMask>,
        start_idx: usize,
        prefix: &mut Vec<usize>,
        out: &mut Vec<ConjunctionCandidate>,
    ) {
        for idx in start_idx..search.literals.len() {
            let literal = &search.literals[idx];
            if prefix
                .iter()
                .any(|&j| search.literals[j].literal.feature_idx == literal.literal.feature_idx)
            {
                continue;
            }
            let next_mask = match &current_mask {
                Some(mask) => mask.and(&literal.mask),
                None => literal.mask.clone(),
            };
            let support = next_mask.count_ones();
            if support < self.min_support {
                continue;
            }
            let positives = next_mask.and_count(search.subset_mask);
            if positives == 0 {
                continue;
            }
            let precision = positives as f64 / support as f64;
            let recall = positives as f64 / search.total_subset as f64;
            let lift = precision / search.base_rate;
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            prefix.push(idx);
            if lift >= self.min_lift {
                out.push(ConjunctionCandidate {
                    literals: prefix
                        .iter()
                        .map(|&j| search.literals[j].literal.clone())
                        .collect(),
                    mask: next_mask.clone(),
                    support,
                    positives,
                    precision,
                    recall,
                    lift,
                    f1,
                });
            }
            if prefix.len() < self.max_depth {
                self.enumerate_conjunctions(search, Some(next_mask), idx + 1, prefix, out);
            }
            prefix.pop();
        }
    }

    fn score_candidates(
        &self,
        candidates: Vec<PhenotypeCandidate>,
        samplewise_losses: &[f64],
        y_true: &[u8],
        y_score: &[f64],
        performance_threshold: f64,
    ) -> Vec<ScoredRule> {
        let n = y_true.len();
        let mut scored = Vec::<ScoredRule>::new();
        for (cand_idx, cand) in candidates.iter().enumerate() {
            let indices = cand.mask.indices();
            if indices.is_empty() || indices.len() >= n {
                continue;
            }
            let mut in_loss = Vec::with_capacity(indices.len());
            let mut out_loss = Vec::with_capacity(n - indices.len());
            for (i, &loss) in samplewise_losses.iter().enumerate().take(n) {
                if cand.mask.get(i) {
                    in_loss.push(loss);
                } else {
                    out_loss.push(loss);
                }
            }
            if in_loss.is_empty() || out_loss.is_empty() {
                continue;
            }
            let effect_size = cohens_d(&in_loss, &out_loss);
            let seed = self
                .random_state
                .wrapping_add(cand_idx as u64)
                .wrapping_add(101);
            let Some((auc, low, high, boot)) = bootstrap_auc_ci(
                y_true,
                y_score,
                &indices,
                self.bootstrap_samples,
                0.95,
                seed,
            ) else {
                continue;
            };
            let p_value = z_test_less(&boot, performance_threshold);
            scored.push(ScoredRule {
                phenotype: cand.phenotype.clone(),
                literals: cand.literals.clone(),
                support: cand.support,
                positives: cand.positives,
                precision: cand.precision,
                recall: cand.recall,
                lift: cand.lift,
                f1: cand.f1,
                frequency: cand.frequency,
                output_true: cand.output_true,
                output_false: cand.output_false,
                backend: cand.backend.clone(),
                auroc: auc,
                auroc_lower: low,
                auroc_upper: high,
                p_value,
                effect_size,
            });
        }
        scored
    }
}

fn holm_bonferroni(mut rules: Vec<ScoredRule>, alpha: f64) -> Vec<ScoredRule> {
    rules.sort_by(|a, b| a.p_value.total_cmp(&b.p_value));
    let m = rules.len();
    let mut out = Vec::new();
    for (rank, rule) in rules.into_iter().enumerate() {
        let cutoff = alpha / (m - rank) as f64;
        if rule.p_value < cutoff {
            out.push(rule);
        } else {
            break;
        }
    }
    out
}
