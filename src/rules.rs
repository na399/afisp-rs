#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Literal {
    pub feature_idx: usize,
    pub positive: bool,
}

pub fn sort_literals(literals: &mut [Literal]) {
    literals.sort_by_key(|lit| (lit.feature_idx, if lit.positive { 1u8 } else { 0u8 }));
}

pub fn render_conjunction(feature_names: &[String], literals: &[Literal]) -> String {
    literals
        .iter()
        .map(|lit| {
            let base = feature_names[lit.feature_idx].clone();
            if lit.positive {
                base
            } else {
                format!("NOT {}", base)
            }
        })
        .collect::<Vec<_>>()
        .join(" & ")
}
