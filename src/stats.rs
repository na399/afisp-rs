pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

pub fn variance_population(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
}

pub fn variance_sample(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0)
}

pub fn std_sample(xs: &[f64]) -> f64 {
    variance_sample(xs).sqrt()
}

pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let q = q.clamp(0.0, 1.0);
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() as f64 - 1.0);
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let w = pos - lower as f64;
        sorted[lower] * (1.0 - w) + sorted[upper] * w
    }
}

pub fn cohens_d(x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return f64::NAN;
    }
    let nx = x.len() as f64;
    let ny = y.len() as f64;
    if nx + ny <= 2.0 {
        return 0.0;
    }
    let vx = variance_sample(x);
    let vy = variance_sample(y);
    let pooled = (((nx - 1.0) * vx) + ((ny - 1.0) * vy)) / (nx + ny - 2.0);
    if pooled <= 0.0 {
        return 0.0;
    }
    (mean(x) - mean(y)) / pooled.sqrt()
}

// Abramowitz-Stegun style approximation of erf, enough for p-values.
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * (-x * x).exp();
    sign * y
}

pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

pub fn z_test_less(samples: &[f64], threshold: f64) -> f64 {
    if samples.is_empty() {
        return 1.0;
    }
    let m = mean(samples);
    let sd = std_sample(samples);
    if sd == 0.0 {
        return if m < threshold { 0.0 } else { 1.0 };
    }
    let se = sd / (samples.len() as f64).sqrt();
    let z = (m - threshold) / se;
    normal_cdf(z)
}

#[cfg(test)]
mod tests {
    use super::{cohens_d, mean, normal_cdf, quantile_sorted};

    #[test]
    fn quantile_linear_interp() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert!((quantile_sorted(&v, 0.25) - 1.75).abs() < 1e-8);
        assert!((quantile_sorted(&v, 0.5) - 2.5).abs() < 1e-8);
    }

    #[test]
    fn cdf_reasonable() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!(normal_cdf(-3.0) < 0.01);
        assert!(normal_cdf(3.0) > 0.99);
    }

    #[test]
    fn effect_size_basic() {
        let x = vec![2.0, 2.0, 2.0, 2.0];
        let y = vec![1.0, 1.0, 1.0, 1.0];
        let d = cohens_d(&x, &y);
        assert!(d >= 0.0 || d.is_nan());
        assert!((mean(&x) - 2.0).abs() < 1e-8);
    }
}
