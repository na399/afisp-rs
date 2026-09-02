mod bitset;
mod metrics;
mod phenotype;
mod prng;
mod rules;
mod sirus;
mod stability;
mod stats;

pub use phenotype::{PhenotypeFitInput, ScoredRule, SubgroupPhenotyperCore};
pub use stability::{WorstSubsetFinderCore, SubsetAucRow};

#[cfg(feature = "extension-module")]
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "extension-module")]
use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::types::{PyDict, PyList, PyModule};

#[cfg(feature = "extension-module")]
use crate::metrics::{
    brier_loss as brier_loss_impl, cross_entropy_loss as cross_entropy_loss_impl,
    roc_auc_score as roc_auc_score_impl, zero_one_loss as zero_one_loss_impl,
};

#[cfg(feature = "extension-module")]
fn as_vec_u8(arr: PyReadonlyArray1<'_, u8>) -> PyResult<Vec<u8>> {
    Ok(arr.as_slice()?.to_vec())
}

#[cfg(feature = "extension-module")]
fn as_vec_f64(arr: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    Ok(arr.as_slice()?.to_vec())
}

#[cfg(feature = "extension-module")]
fn as_vec_vec_u8(arr: PyReadonlyArray2<'_, u8>) -> PyResult<Vec<Vec<u8>>> {
    let view = arr.as_array();
    Ok(view.rows().into_iter().map(|row| row.to_vec()).collect())
}

#[cfg(feature = "extension-module")]
#[pyclass(name = "WorstSubsetFinder")]
pub struct PyWorstSubsetFinder {
    core: WorstSubsetFinderCore,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyWorstSubsetFinder {
    #[new]
    #[pyo3(signature = (subset_fractions=None, eps=0.0, random_state=0))]
    fn new(subset_fractions: Option<Vec<f64>>, eps: f64, random_state: u64) -> PyResult<Self> {
        let fractions =
            subset_fractions.unwrap_or_else(|| vec![0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0]);
        if eps < 0.0 {
            return Err(PyValueError::new_err("eps must be >= 0"));
        }
        Ok(Self {
            core: WorstSubsetFinderCore::new(fractions, eps, random_state),
        })
    }

    #[getter]
    fn subset_fractions(&self) -> Vec<f64> {
        self.core.subset_fractions.clone()
    }

    fn fit_from_mu_hat(
        &mut self,
        samplewise_losses: PyReadonlyArray1<'_, f64>,
        mu_hat: PyReadonlyArray1<'_, f64>,
        fold_ids: Option<PyReadonlyArray1<'_, i64>>,
    ) -> PyResult<Vec<f64>> {
        let losses = as_vec_f64(samplewise_losses)?;
        let mu = as_vec_f64(mu_hat)?;
        let folds = match fold_ids {
            Some(f) => Some(
                f.as_slice()?
                    .iter()
                    .map(|&x| {
                        usize::try_from(x)
                            .map_err(|_| PyValueError::new_err("fold_ids must be non-negative"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            None => None,
        };
        self.core
            .fit_from_mu_hat(losses, mu, folds)
            .map_err(PyValueError::new_err)
    }

    fn confidence_intervals(&self) -> PyResult<Vec<(f64, f64)>> {
        if !self.core.fit_called {
            return Err(PyRuntimeError::new_err("call fit_from_mu_hat first"));
        }
        Ok(self.core.cis.clone())
    }

    fn subset_masks(&self) -> PyResult<Vec<Vec<bool>>> {
        if !self.core.fit_called {
            return Err(PyRuntimeError::new_err("call fit_from_mu_hat first"));
        }
        Ok(self.core.masks.clone())
    }

    fn observed_fractions(&self) -> PyResult<Vec<f64>> {
        self.core
            .observed_fractions()
            .map_err(PyRuntimeError::new_err)
    }

    fn compute_effect_sizes(&mut self) -> PyResult<Vec<f64>> {
        self.core
            .compute_effect_sizes()
            .map_err(PyRuntimeError::new_err)
    }

    fn find_max_effect_size(&mut self) -> PyResult<(usize, f64)> {
        self.core
            .find_max_effect_size()
            .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (metric_values, threshold, smaller_is_worse=true))]
    fn select_largest_below_threshold(
        &self,
        metric_values: PyReadonlyArray1<'_, f64>,
        threshold: f64,
        smaller_is_worse: bool,
    ) -> PyResult<Option<(usize, f64, f64)>> {
        let values = as_vec_f64(metric_values)?;
        self.core
            .select_largest_below_threshold(&values, threshold, smaller_is_worse)
            .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (y_true, y_score, n_bootstrap=100, confidence=0.95))]
    fn subset_auc_table(
        &self,
        py: Python<'_>,
        y_true: PyReadonlyArray1<'_, u8>,
        y_score: PyReadonlyArray1<'_, f64>,
        n_bootstrap: usize,
        confidence: f64,
    ) -> PyResult<Py<PyAny>> {
        let yt = as_vec_u8(y_true)?;
        let ys = as_vec_f64(y_score)?;
        let rows = self
            .core
            .subset_auc_table(&yt, &ys, n_bootstrap, confidence)
            .map_err(PyRuntimeError::new_err)?;
        let out = PyList::empty(py);
        for (fraction, n, auc, low, high) in rows {
            let d = PyDict::new(py);
            d.set_item("subset_fraction", fraction)?;
            d.set_item("n", n)?;
            d.set_item("auroc", auc)?;
            d.set_item("lower", low)?;
            d.set_item("upper", high)?;
            out.append(d)?;
        }
        Ok(out.into_any().unbind())
    }
}

#[cfg(feature = "extension-module")]
#[pyclass(name = "SubgroupPhenotyper")]
pub struct PySubgroupPhenotyper {
    core: SubgroupPhenotyperCore,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySubgroupPhenotyper {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        backend="sirus",
        max_depth=3,
        min_support=50,
        max_candidates=5000,
        include_negations=true,
        max_literal_prevalence=0.95,
        min_lift=1.0,
        effect_threshold=0.4,
        bootstrap_samples=100,
        random_state=0,
        sirus_num_trees=300,
        sirus_p0=None,
        sirus_num_rule=None,
        sirus_mtry=None,
        sirus_min_samples_leaf=25,
        sirus_sample_fraction=1.0,
        sirus_replace=true
    ))]
    fn new(
        backend: &str,
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
    ) -> PyResult<Self> {
        Ok(Self {
            core: SubgroupPhenotyperCore::new(
                backend.to_string(),
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
            ),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        x_binary: PyReadonlyArray2<'_, u8>,
        subset_labels: PyReadonlyArray1<'_, u8>,
        samplewise_losses: PyReadonlyArray1<'_, f64>,
        feature_names: Vec<String>,
        y_true: PyReadonlyArray1<'_, u8>,
        y_score: PyReadonlyArray1<'_, f64>,
        performance_threshold: f64,
    ) -> PyResult<Vec<String>> {
        let x = as_vec_vec_u8(x_binary)?;
        let labels = as_vec_u8(subset_labels)?;
        let losses = as_vec_f64(samplewise_losses)?;
        let yt = as_vec_u8(y_true)?;
        let ys = as_vec_f64(y_score)?;
        self.core
            .fit(PhenotypeFitInput {
                x_binary: &x,
                subset_labels: &labels,
                samplewise_losses: &losses,
                feature_names,
                y_true: &yt,
                y_score: &ys,
                performance_threshold,
            })
            .map_err(PyValueError::new_err)
    }

    fn selected_rules(&self) -> PyResult<Vec<String>> {
        if !self.core.fit_called {
            return Err(PyRuntimeError::new_err("call fit first"));
        }
        Ok(self
            .core
            .selected_rules
            .iter()
            .map(|r| r.phenotype.clone())
            .collect())
    }

    #[getter]
    fn selected_p0(&self) -> f64 {
        self.core.sirus_p0.unwrap_or(0.025)
    }

    fn fit_metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("backend", &self.core.backend)?;
        d.set_item("selected_p0", self.core.sirus_p0.unwrap_or(0.025))?;
        d.set_item("sirus_num_trees", self.core.sirus_num_trees)?;
        d.set_item("sirus_num_rule", self.core.sirus_num_rule)?;
        d.set_item("sirus_mtry", self.core.sirus_mtry)?;
        d.set_item("sirus_min_samples_leaf", self.core.sirus_min_samples_leaf)?;
        d.set_item("sirus_sample_fraction", self.core.sirus_sample_fraction)?;
        d.set_item("sirus_replace", self.core.sirus_replace)?;
        d.set_item("selected_rule_count", self.core.selected_rules.len())?;
        Ok(d.into_any().unbind())
    }

    fn summary_table(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if !self.core.fit_called {
            return Err(PyRuntimeError::new_err("call fit first"));
        }
        let out = PyList::empty(py);
        for rule in &self.core.selected_rules {
            let d = PyDict::new(py);
            let literal_indices = rule
                .literals
                .iter()
                .map(|literal| literal.feature_idx)
                .collect::<Vec<_>>();
            let literal_names = rule
                .literals
                .iter()
                .map(|literal| {
                    self.core
                        .feature_names
                        .get(literal.feature_idx)
                        .cloned()
                        .unwrap_or_else(|| format!("x{}", literal.feature_idx))
                })
                .collect::<Vec<_>>();
            let literal_signs = rule
                .literals
                .iter()
                .map(|literal| literal.positive)
                .collect::<Vec<_>>();
            d.set_item("phenotype", &rule.phenotype)?;
            d.set_item("literal_indices", literal_indices)?;
            d.set_item("literal_names", literal_names)?;
            d.set_item("literal_signs", literal_signs)?;
            d.set_item("literal_count", rule.literals.len())?;
            d.set_item("support", rule.support)?;
            d.set_item("subset_positives", rule.positives)?;
            d.set_item("precision", rule.precision)?;
            d.set_item("recall", rule.recall)?;
            d.set_item("lift", rule.lift)?;
            d.set_item("f1", rule.f1)?;
            d.set_item("frequency", rule.frequency)?;
            d.set_item("output_true", rule.output_true)?;
            d.set_item("output_false", rule.output_false)?;
            d.set_item("backend", &rule.backend)?;
            d.set_item("auroc", rule.auroc)?;
            d.set_item("lower", rule.auroc_lower)?;
            d.set_item("upper", rule.auroc_upper)?;
            d.set_item("p_value", rule.p_value)?;
            d.set_item("effect_size", rule.effect_size)?;
            out.append(d)?;
        }
        Ok(out.into_any().unbind())
    }
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn cross_entropy_loss(
    y_true: PyReadonlyArray1<'_, u8>,
    y_pred: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    let yt = as_vec_u8(y_true)?;
    let yp = as_vec_f64(y_pred)?;
    if yt.len() != yp.len() {
        return Err(PyValueError::new_err(
            "y_true and y_pred must have the same length",
        ));
    }
    Ok(cross_entropy_loss_impl(&yt, &yp))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn brier_loss(
    y_true: PyReadonlyArray1<'_, u8>,
    y_pred: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    let yt = as_vec_u8(y_true)?;
    let yp = as_vec_f64(y_pred)?;
    if yt.len() != yp.len() {
        return Err(PyValueError::new_err(
            "y_true and y_pred must have the same length",
        ));
    }
    Ok(brier_loss_impl(&yt, &yp))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn zero_one_loss(
    y_true: PyReadonlyArray1<'_, u8>,
    y_pred_binary: PyReadonlyArray1<'_, u8>,
) -> PyResult<Vec<f64>> {
    let yt = as_vec_u8(y_true)?;
    let yp = as_vec_u8(y_pred_binary)?;
    if yt.len() != yp.len() {
        return Err(PyValueError::new_err(
            "y_true and y_pred_binary must have the same length",
        ));
    }
    Ok(zero_one_loss_impl(&yt, &yp))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn roc_auc_score(
    y_true: PyReadonlyArray1<'_, u8>,
    y_score: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let yt = as_vec_u8(y_true)?;
    let ys = as_vec_f64(y_score)?;
    if yt.len() != ys.len() {
        return Err(PyValueError::new_err(
            "y_true and y_score must have the same length",
        ));
    }
    roc_auc_score_impl(&yt, &ys).ok_or_else(|| {
        PyValueError::new_err("roc_auc_score requires both positive and negative labels")
    })
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorstSubsetFinder>()?;
    m.add_class::<PySubgroupPhenotyper>()?;
    m.add_function(wrap_pyfunction!(cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(brier_loss, m)?)?;
    m.add_function(wrap_pyfunction!(zero_one_loss, m)?)?;
    m.add_function(wrap_pyfunction!(roc_auc_score, m)?)?;
    Ok(())
}
