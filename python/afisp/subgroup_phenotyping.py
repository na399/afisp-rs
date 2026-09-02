from __future__ import annotations

import logging
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.stats.weightstats import ttest_ind
from tqdm import tqdm

from afisp.utils import bootstrap_ci, cohens_d
from afisp_rs import SubgroupPhenotyper as _RustSubgroupPhenotyper

logger = logging.getLogger(__name__)


def _r_sirus_available() -> bool:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False
    try:
        proc = subprocess.run(
            [rscript, "-e", 'requireNamespace("sirus", quietly=TRUE)'],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0
    except OSError:
        return False


class SubgroupPhenotyper(BaseEstimator):
    """Sklearn-compatible subgroup phenotyping with Rust-native or R SIRUS backends."""

    def __init__(self):
        self.fit_called_ = False
        self._rust_backend = False
        self._rust_phenotyper = None

    def fit(
        self,
        subgroup_feature_data,
        subset_labels,
        test_loss,
        method="DecisionList",
        depth=2,
        cv=False,
        rule_max=50,
        p0=0.025,
        input_fname="data_for_sirus.csv",
        output_fname="sirus_rules.txt",
        verbose=0,
        *,
        y_true=None,
        y_score=None,
        performance_threshold=None,
        random_state=0,
    ):
        phenotype_df = subgroup_feature_data.copy()
        if not isinstance(phenotype_df, pd.DataFrame):
            phenotype_df = pd.DataFrame(phenotype_df)
        phenotype_df["subset_label"] = subset_labels
        self._phenotype_df = phenotype_df

        use_r_sirus = method == "SIRUS" and _r_sirus_available() and not cv
        if method == "SIRUS" and cv:
            warnings.warn(
                "SIRUS cross-validation for p0 is not implemented in afisp-rs; "
                "using Rust-native SIRUS with fixed p0.",
                stacklevel=2,
            )
        if method == "SIRUS" and not use_r_sirus:
            if not _r_sirus_available():
                warnings.warn(
                    "R SIRUS unavailable; using Rust-native SIRUS backend.",
                    stacklevel=2,
                )
            self._fit_rust_backend(
                phenotype_df,
                subset_labels,
                test_loss,
                method=method,
                depth=depth,
                p0=p0,
                rule_max=rule_max,
                y_true=y_true,
                y_score=y_score,
                performance_threshold=performance_threshold,
                random_state=random_state,
            )
        elif method == "SIRUS":
            self._fit_r_sirus(
                phenotype_df,
                test_loss,
                depth=depth,
                p0=p0,
                rule_max=rule_max,
                input_fname=input_fname,
                output_fname=output_fname,
                verbose=verbose,
            )
        elif method == "DecisionList":
            self._fit_rust_backend(
                phenotype_df,
                subset_labels,
                test_loss,
                method=method,
                depth=depth,
                p0=p0,
                rule_max=rule_max,
                y_true=y_true,
                y_score=y_score,
                performance_threshold=performance_threshold,
                random_state=random_state,
            )
        else:
            raise RuntimeError('Method not implemented. Please choose one of "SIRUS" or "DecisionList"')

        self.fit_called_ = True
        return self._extracted_rules

    def _fit_rust_backend(
        self,
        phenotype_df,
        subset_labels,
        test_loss,
        *,
        method,
        depth,
        p0,
        rule_max,
        y_true,
        y_score,
        performance_threshold,
        random_state,
    ):
        if y_true is None or y_score is None or performance_threshold is None:
            raise ValueError(
                "Rust-backed phenotyping requires keyword arguments "
                "y_true, y_score, and performance_threshold."
            )
        backend = "sirus" if method == "SIRUS" else "conjunction"
        self._rust_phenotyper = _RustSubgroupPhenotyper(
            backend=backend,
            max_depth=int(depth),
            min_support=10,
            max_candidates=max(5000, int(rule_max) * 100),
            effect_threshold=0.3,
            bootstrap_samples=100,
            random_state=int(random_state),
            sirus_num_trees=300,
            sirus_p0=float(p0),
            sirus_num_rule=int(rule_max),
            sirus_min_samples_leaf=5,
        )
        feature_cols = [c for c in phenotype_df.columns if c != "subset_label"]
        x_df = phenotype_df[feature_cols]
        self._rust_phenotyper.fit(
            x_df,
            np.asarray(subset_labels, dtype=np.uint8),
            np.asarray(test_loss, dtype=np.float64),
            y_true=np.asarray(y_true, dtype=np.uint8),
            y_score=np.asarray(y_score, dtype=np.float64),
            performance_threshold=float(performance_threshold),
            feature_names=list(feature_cols),
        )
        self._rust_backend = True
        self._extracted_rules = self._rust_phenotyper.selected_rules()

    def _fit_r_sirus(
        self,
        phenotype_df,
        test_loss,
        *,
        depth,
        p0,
        rule_max,
        input_fname,
        output_fname,
        verbose,
    ):
        phenotype_df.to_csv(input_fname, index=False)
        package_path = Path(__file__).parent
        command = (
            f"Rscript {package_path / 'run_sirus.r'}"
            f" --input {input_fname} --output {output_fname}"
            f" --depth {depth} --rule.max {rule_max} --p0 {p0}"
        )
        if verbose > 0:
            print("Beginning call to SIRUS.")
        subprocess.call(command, shell=True)
        if verbose > 0:
            print("Finished call to SIRUS")
        candidate_rules = self._get_sirus_rules(output_fname)
        if os.path.exists(output_fname):
            os.remove(output_fname)
        if os.path.exists(input_fname):
            os.remove(input_fname)
        rule_p_values = self._precompute_p_values(candidate_rules, phenotype_df, test_loss)
        significant_rules = self._holm_bonferroni_correction(rule_p_values)
        self._extracted_rules = self._effect_size_filtering(
            significant_rules,
            phenotype_df,
            test_loss,
            effect_threshold=0.3,
        )
        self._rust_backend = False

    def generate_subgroup_table(self, y_test, test_preds, loss_fn=brier_score_loss):
        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on SubgroupPhenotyper object first.')

        if self._rust_backend:
            table = self._rust_phenotyper.summary_table()
            if hasattr(table, "rename"):
                return table.rename(
                    columns={
                        "phenotype": "Phenotype",
                        "auroc": "Performance",
                        "support": "N",
                        "lower": "Lower",
                        "upper": "Upper",
                    }
                ).sort_values(by="Performance")
            return table

        r_aucs = []
        r_ls = []
        r_us = []
        ns = []
        for rule in self._extracted_rules:
            rows = self._phenotype_df.eval(str(rule))
            ns.append(np.sum(rows))
            m, l, u = bootstrap_ci(y_test[rows], test_preds[rows], loss=loss_fn)
            r_aucs.append(m)
            r_ls.append(l)
            r_us.append(u)

        return pd.DataFrame(
            {
                "Phenotype": self._extracted_rules,
                "Performance": r_aucs,
                "N": ns,
                "Lower": r_ls,
                "Upper": r_us,
            }
        ).sort_values(by="Performance")

    def _negate_simple_rule(self, rule):
        if ">=" in rule:
            return rule.replace(">=", "<")
        if "<=" in rule:
            return rule.replace("<=", ">")
        if ">" in rule:
            return rule.replace(">", "<=")
        if "<" in rule:
            return rule.replace("<", ">=")
        return rule

    def _get_sirus_rules(self, filename):
        with open(filename) as f:
            filelines = [line for line in f]
        sirus_rules = set()
        for line in filelines:
            if " then " not in line:
                continue
            end = line.index(" then ")
            rule = line[3:end].strip()
            if_prob = float(line.split("then")[1].split()[0])
            else_prob = float(line.split("else")[1].split()[0])
            if if_prob > else_prob:
                sirus_rules.add(rule)
            elif "&" in rule:
                for r in rule.split("&"):
                    sirus_rules.add(self._negate_simple_rule(r.strip()))
            else:
                sirus_rules.add(self._negate_simple_rule(rule))
        return sirus_rules

    def _precompute_p_values(self, sirus_rules, phenotype_df, test_loss, alpha=0.05):
        rule_p_values = []
        for rule in tqdm(sirus_rules):
            rows = phenotype_df.eval(str(rule))
            pval = ttest_ind(
                test_loss[rows],
                x2=test_loss[~rows],
                value=0.0,
                alternative="larger",
                usevar="unequal",
            )[1]
            rule_p_values.append((rule, pval, rows.sum()))
        rule_p_values = sorted([x for x in rule_p_values if not np.isnan(x[1])], key=lambda x: x[1])
        return rule_p_values

    def _holm_bonferroni_correction(self, rule_p_values, sig_value=0.05):
        m = len(rule_p_values)
        significant_rules = []
        for k in range(1, m + 1):
            rule_k = rule_p_values[k - 1]
            if rule_k[1] < sig_value / (m + 1 - k):
                significant_rules.append(rule_k)
                continue
            break
        return significant_rules

    def _effect_size_filtering(
        self,
        significant_rules,
        phenotype_df,
        test_loss,
        effect_threshold=0.3,
        verbose=False,
    ):
        rules = []
        for rule in tqdm(significant_rules):
            r = rule[0]
            rows = phenotype_df.eval(str(r))
            cd = cohens_d(test_loss[rows], test_loss[~rows])
            if verbose:
                print(f"{r} Cohen's d {cd:.2f}")
            if cd > effect_threshold:
                rules.append((r, cd))
        rules = sorted(rules, key=lambda x: x[1], reverse=True)
        return [r[0] for r in rules]
