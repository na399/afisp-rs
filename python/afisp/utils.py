from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from afisp_rs import (
    brier_loss as _brier_loss,
    cross_entropy_loss as _cross_entropy_loss,
    roc_auc_score as _roc_auc_score,
    zero_one_loss as _zero_one_loss,
)


def clip_predictions(preds, upper_bound=0.99, lower_bound=0.01):
    if upper_bound >= 1.0 or lower_bound <= 0.0:
        raise RuntimeError("upper_bound must be < 1 and lower_bound must be > 0")
    new_preds = np.copy(preds)
    one_inds = np.where(preds > upper_bound)[0]
    zero_inds = np.where(preds < lower_bound)[0]
    new_preds[one_inds] = np.repeat(upper_bound, one_inds.shape[0])
    new_preds[zero_inds] = np.repeat(lower_bound, zero_inds.shape[0])
    return new_preds


def _as_u8(y):
    return np.asarray(y, dtype=np.uint8).reshape(-1)


def _as_f64(y):
    return np.asarray(y, dtype=np.float64).reshape(-1)


def cross_entropy(y, y_pred):
    y_arr = _as_u8(y)
    y_pred_arr = _as_f64(clip_predictions(y_pred))
    return np.asarray(_cross_entropy_loss(y_arr, y_pred_arr), dtype=np.float64)


def brier(y, y_pred):
    y_arr = _as_u8(y)
    y_pred_arr = _as_f64(y_pred)
    return np.asarray(_brier_loss(y_arr, y_pred_arr), dtype=np.float64)


def zero_one_loss(y, y_pred):
    y_arr = _as_u8(y)
    y_pred_arr = _as_u8(y_pred)
    return np.asarray(_zero_one_loss(y_arr, y_pred_arr), dtype=np.float64)


def mse(y, y_pred):
    y_arr = _as_f64(y)
    y_pred_arr = _as_f64(y_pred)
    return (y_arr - y_pred_arr) ** 2


def entropy(y, y_pred):
    y_pred_arr = _as_f64(y_pred)
    return -np.log(y_pred_arr)


def logit(p):
    clipped = clip_predictions(p)
    return np.log(clipped / (1.0 - clipped))


def hinge_surrogate(labels, logits):
    positives_term = labels * np.maximum(1.0 - logits, 0)
    negatives_term = (1.0 - labels) * np.maximum(1.0 + logits, 0)
    return positives_term + negatives_term


def xent_surrogate(labels, logits):
    softplus_term = np.maximum(-logits, 0.0) + np.log(1.0 + np.exp(-np.abs(logits)))
    return logits - labels * logits + softplus_term


def torch_roc_auc_surrogate(y, y_pred, surrogate="xent"):
    import torch

    y_torch = torch.tensor(y)
    logits_torch = torch.tensor(logit(y_pred))
    logits_difference_torch = logits_torch.unsqueeze(0) - logits_torch.unsqueeze(1)
    labels_difference_torch = y_torch.unsqueeze(0) - y_torch.unsqueeze(1)
    abs_label_difference = torch.abs(labels_difference_torch)
    signed_logits_difference_torch = logits_difference_torch * labels_difference_torch
    if surrogate == "xent":
        loss = torch.log(torch.sigmoid(signed_logits_difference_torch))
        loss = (abs_label_difference * loss).mean(axis=0) * 0.5
    elif surrogate == "hinge":
        loss = torch.maximum(torch.zeros(1), torch.ones(1) - signed_logits_difference_torch)
        loss = (abs_label_difference * loss).mean(axis=0) * 0.5
    else:
        raise ValueError(f"unknown surrogate: {surrogate}")
    return np.array(loss.tolist())


def roc_auc_surrogate(y, y_pred, surrogate="xent"):
    pos_mask = y == 1
    neg_mask = y == 0
    if (np.sum(pos_mask) == 0) or (np.sum(neg_mask) == 0):
        raise Exception("Examples are either all positive or all negative")
    logits = logit(y_pred)
    logits_difference = np.expand_dims(logits, 0) - np.expand_dims(logits, 1)
    labels_difference = np.expand_dims(y, 0) - np.expand_dims(y, 1)
    signed_logits_difference = labels_difference * logits_difference
    if surrogate == "hinge":
        surr_fn = hinge_surrogate
    elif surrogate == "xent":
        surr_fn = xent_surrogate
    else:
        raise ValueError(f"unknown surrogate: {surrogate}")
    surrogate_loss = surr_fn(np.ones_like(signed_logits_difference), signed_logits_difference)
    proxy_auc_loss = np.abs(labels_difference) * surrogate_loss
    return proxy_auc_loss


def bootstrap_ci(
    y_true,
    y_pred,
    n_bootstrap=100,
    confidence=0.95,
    loss=roc_auc_score,
    return_samples=False,
):
    n = y_true.shape[0]
    upper_p = 100 * (1.0 - (1.0 - confidence) / 2)
    lower_p = 100 * ((1.0 - confidence) / 2)
    aucs = []

    def bootstrap_resample_inds():
        return np.array(np.random.choice(range(n), n, replace=True))

    for _ in range(n_bootstrap):
        inds = bootstrap_resample_inds()
        resample_true = y_true[inds]
        resample_pred = y_pred[inds]
        if loss == roc_auc_score:
            if (resample_true.mean() == 1) or (resample_true.mean() == 0):
                continue
        aucs.append(loss(resample_true, resample_pred))

    lower, upper = np.percentile(aucs, [lower_p, upper_p])
    if return_samples:
        return aucs
    return np.mean(aucs), lower, upper


def cohens_d(x, y):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    nx = len(x_arr)
    ny = len(y_arr)
    dof = nx + ny - 2
    return (np.mean(x_arr) - np.mean(y_arr)) / np.sqrt(
        ((nx - 1) * np.std(x_arr, ddof=1) ** 2 + (ny - 1) * np.std(y_arr, ddof=1) ** 2) / dof
    )
