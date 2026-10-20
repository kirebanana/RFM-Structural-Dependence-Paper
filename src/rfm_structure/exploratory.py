"""Paired panel sensitivity checks, explicitly exploratory and non-causal."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def rank_credit(labels, scores):
    """AUROC's positive/negative pair credits, including half-credit ties."""
    labels = np.asarray(labels) > 0
    scores = np.asarray(scores, dtype=float)
    if scores.shape != labels.shape or not np.isfinite(scores).all() or labels.all() or not labels.any():
        raise ValueError("rank credits require finite paired scores and both classes")
    difference = scores[labels, None] - scores[None, ~labels]
    return (difference > 0).astype(float) + 0.5 * (difference == 0)


def clustered_auc_delta(labels, clean, arms, dates, drivers, *, replicates=2000, seed=20261006):
    """Paired percentile sensitivity intervals conditional on saved matching seeds.

    Resample dates, drivers, and crossed date×driver weights separately. Crossed
    resampling multiplies independent cluster multiplicities (a pigeonhole-style
    sensitivity check). It does not prove coverage under temporal dependence,
    multiplicity, or unseen tasks. The same weights are used in every arm.
    """
    labels = np.asarray(labels) > 0
    arms = np.asarray(arms)
    if replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    if arms.ndim != 2 or arms.shape[1] != len(labels) or len(arms) == 0:
        raise ValueError("arms must be seed-by-query arrays")
    clean = np.asarray(clean)
    references = clean[None, :] if clean.ndim == 1 else clean
    if references.ndim != 2 or references.shape[1] != len(labels):
        raise ValueError("clean references must be query or seed-by-query arrays")
    delta = np.mean([rank_credit(labels, arm) for arm in arms], axis=0) - np.mean(
        [rank_credit(labels, reference) for reference in references], axis=0
    )
    result = {"point": float(delta.mean()), "replicates": replicates}
    codes = {}
    counts = {}
    rng = np.random.default_rng(seed)
    for name, groups in (("date", dates), ("driver", drivers)):
        groups = np.asarray(groups)
        if groups.shape != labels.shape:
            raise ValueError("clusters must have one entry per query")
        unique, codes[name] = np.unique(groups, return_inverse=True)
        counts[name] = rng.multinomial(len(unique), np.ones(len(unique)) / len(unique), size=replicates)
    weights = {name: counts[name][:, codes[name]] for name in codes}
    weights["crossed"] = weights["date"] * weights["driver"]
    for name, weight in weights.items():
        if name in codes and len(np.unique(codes[name])) < 2:
            result[f"{name}_percentile_95"] = None
            result[f"{name}_defined_replicates"] = 0
            result[f"{name}_status"] = "single_cluster_not_estimable"
            continue
        numerator = np.sum((weight[:, labels] @ delta) * weight[:, ~labels], axis=1)
        denominator = weight[:, labels].sum(axis=1) * weight[:, ~labels].sum(axis=1)
        defined = denominator > 0
        estimates = numerator[defined] / denominator[defined]
        result[f"{name}_percentile_95"] = np.quantile(estimates, [0.025, 0.975]).tolist()
        result[f"{name}_defined_replicates"] = int(defined.sum())
    result["crossed_status"] = ("two_axes" if all(len(np.unique(code)) >= 2 for code in codes.values())
                                else "only_one_varying_cluster_axis")
    for name, groups in (("date", np.asarray(dates)), ("driver", np.asarray(drivers))):
        estimates = []
        for group in np.unique(groups):
            kept = groups != group
            if len(np.unique(labels[kept])) < 2:
                continue
            estimates.append(float(np.mean([roc_auc_score(labels[kept], arm[kept]) for arm in arms])
                                   - np.mean([roc_auc_score(labels[kept], reference[kept]) for reference in references])))
        result[f"leave_one_{name}_range"] = [min(estimates), max(estimates)] if estimates else None
    return result


def grouped_sensitivity_prediction(features, sensitivity, groups, *, folds=5):
    """Fixed ridge model evaluated out-of-fold by date or driver, not by row.

    Scaling fits within each training fold. Return out-of-fold R² and error on
    log1p(mean absolute paired score change), not outcome prediction performance
    or causal attribution. No test-fold tuning or significance calculation.
    """
    features = np.asarray(features, dtype=float)
    response = np.log1p(np.asarray(sensitivity, dtype=float))
    groups = np.asarray(groups)
    if features.shape[0] != len(response) or groups.shape != response.shape or not np.isfinite(features).all():
        raise ValueError("features, sensitivity and clusters must be aligned")
    if np.any(response < 0) or not np.isfinite(response).all():
        raise ValueError("sensitivity must be finite and nonnegative")
    if len(np.unique(groups)) < 2:
        raise ValueError("grouped evaluation needs at least two distinct clusters")
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    predicted = np.empty(len(response))
    mean_baseline = np.empty(len(response))
    for train, test in splitter.split(features, response, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(features[train], response[train])
        predicted[test] = model.predict(features[test])
        mean_baseline[test] = response[train].mean()
    return {"out_of_fold_r2": float(r2_score(response, predicted)),
            "mean_absolute_log_error": float(mean_absolute_error(response, predicted)),
            "training_mean_baseline_r2": float(r2_score(response, mean_baseline)),
            "n_records": len(response), "n_clusters": len(np.unique(groups)),
            "response": "log1p(seed-mean absolute raw-score shift)", "ridge_alpha": 10.0}
