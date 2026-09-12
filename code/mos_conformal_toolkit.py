"""
mos_conformal_toolkit.py — User-facing toolkit for MOS conformal prediction.

Provides three entry points:

  1. MOSConformalPredictor — full pipeline in one object:
       predictor = MOSConformalPredictor()
       predictor.fit_reference(ref_features)
       predictor.calibrate(cal_features, cal_mos_true, cal_mos_pred)
       df = predictor.predict(test_features, test_mos_pred)
       # df has columns: mos_pred, ci_low, ci_high, ood_score, interval_width

  2. BootstrapCI — compute CIs for any callable metric:
       bci = BootstrapCI(n_boot=2000, groups=my_groups)
       point, lo, hi = bci.compute(scores, abs_errors,
                                   metric_fn=lambda s, e: spearmanr(s, e).statistic)

  3. evaluate_mos() — one call from arrays to a metrics dict with CIs:
       results = evaluate_mos(
           mos_pred=..., mos_true=...,
           ood_scores=..., id_ood_labels=...,   # 0=ID, 1=OOD
           groups=...,                          # optional cluster IDs
           n_boot=2000,
           extra_metrics={"my_metric": my_fn},  # metric_fn(scores, abs_errors)
       )

  4. CLI (see --help):
       python mos_conformal_toolkit.py predict --ref ref.npz --cal cal.npz --test test.npz

Input formats
-------------
Features can be provided as:
  - numpy arrays passed directly (programmatic use)
  - .npz files with keys: features (N,D), mos (N,), mos_pred (N,)
  - .csv files with columns: mos_pred, mos_true, feat_0 .. feat_{D-1}
    (feature columns named feat_N; mos_true not required for predict-only)

Dependencies: numpy, scipy, pandas, scikit-learn  (all in requirements.txt)
"""

from __future__ import annotations

import argparse
import datetime
import json
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a features NPZ.  Returns dict with at least 'features' key."""
    data = np.load(path, allow_pickle=True)
    out = {"features": data["features"].astype(np.float32)}
    for key in ("mos", "mos_pred", "filepaths"):
        if key in data:
            out[key] = data[key]
    return out


def load_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load a CSV with columns mos_pred, mos_true (optional), feat_0..feat_N."""
    df = pd.read_csv(path)
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")],
                       key=lambda c: int(c.split("_")[1]))
    out: dict[str, np.ndarray] = {}
    if feat_cols:
        out["features"] = df[feat_cols].to_numpy(dtype=np.float32)
    if "mos_pred" in df.columns:
        out["mos_pred"] = df["mos_pred"].to_numpy(dtype=np.float32)
    if "mos_true" in df.columns:
        out["mos"] = df["mos_true"].to_numpy(dtype=np.float32)
    if "group" in df.columns:
        out["filepaths"] = df["group"].to_numpy()
    return out


def _load_any(src: str | Path | np.ndarray | dict) -> dict[str, np.ndarray]:
    """Accept ndarray, dict, .npz path, or .csv path."""
    if isinstance(src, np.ndarray):
        return {"features": src.astype(np.float32)}
    if isinstance(src, dict):
        return {k: np.asarray(v) for k, v in src.items()}
    src = Path(src)
    return load_npz(src) if src.suffix == ".npz" else load_csv(src)


# ---------------------------------------------------------------------------
# BootstrapCI
# ---------------------------------------------------------------------------

class BootstrapCI:
    """
    Generic bootstrap confidence interval for any scalar metric.

    Parameters
    ----------
    n_boot   : number of bootstrap replicates (default 2000)
    ci_level : coverage of the CI (default 0.95 → 95% CI)
    seed     : random seed for reproducibility
    groups   : optional array of group/cluster IDs (same length as the arrays
               passed to compute()).  When provided, whole groups are resampled
               rather than individual rows — use this when rows within a group
               are not independent (e.g. multiple recordings from the same talker).

    Usage
    -----
    bci = BootstrapCI(n_boot=2000)
    point, lo, hi = bci.compute(scores, abs_errors,
                                metric_fn=lambda s, e: spearmanr(s, e).statistic)

    # With clusters (e.g. talker IDs)
    bci = BootstrapCI(n_boot=2000, groups=talker_ids)
    point, lo, hi = bci.compute(scores, errors, metric_fn=roc_auc_score)
    """

    def __init__(
        self,
        n_boot: int = 2000,
        ci_level: float = 0.95,
        seed: int = 42,
        groups: np.ndarray | None = None,
    ) -> None:
        self.n_boot = n_boot
        self.ci_level = ci_level
        self.rng = np.random.default_rng(seed)
        self.groups = np.asarray(groups) if groups is not None else None

    def _row_indices(self, n: int) -> np.ndarray:
        """One bootstrap resample of row indices."""
        if self.groups is None:
            return self.rng.integers(0, n, size=n)
        unique = np.unique(self.groups)
        chosen = self.rng.choice(unique, size=len(unique), replace=True)
        return np.concatenate([np.where(self.groups == g)[0] for g in chosen])

    def compute(
        self,
        *arrays: np.ndarray,
        metric_fn: Callable[..., float],
    ) -> tuple[float, float, float]:
        """
        Compute point estimate and percentile bootstrap CI.

        Parameters
        ----------
        *arrays    : one or more arrays passed positionally to metric_fn.
                     All must have the same first-axis length (number of samples).
        metric_fn  : callable(*arrays) -> float.

        Returns
        -------
        point, ci_lo, ci_hi : floats
        """
        arrays = tuple(np.asarray(a) for a in arrays)
        n = arrays[0].shape[0]
        point = metric_fn(*arrays)

        boot_values: list[float] = []
        for _ in range(self.n_boot):
            idx = self._row_indices(n)
            try:
                val = metric_fn(*(a[idx] for a in arrays))
                if np.isfinite(val):
                    boot_values.append(val)
            except Exception:
                pass

        if not boot_values:
            return float(point), float("nan"), float("nan")

        lo_pct = 100 * (1 - self.ci_level) / 2
        hi_pct = 100 - lo_pct
        ci_lo = float(np.percentile(boot_values, lo_pct))
        ci_hi = float(np.percentile(boot_values, hi_pct))
        return float(point), ci_lo, ci_hi


# ---------------------------------------------------------------------------
# OOD scoring (Mahalanobis + k-NN)
# ---------------------------------------------------------------------------

class _OODScorer:
    """Internal: fits Mahalanobis / L2 / kNN on reference features."""

    def __init__(self, lw_threshold: float = 0.9) -> None:
        self._lw_threshold = lw_threshold
        self.mu: np.ndarray | None = None
        self.sigma_inv: np.ndarray | None = None
        self.ref_features: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "_OODScorer":
        features = features.astype(np.float64)
        N, D = features.shape
        self.mu = features.mean(axis=0)
        self.ref_features = features

        use_lw = (N / D) < self._lw_threshold
        cov = LedoitWolf() if use_lw else EmpiricalCovariance()
        cov.fit(features)
        self.sigma_inv = cov.precision_.astype(np.float64)
        return self

    def mahalanobis(self, features: np.ndarray) -> np.ndarray:
        diff = features.astype(np.float64) - self.mu
        sq = np.einsum("ni,ij,nj->n", diff, self.sigma_inv, diff)
        return np.sqrt(np.maximum(sq, 0.0)).astype(np.float32)

    def l2(self, features: np.ndarray) -> np.ndarray:
        return np.linalg.norm(features.astype(np.float64) - self.mu, axis=1).astype(np.float32)

    def knn(self, features: np.ndarray, k: int = 1) -> np.ndarray:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nbrs.fit(self.ref_features)
        dists, _ = nbrs.kneighbors(features.astype(np.float64))
        return dists[:, -1].astype(np.float32)


# ---------------------------------------------------------------------------
# Conformal interval core (mirrors conformal_wrapper.py)
# ---------------------------------------------------------------------------

class _ConformalIntervals:
    """Internal: split conformal intervals (fixed and Mahalanobis-adaptive)."""

    def __init__(self, alpha: float = 0.10, n_bins: int = 5) -> None:
        self.alpha = alpha
        self.n_bins = n_bins
        self._threshold: float | None = None
        self._bin_edges: np.ndarray | None = None
        self._bin_thresholds: np.ndarray | None = None
        self._n_cal: int = 0

    def _quantile(self, scores: np.ndarray) -> float:
        m = len(scores)
        k = min(int(np.ceil((m + 1) * (1 - self.alpha))), m)
        return float(np.partition(scores, k - 1)[k - 1])

    def calibrate(
        self,
        mos_true: np.ndarray,
        mos_pred: np.ndarray,
        ood_scores: np.ndarray | None = None,
    ) -> "_ConformalIntervals":
        residuals = np.abs(np.asarray(mos_true) - np.asarray(mos_pred))
        self._n_cal = len(residuals)
        self._threshold = self._quantile(residuals)

        if ood_scores is not None:
            ood = np.asarray(ood_scores, dtype=np.float64)
            edges = np.quantile(ood, np.linspace(0, 1, self.n_bins + 1))
            self._bin_edges = edges
            thresholds = []
            for k in range(self.n_bins):
                lo, hi = edges[k], edges[k + 1]
                mask = (ood >= lo) & (ood < hi) if k < self.n_bins - 1 else ood >= lo
                thresholds.append(self._quantile(residuals[mask]) if mask.sum() >= 10
                                  else self._threshold)
            self._bin_thresholds = np.array(thresholds)
        return self

    def predict_fixed(self, mos_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hw = self._threshold
        pred = np.asarray(mos_pred, dtype=np.float64)
        return (pred - hw).astype(np.float32), (pred + hw).astype(np.float32)

    def predict_adaptive(
        self, mos_pred: np.ndarray, ood_scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pred = np.asarray(mos_pred, dtype=np.float64)
        ood = np.asarray(ood_scores, dtype=np.float64)
        bin_idx = np.clip(np.digitize(ood, self._bin_edges[1:-1]), 0, self.n_bins - 1)
        hw = self._bin_thresholds[bin_idx]
        return (pred - hw).astype(np.float32), (pred + hw).astype(np.float32)


# ---------------------------------------------------------------------------
# MOSConformalPredictor  (main public class)
# ---------------------------------------------------------------------------

class MOSConformalPredictor:
    """
    Full pipeline: OOD scoring + conformal prediction intervals.

    Typical usage
    -------------
    predictor = MOSConformalPredictor(alpha=0.10)

    # 1. Fit OOD reference on in-distribution training features
    predictor.fit_reference(ref_features)          # numpy array (N, D)

    # 2. Calibrate conformal intervals on a held-out calibration set
    predictor.calibrate(cal_features, cal_mos_true, cal_mos_pred)

    # 3. Predict on new data — returns a DataFrame
    df = predictor.predict(test_features, test_mos_pred)
    #    columns: mos_pred, ci_low_fixed, ci_high_fixed,
    #             ci_low_adaptive, ci_high_adaptive, ood_score

    # 4. Save / load
    predictor.save("predictor.pkl")
    predictor = MOSConformalPredictor.load("predictor.pkl")

    Parameters
    ----------
    alpha  : miscoverage level; 0.10 → 90% nominal coverage
    n_bins : number of Mahalanobis-distance bins for adaptive intervals
    ood_score : which OOD score to use for the adaptive path and the
                'ood_score' output column.
                One of "mahalanobis" (default), "l2", "knn".
    knn_k  : k for kNN distance (only used when ood_score="knn")
    """

    def __init__(
        self,
        alpha: float = 0.10,
        n_bins: int = 5,
        ood_score: str = "mahalanobis",
        knn_k: int = 1,
    ) -> None:
        if ood_score not in {"mahalanobis", "l2", "knn"}:
            raise ValueError("ood_score must be 'mahalanobis', 'l2', or 'knn'")
        self.alpha = alpha
        self.n_bins = n_bins
        self.ood_score_type = ood_score
        self.knn_k = knn_k
        self._ood = _OODScorer()
        self._cp = _ConformalIntervals(alpha=alpha, n_bins=n_bins)
        self._fitted = False
        self._calibrated = False

    # ------------------------------------------------------------------
    # Fitting and calibration
    # ------------------------------------------------------------------

    def fit_reference(self, features: np.ndarray | str | Path) -> "MOSConformalPredictor":
        """
        Fit the OOD reference distribution on in-distribution features.

        Parameters
        ----------
        features : (N, D) float array, or path to .npz/.csv file.
                   These should be penultimate-layer activations of the MOS
                   predictor on in-distribution (reference) audio clips.
        """
        data = _load_any(features)
        self._ood.fit(data["features"])
        self._fitted = True
        return self

    def calibrate(
        self,
        features: np.ndarray | str | Path,
        mos_true: np.ndarray | None = None,
        mos_pred: np.ndarray | None = None,
    ) -> "MOSConformalPredictor":
        """
        Calibrate conformal intervals on a held-out calibration set.

        The calibration set must be disjoint from the reference set passed to
        fit_reference(), and its MOS labels must NOT have been used to train or
        select the base MOS predictor.

        Parameters
        ----------
        features : (m, D) features, or path to .npz/.csv.
                   If .npz, mos and mos_pred are read from there automatically.
        mos_true : (m,) true MOS values (overrides file if provided)
        mos_pred : (m,) predicted MOS values (overrides file if provided)
        """
        if not self._fitted:
            raise RuntimeError("Call fit_reference() before calibrate().")
        data = _load_any(features)
        if mos_true is None:
            mos_true = data["mos"]
        if mos_pred is None:
            mos_pred = data["mos_pred"]

        ood_scores = self._score(data["features"])
        self._cp.calibrate(mos_true, mos_pred, ood_scores)
        self._calibrated = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _score(self, features: np.ndarray) -> np.ndarray:
        if self.ood_score_type == "mahalanobis":
            return self._ood.mahalanobis(features)
        if self.ood_score_type == "l2":
            return self._ood.l2(features)
        return self._ood.knn(features, k=self.knn_k)

    def predict(
        self,
        features: np.ndarray | str | Path,
        mos_pred: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Compute conformal prediction intervals and OOD scores for new samples.

        Parameters
        ----------
        features : (n, D) features, or path to .npz/.csv.
        mos_pred : (n,) MOS point predictions (read from file if None).

        Returns
        -------
        DataFrame with columns:
            mos_pred          — point estimate from the base predictor
            ci_low_fixed      — lower bound, fixed-width 90% interval
            ci_high_fixed     — upper bound, fixed-width 90% interval
            ci_low_adaptive   — lower bound, Mahalanobis-adaptive interval
            ci_high_adaptive  — upper bound, Mahalanobis-adaptive interval
            interval_width_fixed    — 2 * fixed threshold
            interval_width_adaptive — per-sample width of adaptive interval
            ood_score         — feature-space OOD distance (higher = more OOD)

        Coverage interpretation
        -----------------------
        On data exchangeable with the calibration set, ci_low_fixed and
        ci_high_fixed contain the true MOS in >= (1 - alpha) of cases.
        Coverage under distribution shift is not guaranteed.
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before predict().")
        data = _load_any(features)
        feats = data["features"]
        if mos_pred is None:
            mos_pred = data["mos_pred"]
        mos_pred = np.asarray(mos_pred, dtype=np.float32)

        ood = self._score(feats)
        lo_f, hi_f = self._cp.predict_fixed(mos_pred)
        lo_a, hi_a = self._cp.predict_adaptive(mos_pred, ood)

        return pd.DataFrame({
            "mos_pred":               mos_pred,
            "ci_low_fixed":           lo_f,
            "ci_high_fixed":          hi_f,
            "ci_low_adaptive":        lo_a,
            "ci_high_adaptive":       hi_a,
            "interval_width_fixed":   hi_f - lo_f,
            "interval_width_adaptive": hi_a - lo_a,
            "ood_score":              ood,
        })

    def ood_score(self, features: np.ndarray | str | Path) -> np.ndarray:
        """Return only the OOD scores (no intervals required)."""
        if not self._fitted:
            raise RuntimeError("Call fit_reference() first.")
        data = _load_any(features)
        return self._score(data["features"])

    def coverage(
        self,
        features: np.ndarray | str | Path,
        mos_true: np.ndarray | None = None,
        mos_pred: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Return empirical coverage for both interval types on a test set."""
        data = _load_any(features)
        if mos_true is None:
            mos_true = data["mos"]
        if mos_pred is None:
            mos_pred = data["mos_pred"]
        df = self.predict(data["features"], mos_pred)
        mos_true = np.asarray(mos_true)
        cov_fixed = float(((mos_true >= df["ci_low_fixed"]) & (mos_true <= df["ci_high_fixed"])).mean())
        cov_adapt = float(((mos_true >= df["ci_low_adaptive"]) & (mos_true <= df["ci_high_adaptive"])).mean())
        return {"coverage_fixed": cov_fixed, "coverage_adaptive": cov_adapt,
                "nominal": 1.0 - self.alpha}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "MOSConformalPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# evaluate_mos()  — one-call evaluation with CIs
# ---------------------------------------------------------------------------

def evaluate_mos(
    mos_pred: np.ndarray,
    mos_true: np.ndarray,
    ood_scores: np.ndarray | None = None,
    id_ood_labels: np.ndarray | None = None,
    ci_low: np.ndarray | None = None,
    ci_high: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
    extra_metrics: dict[str, Callable] | None = None,
) -> dict[str, dict]:
    """
    Compute all standard metrics with bootstrap CIs from raw arrays.

    Parameters
    ----------
    mos_pred      : (n,) predicted MOS values
    mos_true      : (n,) true MOS values from a subjective test
    ood_scores    : (n,) feature-space OOD distance scores (optional).
                    Used to compute domain-detection and error-detection AUROC.
    id_ood_labels : (n,) binary labels: 0 = in-distribution, 1 = OOD.
                    Required for domain-detection AUROC.
    ci_low        : (n,) lower bounds of conformal intervals (optional)
    ci_high       : (n,) upper bounds of conformal intervals (optional)
    groups        : (n,) cluster/group IDs for non-i.i.d. bootstrap (optional).
                    If provided, whole groups are resampled. Use when rows in
                    the same group share content (e.g. same talker, same utterance
                    recorded with multiple codecs).
    n_boot        : bootstrap replicates (default 2000)
    ci_level      : CI coverage (default 0.95)
    extra_metrics : dict of name -> callable(ood_scores, abs_errors) -> float.
                    Custom metrics computed on (ood_scores, |mos_pred - mos_true|).

    Returns
    -------
    dict mapping metric name to {"point": float, "ci_lo": float, "ci_hi": float}

    Example
    -------
    results = evaluate_mos(
        mos_pred=predictions, mos_true=labels,
        ood_scores=distances, id_ood_labels=domain_labels,
        groups=talker_ids, n_boot=2000,
    )
    for metric, vals in results.items():
        print(f"{metric}: {vals['point']:.3f}  [{vals['ci_lo']:.3f}, {vals['ci_hi']:.3f}]")
    """
    mos_pred  = np.asarray(mos_pred,  dtype=np.float64)
    mos_true  = np.asarray(mos_true,  dtype=np.float64)
    abs_err   = np.abs(mos_pred - mos_true)
    bci       = BootstrapCI(n_boot=n_boot, ci_level=ci_level, seed=seed, groups=groups)

    results: dict[str, dict] = {}

    def _store(name: str, point: float, lo: float, hi: float) -> None:
        results[name] = {"point": round(point, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    # --- Point-estimate metrics ------------------------------------------
    from scipy.stats import pearsonr

    def _srcc(p, t): return float(spearmanr(p, t).statistic)
    def _pcc(p, t):  return float(pearsonr(p, t)[0])
    def _rmse(p, t): return float(np.sqrt(np.mean((p - t) ** 2)))

    for name, fn in [("srcc", _srcc), ("pcc", _pcc), ("rmse", _rmse)]:
        pt, lo, hi = bci.compute(mos_pred, mos_true, metric_fn=fn)
        _store(name, pt, lo, hi)

    # --- Coverage (if intervals supplied) --------------------------------
    if ci_low is not None and ci_high is not None:
        ci_low  = np.asarray(ci_low,  dtype=np.float64)
        ci_high = np.asarray(ci_high, dtype=np.float64)

        def _cov(lo, hi, t): return float(((t >= lo) & (t <= hi)).mean())
        def _width(lo, hi, _t): return float((hi - lo).mean())

        pt, lo, hi = bci.compute(ci_low, ci_high, mos_true, metric_fn=_cov)
        _store("coverage", pt, lo, hi)
        pt, lo, hi = bci.compute(ci_low, ci_high, mos_true, metric_fn=_width)
        _store("interval_width", pt, lo, hi)

    # --- OOD / error-detection metrics (require ood_scores) ---------------
    if ood_scores is not None:
        ood_scores = np.asarray(ood_scores, dtype=np.float64)

        # Spearman correlation: OOD score vs absolute error
        def _srcc_err(s, e): return float(spearmanr(s, e).statistic)
        pt, lo, hi = bci.compute(ood_scores, abs_err, metric_fn=_srcc_err)
        _store("srcc_score_vs_error", pt, lo, hi)

        # Error detection AUROC (top-quartile |error| as positive class)
        q75 = np.percentile(abs_err, 75)
        hard_labels = (abs_err >= q75).astype(int)

        def _auroc_err(s, lbl): return float(roc_auc_score(lbl, s))
        def _aupr_err(s, lbl):  return float(average_precision_score(lbl, s))

        def _fpr95_err(s, lbl):
            fpr, tpr, _ = roc_curve(lbl, s)
            idx = np.searchsorted(tpr, 0.95)
            return float(fpr[min(idx, len(fpr) - 1)])

        for name, fn in [("error_auroc", _auroc_err),
                         ("error_aupr",  _aupr_err),
                         ("error_fpr95", _fpr95_err)]:
            pt, lo, hi = bci.compute(ood_scores, hard_labels, metric_fn=fn)
            _store(name, pt, lo, hi)

        # Domain-detection AUROC (requires id_ood_labels)
        if id_ood_labels is not None:
            id_ood = np.asarray(id_ood_labels, dtype=int)
            pt, lo, hi = bci.compute(ood_scores, id_ood,
                                     metric_fn=lambda s, l: float(roc_auc_score(l, s)))
            _store("domain_auroc", pt, lo, hi)

            def _fpr95_dom(s, lbl):
                fpr, tpr, _ = roc_curve(lbl, s)
                idx = np.searchsorted(tpr, 0.95)
                return float(fpr[min(idx, len(fpr) - 1)])
            pt, lo, hi = bci.compute(ood_scores, id_ood, metric_fn=_fpr95_dom)
            _store("domain_fpr95", pt, lo, hi)

    # --- User-supplied extra metrics --------------------------------------
    if extra_metrics:
        for name, fn in extra_metrics.items():
            pt, lo, hi = bci.compute(ood_scores if ood_scores is not None else abs_err,
                                     abs_err, metric_fn=fn)
            _store(name, pt, lo, hi)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_predict(args: argparse.Namespace) -> None:
    print(f"Loading reference features: {args.ref}")
    predictor = MOSConformalPredictor(
        alpha=args.alpha, n_bins=args.n_bins, ood_score=args.ood_score
    )
    predictor.fit_reference(args.ref)

    print(f"Calibrating on: {args.cal}")
    predictor.calibrate(args.cal)

    print(f"Predicting on: {args.test}")
    df = predictor.predict(args.test)

    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f"Predictions saved → {out}")

    # Print coverage summary if mos_true is available
    test_data = _load_any(args.test)
    if "mos" in test_data:
        cov = predictor.coverage(args.test)
        print(f"\nCoverage (nominal {1-args.alpha:.0%}):")
        print(f"  Fixed    : {cov['coverage_fixed']:.3f}")
        print(f"  Adaptive : {cov['coverage_adaptive']:.3f}")


def _cli_evaluate(args: argparse.Namespace) -> None:
    data = _load_any(args.data)
    mos_pred = data.get("mos_pred")
    mos_true = data.get("mos")
    if mos_pred is None or mos_true is None:
        raise ValueError("Data file must contain mos_pred and mos (mos_true) columns.")

    results = evaluate_mos(
        mos_pred=mos_pred,
        mos_true=mos_true,
        n_boot=args.n_boot,
    )

    print(f"\n{'Metric':<30} {'Point':>8}  {'95% CI':>20}")
    print("-" * 62)
    for metric, vals in results.items():
        ci_str = f"[{vals['ci_lo']:.4f}, {vals['ci_hi']:.4f}]"
        print(f"{metric:<30} {vals['point']:>8.4f}  {ci_str:>20}")

    if args.out:
        rows = [{"metric": k, **v} for k, v in results.items()]
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nResults saved → {args.out}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mos_conformal_toolkit",
        description="MOS conformal prediction toolkit",
    )
    sub = parser.add_subparsers(dest="command")

    # --- predict sub-command ---
    p = sub.add_parser("predict", help="Fit, calibrate, and predict intervals")
    p.add_argument("--ref",   required=True, help=".npz/.csv reference (in-distribution) features")
    p.add_argument("--cal",   required=True, help=".npz/.csv calibration features + mos + mos_pred")
    p.add_argument("--test",  required=True, help=".npz/.csv test features + mos_pred")
    p.add_argument("--out",   default=f"predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    p.add_argument("--alpha",     type=float, default=0.10, help="Miscoverage level (default 0.10 → 90%% CI)")
    p.add_argument("--n_bins",    type=int,   default=5,    help="Bins for adaptive intervals")
    p.add_argument("--ood_score", default="mahalanobis", choices=["mahalanobis", "l2", "knn"])

    # --- evaluate sub-command ---
    e = sub.add_parser("evaluate", help="Compute metrics with bootstrap CIs from a data file")
    e.add_argument("--data",   required=True, help=".npz/.csv with mos_pred and mos_true")
    e.add_argument("--out",    default=None,  help="Optional output CSV for results")
    e.add_argument("--n_boot", type=int, default=2000)

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "predict":
        _cli_predict(args)
    elif args.command == "evaluate":
        _cli_evaluate(args)
    else:
        parser.print_help()
