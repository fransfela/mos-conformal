"""
robustness_mars_mahalpp.py — Repository robustness check: MaRS and Mahalanobis++.

Not included in the paper. Run to assess whether two stronger OOD baselines
improve domain-detection or utterance-level error-detection AUROC beyond the
L2 / Mahalanobis / kNN scores already reported.

Methods
-------
Mahalanobis++ (label-free)
    L2-normalise features before fitting the standard Mahalanobis detector.
    Follows the feature-normalisation spirit of Müller & Hein (ICML 2025)
    without the class-conditional Gaussian, which requires labels unavailable here.

MaRS (Mahalanobis Residual Scoring)
    Di Salvo et al. (arXiv 2606.22649, 2026).
    1. Train a lightweight MLP autoencoder on T-SIM features.
    2. Compute reconstruction residuals r = z - z_hat for every T-SIM clip.
    3. Fit Mahalanobis distance on the residual covariance.
    4. Score any clip: S_MaRS(x) = r(x)^T Sigma^{-1} r(x).
    Key idea: OOD deviations concentrate in low-variance residual directions;
    Mahalanobis scoring up-weights those directions.

Protocol
--------
Identical to evaluate.py / error_detection.py canonical protocol:
  - _REFERENCE_DATASET  : nisqa_train_sim  (fit reference statistics)
  - _ID_EVAL_DATASET    : nisqa_val_sim    (held-out ID evaluation)
  - _OOD_DATASETS       : T-LIV, V-LIV, T-FOR, T-TLK, T-P501
Domain-detection AUROC: binary ID (V-SIM) vs. each OOD subset.
Error-detection AUROC : top-quartile |MOS error| detection within each subset.

Usage
-----
    python robustness_mars_mahalpp.py \\
        --features_dir ../../results/features \\
        --out_dir      ../../results

Outputs
-------
    results/robustness_mars_mahalpp_<timestamp>.csv
        columns: model, dataset, score, auroc_domain, auroc_error
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# Protocol constants — must match evaluate.py
# ---------------------------------------------------------------------------
_REFERENCE_DATASET = "nisqa_train_sim"
_ID_EVAL_DATASET = "nisqa_val_sim"
_OOD_DATASETS = {
    "nisqa_train_live",
    "nisqa_val_live",
    "nisqa_test_for",
    "nisqa_test_livetalk",
    "nisqa_test_p501",
}
_ALL_DATASETS = [
    "nisqa_train_sim",
    "nisqa_train_live",
    "nisqa_val_sim",
    "nisqa_val_live",
    "nisqa_test_for",
    "nisqa_test_livetalk",
    "nisqa_test_p501",
]
_MODELS = ["nisqa", "dnsmos", "utmos"]

# Autoencoder training config (following MaRS paper: 50 epochs, hidden ratio 0.5)
_AE_EPOCHS = 50
_AE_LR = 1e-3
_AE_BATCH = 256
_AE_BOTTLENECK_MAX = 128  # cap bottleneck at 128 regardless of D


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_features(features_dir: Path, model: str, dataset: str) -> dict | None:
    path = features_dir / f"features_{model}_{dataset}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "features": data["features"].astype(np.float32),
        "mos": data["mos"].astype(np.float32),
        "mos_pred": data["mos_pred"].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Mahalanobis++ (label-free): L2-normalise then standard Mahalanobis
# ---------------------------------------------------------------------------

def fit_mahalpp(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit Mahalanobis++ reference on features.
    Returns (mu, sigma_inv, scale) where scale is the per-sample L2 mean used
    to normalise at score time; we store the mean norm for reference only.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    z_norm = features / norms                            # (N, D) unit vectors

    mu = z_norm.mean(axis=0)
    N, D = z_norm.shape
    use_lw = (N / D) < 0.9
    cov = LedoitWolf() if use_lw else EmpiricalCovariance()
    cov.fit(z_norm)
    sigma_inv = cov.precision_.astype(np.float64)
    return mu.astype(np.float64), sigma_inv, norms.mean()


def mahalpp_scores(features: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    z_norm = features.astype(np.float64) / norms
    diff = z_norm - mu[np.newaxis, :]
    sq = np.einsum("ni,ij,nj->n", diff, sigma_inv, diff)
    return np.sqrt(np.maximum(sq, 0.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# MaRS autoencoder
# ---------------------------------------------------------------------------

class _MLP_AE(nn.Module):
    def __init__(self, d: int, bottleneck: int) -> None:
        super().__init__()
        hidden = max(d // 2, bottleneck)
        self.encoder = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    features: np.ndarray,
    epochs: int = _AE_EPOCHS,
    lr: float = _AE_LR,
    batch_size: int = _AE_BATCH,
    seed: int = 0,
) -> _MLP_AE:
    """Train an autoencoder on ID features and return the fitted model."""
    torch.manual_seed(seed)
    N, D = features.shape
    bottleneck = min(D // 2, _AE_BOTTLENECK_MAX)
    model = _MLP_AE(D, bottleneck)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    tensor = torch.from_numpy(features)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            batch = tensor[perm[start : start + batch_size]]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(batch), batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 10 == 0:
            print(f"  AE epoch {epoch+1:3d}/{epochs}  loss={epoch_loss/n_batches:.6f}")

    model.eval()
    return model


def compute_residuals(model: _MLP_AE, features: np.ndarray) -> np.ndarray:
    """Return reconstruction residuals r = z - z_hat for all rows."""
    with torch.no_grad():
        z = torch.from_numpy(features)
        z_hat = model(z).numpy()
    return (features - z_hat).astype(np.float32)


def fit_mars_covariance(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit covariance of ID residuals.  Returns (mu_r, sigma_inv_r).
    Uses Ledoit-Wolf when N/D < threshold (UTMOS: D=1024, N=10k → N/D≈9.8).
    """
    N, D = residuals.shape
    mu_r = residuals.mean(axis=0)
    use_lw = (N / D) < 0.9
    cov = LedoitWolf() if use_lw else EmpiricalCovariance()
    cov.fit(residuals)
    return mu_r.astype(np.float64), cov.precision_.astype(np.float64)


def mars_scores(
    model: _MLP_AE,
    features: np.ndarray,
    mu_r: np.ndarray,
    sigma_inv_r: np.ndarray,
) -> np.ndarray:
    """Compute S_MaRS(x) = (r - mu_r)^T Sigma_r^{-1} (r - mu_r)."""
    residuals = compute_residuals(model, features).astype(np.float64)
    diff = residuals - mu_r[np.newaxis, :]
    sq = np.einsum("ni,ij,nj->n", diff, sigma_inv_r, diff)
    return np.sqrt(np.maximum(sq, 0.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# AUROC helpers (matching evaluate.py / error_detection.py)
# ---------------------------------------------------------------------------

def domain_auroc(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> float:
    """Binary AUROC: ID=0, OOD=1. Higher score → more OOD."""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return roc_auc_score(labels, scores)


def error_metrics(scores: np.ndarray, errors: np.ndarray) -> dict:
    """
    Compute all error-detection metrics for one (score, errors) pair.
    Hard-label task: top-quartile |error| = 1, rest = 0 (matches error_detection.py).
    """
    abs_err = np.abs(errors)
    threshold = np.percentile(abs_err, 75)
    hard_labels = (abs_err >= threshold).astype(int)

    if hard_labels.sum() == 0 or hard_labels.sum() == len(hard_labels):
        return {"auroc": float("nan"), "fpr95": float("nan"),
                "aupr": float("nan"), "srcc": float("nan")}

    auroc = roc_auc_score(hard_labels, scores)
    aupr  = average_precision_score(hard_labels, scores)

    fpr, tpr, _ = roc_curve(hard_labels, scores)
    # FPR at the threshold where TPR >= 0.95 (first crossing)
    idx = np.searchsorted(tpr, 0.95)
    fpr95 = float(fpr[min(idx, len(fpr) - 1)])

    srcc = float(spearmanr(scores, abs_err).statistic)

    return {"auroc": auroc, "fpr95": fpr95, "aupr": aupr, "srcc": srcc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(features_dir: Path, out_dir: Path) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict] = []

    for model in _MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        # Load all feature sets
        all_data = {}
        for ds in _ALL_DATASETS:
            d = load_features(features_dir, model, ds)
            if d is None:
                print(f"  [SKIP] features not found: {model} / {ds}")
            else:
                all_data[ds] = d

        if _REFERENCE_DATASET not in all_data or _ID_EVAL_DATASET not in all_data:
            print(f"  [SKIP] missing reference or ID-eval features for {model}")
            continue

        ref_feats = all_data[_REFERENCE_DATASET]["features"]   # T-SIM
        id_feats  = all_data[_ID_EVAL_DATASET]["features"]     # V-SIM (held-out)
        id_errors = all_data[_ID_EVAL_DATASET]["mos_pred"] - all_data[_ID_EVAL_DATASET]["mos"]

        # ------------------------------------------------------------------
        # Mahalanobis++ (label-free)
        # ------------------------------------------------------------------
        print("\n--- Mahalanobis++ ---")
        mu_pp, sig_pp, _ = fit_mahalpp(ref_feats)
        id_scores_pp = mahalpp_scores(id_feats, mu_pp, sig_pp)

        for ds in _OOD_DATASETS:
            if ds not in all_data:
                continue
            ood = all_data[ds]
            ood_scores = mahalpp_scores(ood["features"], mu_pp, sig_pp)

            auc_d = domain_auroc(id_scores_pp, ood_scores)
            em = error_metrics(ood_scores, ood["mos_pred"] - ood["mos"])
            print(f"  {ds:30s}  domain={auc_d:.3f}  "
                  f"err_auroc={em['auroc']:.3f}  fpr95={em['fpr95']:.3f}  "
                  f"aupr={em['aupr']:.3f}  srcc={em['srcc']:+.3f}")
            rows.append({
                "model": model, "dataset": ds, "score": "mahalpp",
                "auroc_domain": auc_d,
                "auroc_error": em["auroc"], "fpr95_error": em["fpr95"],
                "aupr_error": em["aupr"], "srcc_error": em["srcc"],
            })

        # ------------------------------------------------------------------
        # MaRS
        # ------------------------------------------------------------------
        print("\n--- MaRS ---")
        print("  Training autoencoder on T-SIM features ...")
        ae = train_autoencoder(ref_feats)

        print("  Fitting residual covariance on T-SIM ...")
        id_residuals = compute_residuals(ae, ref_feats)
        mu_r, sig_r = fit_mars_covariance(id_residuals)

        id_scores_mars = mars_scores(ae, id_feats, mu_r, sig_r)

        for ds in _OOD_DATASETS:
            if ds not in all_data:
                continue
            ood = all_data[ds]
            ood_scores = mars_scores(ae, ood["features"], mu_r, sig_r)

            auc_d = domain_auroc(id_scores_mars, ood_scores)
            em = error_metrics(ood_scores, ood["mos_pred"] - ood["mos"])
            print(f"  {ds:30s}  domain={auc_d:.3f}  "
                  f"err_auroc={em['auroc']:.3f}  fpr95={em['fpr95']:.3f}  "
                  f"aupr={em['aupr']:.3f}  srcc={em['srcc']:+.3f}")
            rows.append({
                "model": model, "dataset": ds, "score": "mars",
                "auroc_domain": auc_d,
                "auroc_error": em["auroc"], "fpr95_error": em["fpr95"],
                "aupr_error": em["aupr"], "srcc_error": em["srcc"],
            })

    # Save
    df = pd.DataFrame(rows)
    out_path = out_dir / f"robustness_mars_mahalpp_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # Summary table
    if not df.empty:
        print("\n--- Mean metrics by score (over all model-dataset pairs) ---")
        cols = ["auroc_domain", "auroc_error", "fpr95_error", "aupr_error", "srcc_error"]
        summary = df.groupby("score")[cols].mean().round(3)
        print(summary.to_string())

        print("\n--- Canonical baselines (from paper / error_detection canonical) ---")
        print("  score         auroc_domain  auroc_error  fpr95_error  aupr_error  srcc_error")
        print("  l2            0.615         0.433        ~0.91        ~0.27       ~-0.08")
        print("  mahalanobis   0.678         0.478        ~0.88        ~0.28       ~+0.05")
        print("  knn           0.690         0.499        ~0.87        ~0.28       ~+0.05")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaRS and Mahalanobis++ robustness check")
    parser.add_argument(
        "--features_dir", type=Path,
        default=Path(__file__).parent.parent.parent / "results" / "features",
    )
    parser.add_argument(
        "--out_dir", type=Path,
        default=Path(__file__).parent.parent.parent / "results",
    )
    args = parser.parse_args()
    main(args.features_dir, args.out_dir)
