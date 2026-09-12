"""
fit_mahalanobis.py — Fit Mahalanobis distance model on training features (DNS 2020).

The Mahalanobis distance is used as an OOD score (Lee et al., 2018, NeurIPS).
Given training features Z_train ∈ R^{N x D} (penultimate-layer activations),
we estimate the empirical mean μ and regularised covariance Σ, then
compute d_M(x) = sqrt((z - μ)^T Σ^{-1} (z - μ)) for any new sample z.

Regularisation: Ledoit-Wolf shrinkage (sklearn.covariance.LedoitWolf).
This is used when D is large relative to N (feature dim > sample count).

Outputs (saved to out_dir):
  mahal_<model>.npz
    Keys: "mu" (D,), "sigma_inv" (D, D), "feature_dim" (scalar), "n_train" (scalar)
  mahal_<model>_train_scores.csv
    Columns: filepath, mos, mos_pred, mahal_score

Usage:
  python fit_mahalanobis.py \\
    --features ../../results/features/features_dnsmos_dns2020.npz \\
    --model dnsmos \\
    --out_dir ../../results/mahal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, LedoitWolf


# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------

def fit_mahalanobis(
    features: np.ndarray,
    regularise: bool | None = None,
    shrinkage_threshold: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a Gaussian model on training features and return (mu, Sigma_inv).

    Parameters
    ----------
    features : (N, D) float32 array of training features
    regularise : if True, use Ledoit-Wolf; if False, use empirical covariance;
                 if None (default), choose automatically based on N vs D ratio.
    shrinkage_threshold : N/D ratio below which Ledoit-Wolf is applied when
                          regularise=None.

    Returns
    -------
    mu       : (D,) mean vector
    sigma_inv: (D, D) inverse covariance matrix
    """
    N, D = features.shape
    mu = features.mean(axis=0)

    if regularise is None:
        regularise = (N / D) < shrinkage_threshold

    if regularise:
        print(f"[fit_mahalanobis] N={N}, D={D} → using Ledoit-Wolf shrinkage.")
        cov_estimator = LedoitWolf()
    else:
        print(f"[fit_mahalanobis] N={N}, D={D} → using empirical covariance.")
        cov_estimator = EmpiricalCovariance()

    cov_estimator.fit(features)
    sigma_inv = cov_estimator.precision_  # (D, D) — precomputed inverse

    return mu.astype(np.float64), sigma_inv.astype(np.float64)


def mahalanobis_scores(
    features: np.ndarray,
    mu: np.ndarray,
    sigma_inv: np.ndarray,
) -> np.ndarray:
    """
    Compute Mahalanobis distance for each row in features.

    Parameters
    ----------
    features  : (N, D) float array
    mu        : (D,) mean vector
    sigma_inv : (D, D) precision matrix

    Returns
    -------
    scores : (N,) float array — d_M(x) for each sample
    """
    diff = features.astype(np.float64) - mu[np.newaxis, :]  # (N, D)
    # d_M^2 = diag(diff @ sigma_inv @ diff^T) — O(d^2) per sample with precomputed sigma_inv
    mahal_sq = np.einsum("ni,ij,nj->n", diff, sigma_inv, diff)
    return np.sqrt(np.maximum(mahal_sq, 0.0)).astype(np.float32)


def l2_scores(
    features: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """
    Euclidean (L2) distance from the training-set mean for each row in features.
    This is the ablation baseline that ignores feature correlations.

    Parameters
    ----------
    features : (N, D) float array
    mu       : (D,) mean vector

    Returns
    -------
    scores : (N,) float array — ||z - mu||_2 for each sample
    """
    diff = features.astype(np.float64) - mu[np.newaxis, :]  # (N, D)
    return np.linalg.norm(diff, axis=1).astype(np.float32)


def knn_scores(
    features: np.ndarray,
    train_features: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """
    k-th nearest-neighbour distance in training-feature space (Sun et al., 2022).
    A large k-NN distance indicates the test sample is far from any training point.

    Parameters
    ----------
    features       : (N, D) float array — test features
    train_features : (M, D) float array — training features
    k              : number of neighbours (default 1; Sun et al. 2022 use k=1)

    Returns
    -------
    scores : (N,) float array — distance to k-th nearest training neighbour
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nbrs.fit(train_features.astype(np.float64))
    distances, _ = nbrs.kneighbors(features.astype(np.float64))
    return distances[:, -1].astype(np.float32)  # distance to k-th neighbour


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_model(
    out_dir: Path,
    model_name: str,
    mu: np.ndarray,
    sigma_inv: np.ndarray,
    n_train: int,
) -> Path:
    out_path = out_dir / f"mahal_{model_name}.npz"
    np.savez_compressed(
        out_path,
        mu=mu,
        sigma_inv=sigma_inv,
        feature_dim=np.array([mu.shape[0]]),
        n_train=np.array([n_train]),
    )
    print(f"[fit_mahalanobis] Saved model → {out_path}")
    return out_path


def load_model(model_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved Mahalanobis model and return (mu, sigma_inv)."""
    data = np.load(model_path)
    return data["mu"], data["sigma_inv"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit Mahalanobis OOD model on training features.")
    p.add_argument("--features", required=True,
                   help="Path to .npz feature file (from extract_features.py, DNS 2020 split).")
    p.add_argument("--model", required=True,
                   help="Model name (dnsmos | nisqa | utmos) — used in output filename.")
    p.add_argument("--out_dir", default="../../results/mahal",
                   help="Output directory for fitted model .npz.")
    p.add_argument("--no_regularise", action="store_true",
                   help="Force empirical covariance (skip Ledoit-Wolf).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.features, allow_pickle=True)
    features = data["features"].astype(np.float32)          # (N, D)
    mos = data["mos"].astype(np.float32)
    mos_pred = data["mos_pred"].astype(np.float32) if "mos_pred" in data else np.zeros_like(mos)
    filepaths = data["filepaths"]

    N, D = features.shape
    print(f"[fit_mahalanobis] Training features: N={N}, D={D}")

    mu, sigma_inv = fit_mahalanobis(
        features,
        regularise=False if args.no_regularise else None,
    )
    save_model(out_dir, args.model, mu, sigma_inv, n_train=N)

    # Compute and save training-set scores (for sanity check)
    scores = mahalanobis_scores(features, mu, sigma_inv)
    df_scores = pd.DataFrame({
        "filepath": filepaths,
        "mos": mos,
        "mos_pred": mos_pred,
        "mahal_score": scores,
    })
    score_csv = out_dir / f"mahal_{args.model}_train_scores.csv"
    df_scores.to_csv(score_csv, index=False)
    print(f"[fit_mahalanobis] Training scores saved → {score_csv}")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}], "
          f"mean={scores.mean():.3f}, std={scores.std():.3f}")


if __name__ == "__main__":
    main()
