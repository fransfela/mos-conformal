"""
conformal_wrapper.py — Split conformal prediction wrapper for MOS predictors.

Implements:
  1. Fixed-width conformal interval (Eq. 4 in paper)
  2. OOD-adaptive conformal interval with Mahalanobis-distance binning (Eq. 5 in paper)

Both methods use split conformal prediction (Papadopoulos et al., 2002).
The non-conformity score is the absolute residual: s = |q* - q_hat|.

Usage (as a library):
  from conformal_wrapper import ConformalWrapper
  wrapper = ConformalWrapper(alpha=0.10, adaptive=True, n_bins=5)
  wrapper.calibrate(mos_true, mos_pred, mahal_scores)
  lo, hi = wrapper.predict(mos_pred_test, mahal_scores_test)

Usage (CLI — calibrate and save):
  python conformal_wrapper.py \\
    --pred_csv ../../results/scores_dnsmos_dns2021.csv \\
    --mahal_csv ../../results/mahal/mahal_dnsmos_dns2021_scores.csv \\
    --alpha 0.10 \\
    --n_bins 5 \\
    --out ../../results/conformal/wrapper_dnsmos.npz
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core conformal wrapper
# ---------------------------------------------------------------------------

class ConformalWrapper:
    """
    Split conformal prediction wrapper for a MOS predictor.

    Parameters
    ----------
    alpha   : miscoverage level; coverage target is 1 - alpha (e.g. 0.10 → 90%)
    adaptive: if True, use Mahalanobis-binned adaptive intervals (Eq. 5);
              if False, use fixed-width intervals (Eq. 4)
    n_bins  : number of quantile bins for adaptive mode (K in the paper)
    """

    def __init__(
        self,
        alpha: float = 0.10,
        adaptive: bool = False,
        n_bins: int = 5,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha
        self.adaptive = adaptive
        self.n_bins = n_bins

        self._threshold: float | None = None            # fixed-width threshold
        self._bin_edges: np.ndarray | None = None        # adaptive bin edges
        self._bin_thresholds: np.ndarray | None = None   # per-bin thresholds
        self._n_cal: int = 0
        self._is_calibrated: bool = False

    def _conformal_quantile(self, scores: np.ndarray) -> float:
        """Return the exact split-conformal order statistic s_(k)."""
        scores = np.asarray(scores, dtype=np.float64)
        m = len(scores)
        if m == 0:
            raise ValueError("Cannot calibrate from an empty score array.")
        k = min(int(np.ceil((m + 1) * (1 - self.alpha))), m)
        return float(np.partition(scores, k - 1)[k - 1])

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        mos_true: np.ndarray,
        mos_pred: np.ndarray,
        mahal_scores: np.ndarray | None = None,
    ) -> "ConformalWrapper":
        """
        Calibrate the wrapper using a held-out calibration set.

        Parameters
        ----------
        mos_true     : (m,) true MOS values from subjective test
        mos_pred     : (m,) predicted MOS from the base model
        mahal_scores : (m,) Mahalanobis distance scores (required if adaptive=True)
        """
        mos_true = np.asarray(mos_true, dtype=np.float64)
        mos_pred = np.asarray(mos_pred, dtype=np.float64)
        m = len(mos_true)
        self._n_cal = m

        residuals = np.abs(mos_true - mos_pred)  # non-conformity scores s_j = |q* - q_hat|

        if not self.adaptive:
            # Fixed-width: single quantile threshold  (Eq. 4)
            self._threshold = self._conformal_quantile(residuals)
        else:
            # Adaptive: K bins based on Mahalanobis score quantiles (Eq. 5)
            if mahal_scores is None:
                raise ValueError("adaptive=True requires mahal_scores for calibration.")
            mahal_scores = np.asarray(mahal_scores, dtype=np.float64)

            bin_quantiles = np.linspace(0, 1, self.n_bins + 1)
            self._bin_edges = np.quantile(mahal_scores, bin_quantiles)
            self._bin_thresholds = np.zeros(self.n_bins, dtype=np.float64)

            for k in range(self.n_bins):
                lo_edge = self._bin_edges[k]
                hi_edge = self._bin_edges[k + 1]
                # Include upper edge in last bin
                if k < self.n_bins - 1:
                    mask = (mahal_scores >= lo_edge) & (mahal_scores < hi_edge)
                else:
                    mask = mahal_scores >= lo_edge

                m_k = mask.sum()
                if m_k < 10:
                    # Too few calibration samples in this bin — fall back to global threshold
                    self._bin_thresholds[k] = self._conformal_quantile(residuals)
                    if m_k > 0:
                        print(f"  [WARN] Bin {k} has only {m_k} calibration samples; "
                              "using global threshold for this bin.")
                else:
                    self._bin_thresholds[k] = self._conformal_quantile(residuals[mask])

        self._is_calibrated = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        mos_pred: np.ndarray,
        mahal_scores: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Produce conformal prediction intervals for new samples.

        Parameters
        ----------
        mos_pred     : (n,) point MOS predictions
        mahal_scores : (n,) Mahalanobis scores (required if adaptive=True)

        Returns
        -------
        lo, hi : (n,) lower and upper bounds of the prediction interval
        """
        if not self._is_calibrated:
            raise RuntimeError("Wrapper must be calibrated before calling predict().")

        mos_pred = np.asarray(mos_pred, dtype=np.float64)

        if not self.adaptive:
            half_width = self._threshold
            lo = mos_pred - half_width
            hi = mos_pred + half_width
        else:
            if mahal_scores is None:
                raise ValueError("adaptive=True requires mahal_scores for predict().")
            mahal_scores = np.asarray(mahal_scores, dtype=np.float64)

            half_widths = np.zeros_like(mos_pred)
            bin_indices = np.digitize(mahal_scores, self._bin_edges[1:-1])
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

            for k in range(self.n_bins):
                mask = bin_indices == k
                half_widths[mask] = self._bin_thresholds[k]

            lo = mos_pred - half_widths
            hi = mos_pred + half_widths

        return lo.astype(np.float32), hi.astype(np.float32)

    def interval_width(
        self,
        mahal_scores: np.ndarray | None = None,
        n_samples: int = 1,
    ) -> float | np.ndarray:
        """Return mean interval width (scalar for fixed; per-sample for adaptive)."""
        if not self.adaptive:
            return 2.0 * self._threshold
        dummy_pred = np.zeros(n_samples)
        lo, hi = self.predict(dummy_pred, mahal_scores)
        return hi - lo

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def coverage(
        self,
        mos_true: np.ndarray,
        mos_pred: np.ndarray,
        mahal_scores: np.ndarray | None = None,
    ) -> float:
        """Compute empirical marginal coverage on a test set."""
        lo, hi = self.predict(mos_pred, mahal_scores)
        covered = ((mos_true >= lo) & (mos_true <= hi)).mean()
        return float(covered)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[ConformalWrapper] Saved → {path}")

    @staticmethod
    def load(path: str | Path) -> "ConformalWrapper":
        class _WrapperUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                # CLI-created files record the class under ``__main__``.
                if module == "__main__" and name == "ConformalWrapper":
                    return ConformalWrapper
                return super().find_class(module, name)

        with open(path, "rb") as f:
            wrapper = _WrapperUnpickler(f).load()
        return wrapper

    def __repr__(self) -> str:
        status = "calibrated" if self._is_calibrated else "NOT calibrated"
        mode = "adaptive" if self.adaptive else "fixed"
        return (
            f"ConformalWrapper(alpha={self.alpha}, mode={mode}, "
            f"n_bins={self.n_bins}, {status}, n_cal={self._n_cal})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate and save a conformal prediction wrapper."
    )
    p.add_argument("--pred_csv", required=True,
                   help="CSV with columns mos, mos_pred (calibration set — DNS 2021).")
    p.add_argument("--mahal_csv", required=True,
                   help="CSV with column mahal_score for each calibration sample.")
    p.add_argument("--alpha", type=float, default=0.10,
                   help="Miscoverage level (default: 0.10 → 90%% coverage).")
    p.add_argument("--n_bins", type=int, default=5,
                   help="Number of Mahalanobis bins for adaptive interval (default: 5).")
    p.add_argument("--no_adaptive", action="store_true",
                   help="Use fixed-width interval only (no Mahalanobis binning).")
    p.add_argument("--out", required=True,
                   help="Output path for saved wrapper .pkl.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    df_pred = pd.read_csv(args.pred_csv)
    df_mahal = pd.read_csv(args.mahal_csv)

    # Merge on filepath if both have it, otherwise assume same order
    if "filepath" in df_pred.columns and "filepath" in df_mahal.columns:
        df = df_pred.merge(df_mahal[["filepath", "mahal_score"]], on="filepath", how="inner")
    else:
        df = df_pred.copy()
        df["mahal_score"] = df_mahal["mahal_score"].to_numpy()

    mos_true = df["mos"].to_numpy(dtype=np.float32)
    mos_pred = df["mos_pred"].to_numpy(dtype=np.float32)
    mahal = df["mahal_score"].to_numpy(dtype=np.float32)

    print(f"[conformal_wrapper] Calibrating on {len(df)} samples ...")
    print(f"  alpha={args.alpha}, adaptive={not args.no_adaptive}, n_bins={args.n_bins}")

    # Fit both fixed and adaptive wrappers and save
    out = Path(args.out)

    wrapper_fixed = ConformalWrapper(alpha=args.alpha, adaptive=False)
    wrapper_fixed.calibrate(mos_true, mos_pred, mahal)
    wrapper_fixed.save(out.with_stem(out.stem + "_fixed"))
    print(f"  Fixed-width threshold: {wrapper_fixed._threshold:.4f} MOS")

    if not args.no_adaptive:
        wrapper_adapt = ConformalWrapper(alpha=args.alpha, adaptive=True, n_bins=args.n_bins)
        wrapper_adapt.calibrate(mos_true, mos_pred, mahal)
        wrapper_adapt.save(out.with_stem(out.stem + "_adaptive"))
        print(f"  Adaptive bin thresholds: {wrapper_adapt._bin_thresholds.round(4).tolist()}")


if __name__ == "__main__":
    main()
