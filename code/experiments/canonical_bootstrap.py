"""Paired end-to-end bootstrap for the canonical MOS reliability analysis.

Calibration groups and evaluation groups are resampled independently. Each
replicate refits the fixed radius, adaptive distance-bin edges, and bin-specific
radii. Distance scores themselves remain fixed because the feature reference
set and frozen MOS predictors are not refitted in this study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from bootstrap_ci import extract_groups, build_group_index, bootstrap_group_indices
from canonical_analysis import (
    ID_DATASET,
    SHIFTED_DATASETS,
    exact_conformal_radius,
    fit_intervals,
    predict_intervals,
    mean_interval_score,
    safe_auroc,
    balanced_weights,
)

SEED = 20270818
SCORES = ("mahalanobis", "l2", "knn")
MODELS = ("dnsmos", "nisqa", "utmos")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_score(cache_dir: Path, model: str, dataset: str) -> dict[str, np.ndarray]:
    loaded = np.load(cache_dir / "scores" / f"scores_{model}_{dataset}.npz", allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def percentile(values: list[float]) -> tuple[float, float]:
    return tuple(np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5]))


def make_group_sampler(filepaths: np.ndarray):
    unique, mapping = build_group_index(extract_groups(filepaths))
    return lambda rng: bootstrap_group_indices(unique, mapping, rng)


def interval_metrics(
    target: np.ndarray, pred: np.ndarray, score: np.ndarray, fitted: dict, alpha: float
) -> dict[str, float]:
    lo, hi = predict_intervals(pred, score, fitted)
    return {
        "coverage": float(np.mean((target >= lo) & (target <= hi))),
        "width": float(np.mean(hi - lo)),
        "interval_score": mean_interval_score(target, lo, hi, alpha),
    }


def bootstrap_intervals(
    cache_dir: Path, model: str, score_name: str, alpha: float,
    n_bins: int, n_boot: int, rng: np.random.Generator,
) -> list[dict]:
    split = pd.read_csv(cache_dir / "val_sim_split.csv")
    id_data = load_score(cache_dir, model, ID_DATASET)
    cal_mask = split["role"].eq("calibration").to_numpy()
    eval_mask = ~cal_mask
    cal = {key: value[cal_mask] for key, value in id_data.items()}
    eval_id = {key: value[eval_mask] for key, value in id_data.items()}
    datasets = {ID_DATASET: eval_id}
    datasets.update({dataset: load_score(cache_dir, model, dataset) for dataset in SHIFTED_DATASETS})
    cal_sampler = make_group_sampler(cal["filepaths"])

    def fit(cal_idx: np.ndarray) -> tuple[dict, dict]:
        args = (cal["mos_pred"][cal_idx], cal["mos"][cal_idx], cal[score_name][cal_idx], alpha)
        return fit_intervals(*args, None), fit_intervals(*args, n_bins)

    point_fixed, point_adaptive = fit(np.arange(len(cal["mos"])))
    rows = []
    for dataset, test in datasets.items():
        test_sampler = make_group_sampler(test["filepaths"])
        point = {}
        for method, fitted in (("fixed", point_fixed), ("adaptive", point_adaptive)):
            point[method] = interval_metrics(
                test["mos"], test["mos_pred"], test[score_name], fitted, alpha
            )
        boot_delta = {metric: [] for metric in ("coverage", "width", "interval_score")}
        for _ in range(n_boot):
            cal_idx = cal_sampler(rng)
            test_idx = test_sampler(rng)
            fitted_fixed, fitted_adaptive = fit(cal_idx)
            fixed = interval_metrics(
                test["mos"][test_idx], test["mos_pred"][test_idx],
                test[score_name][test_idx], fitted_fixed, alpha,
            )
            adaptive = interval_metrics(
                test["mos"][test_idx], test["mos_pred"][test_idx],
                test[score_name][test_idx], fitted_adaptive, alpha,
            )
            for metric in boot_delta:
                boot_delta[metric].append(adaptive[metric] - fixed[metric])
        for metric, values in boot_delta.items():
            lo, hi = percentile(values)
            rows.append({
                "model": model,
                "score": score_name,
                "dataset": dataset,
                "n_bins": n_bins,
                "metric": metric,
                "fixed": point["fixed"][metric],
                "adaptive": point["adaptive"][metric],
                "adaptive_minus_fixed": point["adaptive"][metric] - point["fixed"][metric],
                "ci_lo": lo,
                "ci_hi": hi,
                "n_boot": n_boot,
            })
    return rows


def pooled_frame(cache_dir: Path, model: str) -> pd.DataFrame:
    split = pd.read_csv(cache_dir / "val_sim_split.csv")
    eval_mask = split["role"].eq("id_evaluation").to_numpy()
    frames = []
    for dataset in (ID_DATASET, *SHIFTED_DATASETS):
        data = load_score(cache_dir, model, dataset)
        mask = eval_mask if dataset == ID_DATASET else np.ones(len(data["mos"]), dtype=bool)
        frame = pd.DataFrame({
            "dataset": dataset,
            "filepath": data["filepaths"][mask].astype(str),
            "mos": data["mos"][mask],
            "mos_pred": data["mos_pred"][mask],
            **{score: data[score][mask] for score in SCORES},
            "mos_std": data["mos_std"][mask],
            "votes": data["votes"][mask],
        })
        frame["abs_error"] = np.abs(frame["mos"] - frame["mos_pred"])
        frame["standardized_abs_error"] = frame["abs_error"] / (
            frame["mos_std"] / np.sqrt(frame["votes"])
        ).clip(lower=1e-12)
        frame["subjective_ci_tcrit"] = student_t.ppf(0.975, frame["votes"] - 1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_stratified_sampler(pool: pd.DataFrame):
    samplers = []
    for _, frame in pool.groupby("dataset", sort=False):
        samplers.append((frame.index.to_numpy(), make_group_sampler(frame["filepath"].to_numpy())))

    def sample(rng: np.random.Generator) -> np.ndarray:
        return np.concatenate([indices[sampler(rng)] for indices, sampler in samplers])

    return sample


def bootstrap_pooled_error(
    pool: pd.DataFrame, model: str, n_boot: int, rng: np.random.Generator,
    tasks: tuple[str, ...] | None = None,
) -> list[dict]:
    rows = []
    sampler = make_stratified_sampler(pool)
    if tasks is None:
        tasks = (
            "top_quartile", "abs_error_gt_0.75",
            "standardized_error_top_quartile", "outside_subjective_95ci",
        )
    for task in tasks:
        for weighting in ("natural", "domain_balanced"):
            for score_name in SCORES:
                def compute(frame: pd.DataFrame) -> float:
                    if task == "top_quartile":
                        labels = (frame["abs_error"] >= frame["abs_error"].quantile(0.75)).astype(int)
                    elif task == "abs_error_gt_0.75":
                        labels = (frame["abs_error"] > 0.75).astype(int)
                    elif task == "standardized_error_top_quartile":
                        labels = (
                            frame["standardized_abs_error"]
                            >= frame["standardized_abs_error"].quantile(0.75)
                        ).astype(int)
                    else:
                        labels = (
                            frame["standardized_abs_error"] > frame["subjective_ci_tcrit"]
                        ).astype(int)
                    weights = None if weighting == "natural" else balanced_weights(frame["dataset"].to_numpy())
                    return safe_auroc(labels.to_numpy(), frame[score_name].to_numpy(), weights)

                point = compute(pool)
                boots = [compute(pool.loc[sampler(rng)].reset_index(drop=True))
                         for _ in range(n_boot)]
                lo, hi = percentile(boots)
                rows.append({
                    "model": model, "task": task, "weighting": weighting,
                    "score": score_name, "auroc": point, "ci_lo": lo,
                    "ci_hi": hi, "n_boot": n_boot,
                })
    return rows


def update_manifest(cache_dir: Path, n_boot: int, n_bins: int) -> None:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    new_outputs = ["bootstrap_interval_differences.csv", "bootstrap_pooled_error.csv"]
    missing = [name for name in new_outputs if not (cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot register missing bootstrap outputs: {missing}")
    manifest["bootstrap"] = {"seed": SEED, "n_boot": n_boot, "n_bins": n_bins}
    manifest["outputs"] = list(dict.fromkeys([*manifest["outputs"], *new_outputs]))
    manifest.setdefault("code_sha256", {})["canonical_bootstrap.py"] = sha256(Path(__file__).resolve())
    manifest["output_sha256"] = {name: sha256(cache_dir / name) for name in manifest["outputs"]}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="results/canonical")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--manifest_only", action="store_true")
    parser.add_argument("--uncertainty_only", action="store_true")
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)
    if args.manifest_only:
        update_manifest(cache_dir, args.n_boot, args.n_bins)
        print("[bootstrap] Manifest updated", flush=True)
        return
    rng = np.random.default_rng(SEED)

    interval_rows = []
    error_rows = []
    for model in MODELS:
        print(f"[bootstrap] {model}", flush=True)
        if not args.uncertainty_only:
            for score_name in SCORES:
                interval_rows.extend(bootstrap_intervals(
                    cache_dir, model, score_name, args.alpha, args.n_bins, args.n_boot, rng
                ))
        selected_tasks = (
            ("standardized_error_top_quartile", "outside_subjective_95ci")
            if args.uncertainty_only else None
        )
        error_rows.extend(bootstrap_pooled_error(
            pooled_frame(cache_dir, model), model, args.n_boot, rng, selected_tasks
        ))
    interval_path = cache_dir / "bootstrap_interval_differences.csv"
    error_path = cache_dir / "bootstrap_pooled_error.csv"
    if args.uncertainty_only:
        existing = pd.read_csv(error_path)
        existing = existing[~existing["task"].isin(selected_tasks)]
        pd.concat([existing, pd.DataFrame(error_rows)], ignore_index=True).to_csv(error_path, index=False)
    else:
        pd.DataFrame(interval_rows).to_csv(interval_path, index=False)
        pd.DataFrame(error_rows).to_csv(error_path, index=False)
    update_manifest(cache_dir, args.n_boot, args.n_bins)
    print("[bootstrap] Complete", flush=True)


if __name__ == "__main__":
    main()
