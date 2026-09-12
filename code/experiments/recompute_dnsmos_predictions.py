"""Recompute official polynomial-mapped DNSMOS P.835 predictions.

Existing penultimate features are retained. The previous raw averaged ONNX
output is preserved as ``mos_pred_raw`` in each NPZ and CSV artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import soxr


class DNSMOSPredictor:
    SAMPLE_RATE = 16000
    INPUT_LENGTH_SECONDS = 9.01
    INPUT_SAMPLES = int(SAMPLE_RATE * INPUT_LENGTH_SECONDS)

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def segments(self, audio: np.ndarray) -> list[np.ndarray]:
        while len(audio) < self.INPUT_SAMPLES:
            audio = np.concatenate([audio, audio])
        n_hops = int(np.floor(len(audio) / self.SAMPLE_RATE) - self.INPUT_LENGTH_SECONDS) + 1
        segments = []
        for index in range(max(1, n_hops)):
            start = index * self.SAMPLE_RATE
            segment = audio[start:start + self.INPUT_SAMPLES]
            if len(segment) < self.INPUT_SAMPLES:
                continue
            segments.append(segment.astype(np.float32))
        return segments

    def predict_files(self, paths: list[Path], batch_size: int = 64) -> np.ndarray:
        sums = np.zeros(len(paths), dtype=np.float64)
        counts = np.zeros(len(paths), dtype=np.int64)
        queued_segments: list[np.ndarray] = []
        queued_owners: list[int] = []

        def flush() -> None:
            if not queued_segments:
                return
            batch = np.stack(queued_segments)
            raw = self.session.run(None, {"input_1": batch})[0][:, 2].astype(np.float64)
            mapped = -0.06766283 * raw**2 + 1.11546468 * raw + 0.04602535
            np.add.at(sums, np.asarray(queued_owners), mapped)
            np.add.at(counts, np.asarray(queued_owners), 1)
            queued_segments.clear()
            queued_owners.clear()

        for owner, path in enumerate(paths):
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sample_rate != self.SAMPLE_RATE:
                audio = soxr.resample(audio, sample_rate, self.SAMPLE_RATE, quality="HQ")
            for segment in self.segments(audio):
                queued_segments.append(segment)
                queued_owners.append(owner)
                if len(queued_segments) >= batch_size:
                    flush()
            if (owner + 1) % 500 == 0:
                print(f"  loaded {owner + 1}/{len(paths)} files", flush=True)
        flush()
        if np.any(counts == 0):
            raise RuntimeError("At least one DNSMOS file produced no valid segments")
        return sums / counts


def resolve_audio_path(stored_path: str, project_root: Path) -> Path:
    path = Path(stored_path)
    if path.is_absolute() and path.exists():
        return path
    candidates = (
        project_root / path,
        project_root / "code" / "experiments" / path,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(stored_path)


def update_artifact(
    npz_path: Path, extractor: DNSMOSPredictor, project_root: Path,
    fast_from_stored_mean: bool,
) -> None:
    loaded = np.load(npz_path, allow_pickle=True)
    payload = {key: loaded[key] for key in loaded.files}
    filepaths = payload["filepaths"]
    old_pred = payload["mos_pred"].astype(np.float32)
    raw_pred = payload.get("mos_pred_raw", old_pred).astype(np.float32)
    if fast_from_stored_mean:
        predictions = -0.06766283 * raw_pred.astype(np.float64) ** 2 \
            + 1.11546468 * raw_pred.astype(np.float64) + 0.04602535
        mapping = "official_polynomial_applied_to_stored_clip_mean_raw_output"
    else:
        audio_paths = [resolve_audio_path(str(stored_path), project_root) for stored_path in filepaths]
        predictions = extractor.predict_files(audio_paths)
        mapping = "official_polynomial_applied_per_segment_before_clip_mean"
    payload["mos_pred_raw"] = raw_pred
    payload["mos_pred"] = np.asarray(predictions, dtype=np.float32)
    payload["mos_pred_mapping"] = np.asarray([mapping])
    np.savez_compressed(npz_path, **payload)

    csv_path = npz_path.with_suffix(".csv")
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        if "mos_pred_raw" not in frame:
            frame["mos_pred_raw"] = frame["mos_pred"]
        frame["mos_pred"] = payload["mos_pred"]
        frame.to_csv(csv_path, index=False)
    delta = payload["mos_pred"] - old_pred
    print(
        f"[DNSMOS] {npz_path.name}: mean mapping delta={delta.mean():+.4f}, "
        f"range=[{delta.min():+.4f}, {delta.max():+.4f}]",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--fast_from_stored_mean", action="store_true",
        help="Apply the official polynomial to the stored clip-mean raw score. "
             "This is exact for one-segment clips and approximate otherwise.",
    )
    parser.add_argument(
        "--datasets", nargs="*",
        help="Optional dataset suffixes, for example nisqa_test_for. "
             "By default every DNSMOS feature artifact is updated.",
    )
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    extractor = DNSMOSPredictor(project_root / "models" / "DNSMOS" / "sig_bak_ovr.onnx")
    artifacts = sorted((project_root / "results" / "features").glob("features_dnsmos_*.npz"))
    if args.datasets:
        requested = set(args.datasets)
        artifacts = [
            path for path in artifacts
            if path.stem.removeprefix("features_dnsmos_") in requested
        ]
        found = {path.stem.removeprefix("features_dnsmos_") for path in artifacts}
        missing = requested - found
        if missing:
            raise FileNotFoundError(f"No DNSMOS feature artifacts for: {sorted(missing)}")
    if not artifacts:
        raise FileNotFoundError("No DNSMOS feature artifacts found")
    for artifact in artifacts:
        update_artifact(artifact, extractor, project_root, args.fast_from_stored_mean)


if __name__ == "__main__":
    main()
