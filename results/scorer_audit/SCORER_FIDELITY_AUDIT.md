# Scorer fidelity audit

Date: 2026-08-18

## DNSMOS P.835

The paper artifacts use the official non-personalized OVRL polynomial

`-0.06766283 r^2 + 1.11546468 r + 0.04602535`

applied to the stored clip-mean raw prediction. The official implementation applies the polynomial to each 9.01-second segment and then averages the mapped scores. These operations differ for multi-segment clips because the mapping is nonlinear.

An exact full-subset check was run on all 240 T-FOR clips. The exact segment-wise results were compared with the stored clip-mean approximation and then the canonical artifact was restored to the common approximation so every dataset uses one consistent scoring rule.

| Quantity | Result |
|---|---:|
| Mean absolute prediction difference | 0.00444 MOS |
| 95th percentile absolute difference | 0.02815 MOS |
| Maximum absolute difference | 0.09675 MOS |
| Approximate RMSE | 0.84466 MOS |
| Exact RMSE | 0.84411 MOS |
| Exact minus approximate RMSE | -0.00056 MOS |
| Approximate SRCC | 0.54066 |
| Exact SRCC | 0.54384 |

This check supports the statement that the approximation has negligible effect on the aggregate conclusions. It does not make the approximate predictions identical to official segment-wise inference. A full exact rerun remains preferable before archival release if compute time permits. The measured runtime was about 5.6 minutes for 240 clips on the current CPU setup, implying several hours for all 14,432 clips.

## NISQA

The custom extractor loads the official `nisqa.tar` checkpoint and reconstructs the published NISQA preprocessing and prediction path. The bundled reference command successfully loaded the checkpoint and entered prediction on both CUDA and CPU. A single-file reference run did not finish within bounded runs of 10 minutes on CUDA or 5 minutes on CPU in the current shared environment. No numerical discrepancy was observed, but direct reference-command equivalence remains unverified.

## UTMOS

The custom extractor loads the official strong-learner checkpoint. It maps the checkpoint wav2vec 2.0 parameters into the corresponding `torchaudio` modules, requires that no trainable wav2vec parameters are missing, copies the domain embedding, judge embedding, bidirectional LSTM, and projection weights, uses domain ID 0 and judge ID 288, averages frame outputs, and applies the official `2y+3` rescaling.

The bundled official scorer cannot run in the current Python 3.13 environment because its pinned stack requires `fairseq` and PyTorch 1.11. Structural and weight-level checks pass, but direct numerical equivalence with the original `fairseq` execution remains unverified. The manuscript should not claim that this conversion is independently validated against the official runtime.

## External-data inventory

The canonical evaluation uses all seven NISQA subsets. T-SIM is used only as the feature reference, V-SIM is split into calibration and held-out in-domain evaluation, and T-LIV, V-LIV, T-FOR, T-TLK, and T-P501 are shifted evaluations.

The official URGENT 2024 MOS release was selected for independent evaluation because it contains enhanced speech, a compatible 1--5 MOS scale, eight raw ratings per clip, and 23 enhancement systems. The complete release has 6,900 clips under CC BY-NC-SA 4.0. The downloader in `code/data_prep/download_urgent2024_mos_subset.py` uses seed 20270818 and a label-blind SHA-256 ordering within each system. The current CSV contains 46 clips, two per system, while the initially downloaded ten-per-system superset leaves 230 audio files available locally.

External feature extraction is not complete. Atomic NISQA runs at 230, 115, and 46 clips each exceeded the one-hour execution bound before writing an artifact. DNSMOS hidden-output inference was also prohibitively slow in the current ONNX environment. No partial predictions were inspected and no external result is reported in the paper. The next implementation step is checkpointed, resumable extraction followed by evaluation with the frozen NISQA reference and V-SIM calibration. ODAQ remains unsuitable as the primary external corpus because it is not speech-specific and uses a MUSHRA-like scale.
