# mos-conformal

Conformal prediction wrappers for no-reference MOS predictors.
Companion code for the ICASSP 2027 submission
*Conformal Uncertainty for No-Reference MOS Prediction under Distribution
Shift: A Leakage-Controlled Study*.

---

## What this is

No-reference MOS predictors (DNSMOS P.835, NISQA, UTMOS) return a point
estimate with no signal of when to trust it. This repo tests whether
feature-space distance from a reference distribution can flag unreliable
predictions and set per-utterance conformal interval widths, using a
leakage-controlled design that separates feature-reference fitting,
content-disjoint calibration, and evaluation.

This repo wraps any of the three models with:

- A **split conformal prediction interval** with a finite-sample marginal
  coverage guarantee under exchangeability.
- **Euclidean, Mahalanobis, and nearest-neighbour distance scores** computed
  in penultimate-layer feature space, used both for domain detection and as
  an optional stratification signal for interval width.

At inference each audio clip gets three outputs instead of one:

```
(mos_hat, interval, distance_score)
```

## Key finding

Feature-space distance detects some corpus shifts but does not reliably flag
which individual predictions are wrong. Mean domain-detection AUROC is 0.615
to 0.690 across three distance scores. Mean AUROC for detecting the
highest-error quartile within a domain is only 0.433 to 0.499, near chance.
Distance-stratified conformal intervals help some predictor-domain pairs and
hurt others. Treat feature-space alarms as domain-mismatch flags, not
per-utterance error flags.

---

## Models and dataset

| MOS predictor  | Feature dim | Domain AUROC (Mahalanobis, mean over 5 shifts) |
|----------------|-------------|-------------------------------------------------|
| DNSMOS P.835   | 64          | 0.668                                            |
| NISQA v2       | 64          | 0.803                                            |
| UTMOS          | 1024        | 0.561                                            |

Full per-model and per-score numbers are in the paper (Tables 2 to 4).

Evaluated on the **NISQA Corpus** (Mittag et al. 2021, Zenodo 4728081):
14,432 clips across seven subsets covering a simulated-to-live distribution
shift sequence. TRAIN_SIM (N=10,000) fits the feature reference. VAL_SIM is
split into disjoint calibration and in-domain evaluation halves of 1,250
clips each. TRAIN_LIVE, VAL_LIVE, TEST_FOR, TEST_LIVETALK, and TEST_P501 are
evaluated as metadata-defined shifts.

**Domain detection versus error detection (mean over 15 predictor-shift
pairs):**

| Score           | Domain AUROC | Error AUROC | Storage |
|-----------------|--------------|-------------|---------|
| L2 (Euclidean)  | 0.615        | 0.433       | O(d)    |
| Mahalanobis     | 0.678        | 0.478       | O(d^2)  |
| kNN (k=1)       | 0.690        | 0.499       | O(N)    |

Domain AUROC: can the score separate in-distribution from shifted clips.
Error AUROC: can the score identify which individual predictions are wrong.
All three scores fail at the error-detection task, near chance at 0.43 to
0.50. This dissociation is the paper's central finding and reproduces on an
independent multilingual speech-enhancement corpus (Interspeech 2025 URGENT
Challenge).

---

## Quick start: apply the wrapper to your own MOS predictor

```python
from code.mos_conformal_toolkit import MOSConformalPredictor, BootstrapCI, evaluate_mos
import numpy as np

# 1. Fit on your in-distribution (reference) features
predictor = MOSConformalPredictor(alpha=0.10)   # 90% nominal coverage
predictor.fit_reference(ref_features)           # (N, D) penultimate-layer activations

# 2. Calibrate on a held-out set with ground-truth MOS
predictor.calibrate(cal_features, cal_mos_true, cal_mos_pred)

# 3. Predict: returns a DataFrame with intervals and distance scores
df = predictor.predict(test_features, test_mos_pred)
# columns: mos_pred, ci_low_fixed, ci_high_fixed,
#          ci_low_adaptive, ci_high_adaptive, ood_score

# 4. Evaluate any metric with bootstrap CIs
bci = BootstrapCI(n_boot=2000, groups=talker_ids)  # groups= for cluster bootstrap
point, lo, hi = bci.compute(
    df["ood_score"].values, np.abs(test_mos_pred - test_mos_true),
    metric_fn=lambda s, e: float(np.corrcoef(s, e)[0, 1]),
)
print(f"PCC: {point:.3f}  [{lo:.3f}, {hi:.3f}]")

# 5. All standard metrics at once with CIs
results = evaluate_mos(
    mos_pred=test_mos_pred, mos_true=test_mos_true,
    ood_scores=df["ood_score"].values,
    id_ood_labels=domain_labels,   # 0=ID, 1=OOD (optional)
    groups=talker_ids,
    n_boot=2000,
    extra_metrics={"my_metric": my_fn},   # metric_fn(scores, abs_errors)
)
```

Or from the command line:
```bash
python code/mos_conformal_toolkit.py predict \
    --ref ref_features.npz --cal calibration.npz --test test.npz \
    --alpha 0.10 --out predictions.csv

python code/mos_conformal_toolkit.py evaluate --data results.npz --n_boot 2000
```

Input files: `.npz` with keys `features`, `mos`, `mos_pred`, or `.csv` with
columns `mos_true`, `mos_pred`, `feat_0..feat_{D-1}`.

---

## Repo structure

```
code/
  mos_conformal_toolkit.py  User-facing API: MOSConformalPredictor, BootstrapCI, evaluate_mos
  data_prep/          Dataset loaders (NISQA Corpus, URGENT)
  experiments/
    extract_features.py         Penultimate-layer feature extraction
    fit_mahalanobis.py          Fit Gaussian and Mahalanobis, L2, kNN distance scores
    conformal_wrapper.py        Split conformal calibration and prediction
    evaluate.py                 Full evaluation pipeline (paper tables)
  analysis/
    plot_results.py   Generate all paper figures
  run_pipeline.py     End-to-end pipeline driver
  requirements.txt

paper/
  main.tex            Root LaTeX file
  sections/           Abstract, Introduction, Related Work, Method, Experiments, Conclusion
  tables/             Standalone .tex table files
  figures/            PDF figures (vector)
  refs.bib

results/              CSVs and JSON summaries from each evaluation run
```

---

## Setup

```bash
pip install -r code/requirements.txt
```

Model weights are not included in this repo.
Download and place them as follows:

```
models/DNSMOS/   sig.onnx, bak_ovr.onnx, sig_bak_ovr.onnx
models/NISQA/    (clone https://github.com/gabrielmittag/NISQA)
models/UTMOS/    (clone https://github.com/sarulab-speech/UTMOS22)
```

NISQA Corpus audio files: download from Zenodo 4728081 and place under `data/NISQA_Corpus/`.

---

## Usage

### Feature extraction

```bash
python code/experiments/extract_features.py \
  --model nisqa \
  --data_dir data/NISQA_Corpus \
  --out_dir results/features
```

### Fit Mahalanobis model

```bash
python code/experiments/fit_mahalanobis.py \
  --features_path results/features/features_nisqa_nisqa_train_sim.npz \
  --out_dir results/mahal
```

### Calibrate conformal wrapper

```bash
python code/experiments/conformal_wrapper.py \
  --model nisqa \
  --features_path results/features/features_nisqa_nisqa_train_live.npz \
  --mahal_path results/mahal/mahal_nisqa.npz \
  --out_dir results/conformal
```

### Run full evaluation

```bash
python code/experiments/evaluate.py \
  --features_dir results/features \
  --mahal_dir    results/mahal \
  --wrapper_dir  results/conformal \
  --out_dir      results \
  --models nisqa dnsmos utmos
```

Or run the full pipeline in one step:

```bash
python code/run_pipeline.py
```

---

## Results

All numerical results are in `results/`. The latest evaluation run is
`eval_auroc_20260810_214127.csv`.Canonical tables and figures are
generated from `results/canonical/`.

Conformal coverage at the 90% nominal level, fixed split conformal intervals,
held-out V-SIM: 90.2% (DNSMOS), 90.1% (NISQA), 91.8% (UTMOS). Coverage under
shift ranges from 66.4% to 100.0%, consistent with there being no
arbitrary-shift guarantee. Full per-subset and adaptive-interval numbers are
in the paper (Table 4).

---

## Citation

```
[Anonymized for double-blind review. Citation details added after decisions.

---

## License

Code released under the MIT License.
Model weights and corpus audio files are subject to their respective licenses.
