"""
extract_features.py — Extract penultimate-layer features from MOS predictors.

Supported models:
  - dnsmos  : DNSMOS P.835 (Microsoft) — ONNX model
  - nisqa   : NISQA v2 (PyTorch)
  - utmos   : UTMOS (UTokyo-SaruLab, PyTorch + wav2vec2)

For each audio file in the input CSV, this script:
  1. Loads the audio (16 kHz, mono).
  2. Runs a forward pass through the model.
  3. Extracts the penultimate-layer activation vector.
  4. Saves features as a numpy .npz archive alongside a companion CSV.

Output:
  <out_dir>/features_<model>_<dataset>.npz
    Keys: "features" (N x D float32), "filepaths" (N,), "mos" (N,)
  <out_dir>/features_<model>_<dataset>.csv
    Columns: filepath, mos, dataset, feat_0 ... feat_{D-1}  (for inspection)

Usage:
  python extract_features.py \\
    --model dnsmos \\
    --model_path /path/to/dnsmos/DNSMOS \\
    --data_csv ../../results/dns2020.csv \\
    --out_dir ../../results/features \\
    --batch_size 32

  python extract_features.py \\
    --model nisqa \\
    --model_path /path/to/NISQA \\
    --data_csv ../../results/dns2020.csv \\
    --out_dir ../../results/features

  python extract_features.py \\
    --model utmos \\
    --model_path /path/to/utmos \\
    --data_csv ../../results/dns2020.csv \\
    --out_dir ../../results/features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import librosa


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(filepath: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target_sr, return mono float32 array."""
    audio, sr = sf.read(filepath, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


# ---------------------------------------------------------------------------
# DNSMOS P.835 feature extractor (ONNX)
# ---------------------------------------------------------------------------

class DNSMOSExtractor:
    """
    Extract penultimate-layer features from DNSMOS P.835 (Microsoft ONNX model).

    The model processes 9.01-second raw audio segments at 16 kHz. For longer
    clips, segments are extracted with 1-second hops and features/predictions
    are averaged over segments (matching the official dnsmos_local.py logic).

    Penultimate feature: output of the second dense layer before the final
    3-way [SIG, BAK, OVR] prediction head — 64-dimensional float32.
    MOS prediction: raw OVRL output (index 2 of the 3-way head), averaged
    over 1-second hops. No polynomial correction is applied.

    Parameters
    ----------
    model_dir : directory containing 'sig_bak_ovr.onnx'
    """

    INPUT_LENGTH_SEC = 9.01
    SAMPLE_RATE = 16000
    INPUT_SAMPLES = int(INPUT_LENGTH_SEC * SAMPLE_RATE)   # 144160
    HOP_SAMPLES = SAMPLE_RATE                              # 1-second hops
    PENULTIMATE_NODE = "mos_estimator_logpow/dense_1/Relu:0"

    def __init__(self, model_dir: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for DNSMOS: pip install onnxruntime")
        try:
            import onnx
        except ImportError:
            raise ImportError("onnx is required for DNSMOS: pip install onnx onnxruntime")

        import onnx
        import onnxruntime as ort

        model_dir = Path(model_dir)
        onnx_path = model_dir / "sig_bak_ovr.onnx"
        if not onnx_path.exists():
            candidates = list(model_dir.glob("*.onnx"))
            if not candidates:
                raise FileNotFoundError(f"No .onnx file found in {model_dir}")
            onnx_path = candidates[0]

        # Standard session for MOS prediction
        self._session = ort.InferenceSession(str(onnx_path))

        # Session with penultimate node added as an extra output
        self._feat_session = self._build_feature_session(onnx_path)

    def _build_feature_session(self, onnx_path: Path):
        """Return an ONNX session that also exposes the penultimate dense output."""
        import onnx
        import onnxruntime as ort
        from io import BytesIO

        model = onnx.load(str(onnx_path))
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                self.PENULTIMATE_NODE, onnx.TensorProto.FLOAT, None
            )
        )
        buf = BytesIO()
        onnx.save(model, buf)
        buf.seek(0)
        return ort.InferenceSession(buf.read())

    def _get_segments(self, audio: np.ndarray) -> list:
        """
        Slice audio into 9.01s segments with 1-second hops.
        Short clips are repeat-padded to reach INPUT_SAMPLES.
        """
        # Repeat-pad to minimum required length
        while len(audio) < self.INPUT_SAMPLES:
            audio = np.concatenate([audio, audio])

        num_hops = max(1, int(np.floor(len(audio) / self.SAMPLE_RATE) - self.INPUT_LENGTH_SEC) + 1)
        segments = []
        for i in range(num_hops):
            start = i * self.HOP_SAMPLES
            seg = audio[start : start + self.INPUT_SAMPLES]
            if len(seg) < self.INPUT_SAMPLES:
                break
            segments.append(seg.astype(np.float32))
        return segments

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Run DNSMOS forward pass and return penultimate-layer feature vector.

        Returns
        -------
        np.ndarray of shape (64,) — mean over hops of dense_1 output
        """
        segments = self._get_segments(audio)
        feats = []
        for seg in segments:
            inp = seg[np.newaxis, :]                           # (1, 144160)
            outs = self._feat_session.run(None, {"input_1": inp})
            # outs[0] = final (1, 3); outs[1] = penultimate (1, 64)
            feats.append(outs[1].reshape(-1))
        return np.mean(feats, axis=0).astype(np.float32)       # (64,)

    def predict(self, audio: np.ndarray) -> float:
        """
        Return the regular (non-personalized) DNSMOS P.835 OVRL score.

        The official DNSMOS implementation applies the polynomial mapping to
        every segment-level raw output before averaging.
        """
        segments = self._get_segments(audio)
        ovrl_scores = []
        for seg in segments:
            inp = seg[np.newaxis, :]
            out = self._session.run(None, {"input_1": inp})
            # out[0][0] = [SIG, BAK, OVR]; OVRL is index 2
            raw_ovrl = float(out[0][0][2])
            mapped_ovrl = -0.06766283 * raw_ovrl**2 + 1.11546468 * raw_ovrl + 0.04602535
            ovrl_scores.append(mapped_ovrl)
        return float(np.mean(ovrl_scores))


# ---------------------------------------------------------------------------
# NISQA v2 feature extractor (PyTorch)
# ---------------------------------------------------------------------------

class NISQAExtractor:
    """
    Extract penultimate-layer features from NISQA v2.

    NISQA v2 is available from: https://github.com/gabrielmittag/NISQA
    The penultimate layer is the output of the self-attention time-dependency
    module (time_dependency) before the per-dimension pooling heads.
    Features are mean-pooled over valid time steps to give a fixed-size
    64-dimensional vector (td_sa_d_model from the checkpoint args).

    Parameters
    ----------
    model_path : path to the NISQA checkpoint (.tar) file,
                 OR path to NISQA repository root (will locate nisqa.tar automatically)
    """

    SAMPLE_RATE = 16000

    def __init__(self, model_path: str | Path) -> None:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for NISQA: pip install torch")
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(model_path)

        # Locate checkpoint — prefer nisqa.tar (speech quality, not TTS)
        if model_path.is_dir():
            # Prefer the main speech quality checkpoint
            candidates = list(model_path.rglob("nisqa.tar"))
            if not candidates:
                candidates = list(model_path.rglob("*.tar")) + list(model_path.rglob("*.pth"))
            if not candidates:
                raise FileNotFoundError(f"No NISQA checkpoint found under {model_path}")
            checkpoint_path = candidates[0]
        else:
            checkpoint_path = model_path

        # Determine NISQA repo root (directory containing nisqa/ package)
        # If checkpoint is at <repo>/weights/nisqa.tar, repo root is <repo>
        repo_root = checkpoint_path.parent
        while repo_root != repo_root.parent:
            if (repo_root / "nisqa").is_dir():
                break
            repo_root = repo_root.parent

        sys.path.insert(0, str(repo_root))
        try:
            from nisqa.NISQA_lib import NISQA_DIM, segment_specs  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"Could not import nisqa.NISQA_lib from {repo_root}. "
                "Ensure the NISQA repository has been cloned."
            ) from exc

        # Load model directly from checkpoint (bypasses nisqaModel dataset loading)
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        self._args = ckpt["args"]
        self._model = NISQA_DIM(self._args).to(self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        # Cache preprocessing parameters
        self._hop_length = int(self.SAMPLE_RATE * self._args["ms_hop_length"])
        self._win_length = int(self.SAMPLE_RATE * self._args["ms_win_length"])
        # Cap fmax at Nyquist to avoid empty mel filter warnings
        self._fmax = min(self._args["ms_fmax"], self.SAMPLE_RATE / 2)

        # Register forward hook on time_dependency (returns tuple (x, n_wins))
        self._hook_output: np.ndarray | None = None
        self._n_wins_hook: int | None = None
        self._model.time_dependency.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, output) -> None:
        # time_dependency returns (x, n_wins); x shape: (batch, max_segs, d_model)
        td_x = output[0] if isinstance(output, tuple) else output
        self._hook_output = td_x.detach().cpu()

    def _preprocess(self, audio: np.ndarray):
        """Convert raw audio (float32, 16 kHz) to segmented mel spectrogram tensors."""
        import torch
        from nisqa.NISQA_lib import segment_specs

        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.SAMPLE_RATE,
            n_fft=self._args["ms_n_fft"],
            hop_length=self._hop_length,
            win_length=self._win_length,
            n_mels=self._args["ms_n_mels"],
            fmax=self._fmax,
            power=1.0,
            norm="slaney",
            htk=False,
        )
        spec = librosa.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)

        x, n_wins = segment_specs(
            None, spec,
            self._args["ms_seg_length"],
            self._args["ms_seg_hop_length"],
            self._args["ms_max_segments"],
        )
        # x: (max_segs, 1, n_mels, seg_length); n_wins: 0-d numpy array
        x_batch = x.unsqueeze(0).float().to(self._device)          # (1, max_segs, 1, n_mels, seg)
        n_wins_t = torch.tensor([int(n_wins)], device=self._device)
        return x_batch, n_wins_t, int(n_wins)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Run NISQA forward pass and return penultimate-layer feature vector.

        Returns
        -------
        np.ndarray of shape (64,) — mean-pooled self-attention output
        """
        import torch
        self._hook_output = None
        x_batch, n_wins_t, n_valid = self._preprocess(audio)
        with torch.no_grad():
            _ = self._model(x_batch, n_wins_t)
        if self._hook_output is None:
            raise RuntimeError("NISQA time_dependency hook did not fire.")
        # Mean-pool over valid time steps only
        feat = self._hook_output[0, :n_valid, :].mean(0).numpy().astype(np.float32)
        return feat

    def predict(self, audio: np.ndarray) -> float:
        """Return NISQA overall MOS score (dimension index 0)."""
        import torch
        x_batch, n_wins_t, _ = self._preprocess(audio)
        with torch.no_grad():
            out = self._model(x_batch, n_wins_t)
        # out shape: (batch, n_dims); dim 0 is overall MOS
        return float(out[0, 0].cpu().item())


# ---------------------------------------------------------------------------
# UTMOS feature extractor (PyTorch + torchaudio wav2vec2, no fairseq)
# ---------------------------------------------------------------------------

def _map_fairseq_key(key: str) -> "str | None":
    """
    Map a fairseq wav2vec 2.0 state-dict key to a torchaudio Wav2Vec2Model key.

    Derived from torchaudio.models.wav2vec2.utils.import_fairseq._map_key.
    Returns None for keys that do not exist in the torchaudio model
    (e.g. mask_emb, quantizer, final_proj).
    """
    import re

    if re.match(r"(mask_emb|quantizer|project_q|final_proj|dropout)", key):
        return None
    # Feature extractor — group norm on first layer
    # "conv_layers.0.2.weight/bias" → "conv_layers.0.layer_norm.weight/bias"
    match = re.match(r"feature_extractor\.conv_layers\.0\.2\.(weight|bias)", key)
    if match:
        return f"feature_extractor.conv_layers.0.layer_norm.{match.group(1)}"
    # Convolutions: "conv_layers.X.0.weight" → "conv_layers.X.conv.weight"
    match = re.match(r"feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)", key)
    if match:
        return f"feature_extractor.conv_layers.{match.group(1)}.conv.{match.group(2)}"
    # Layer-norm feature extractor (layer_norm mode):
    # "conv_layers.X.2.1.weight/bias" → "conv_layers.X.layer_norm.weight/bias"
    match = re.match(r"feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)", key)
    if match:
        return f"feature_extractor.conv_layers.{match.group(1)}.layer_norm.{match.group(2)}"
    # Feature projection
    match = re.match(r"post_extract_proj\.(weight|bias)", key)
    if match:
        return f"encoder.feature_projection.projection.{match.group(1)}"
    match = re.match(r"layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.feature_projection.layer_norm.{match.group(1)}"
    # Transformer — positional conv embedding
    match = re.match(r"encoder\.pos_conv\.0\.(bias|weight_g|weight_v)", key)
    if match:
        return f"encoder.transformer.pos_conv_embed.conv.{match.group(1)}"
    match = re.match(r"encoder\.layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layer_norm.{match.group(1)}"
    # Transformer — self-attention
    match = re.match(
        r"encoder\.layers\.(\d+)\.self_attn\.((k_|v_|q_|out_)proj\.(weight|bias))", key
    )
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.attention.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.layer_norm.{match.group(2)}"
    # Transformer — feed-forward
    match = re.match(r"encoder\.layers\.(\d+)\.fc1\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.feed_forward.intermediate_dense.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.fc2\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.feed_forward.output_dense.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.final_layer_norm.{match.group(2)}"
    # Skip anything else silently (dropout buffers etc.)
    return None


class UTMOSExtractor:
    """
    Extract penultimate-layer features from UTMOS (UTokyo-SaruLab).

    Loads the UTMOS strong-learner checkpoint (epoch=3-step=7459.ckpt) without
    requiring fairseq. The SSL backbone (wav2vec 2.0 small) is loaded via
    torchaudio; the remaining layers (DomainEmbedding, LDConditioner, Projection)
    are reconstructed directly from the checkpoint weights.

    Penultimate feature: mean-pooled BiLSTM output from LDConditioner (1024-dim).
    MOS prediction: mean per-frame Projection output, rescaled to [1, 5].

    Parameters
    ----------
    model_path : directory containing 'epoch=3-step=7459.ckpt', or direct path
                 to the checkpoint .ckpt file.
    """

    SAMPLE_RATE = 16000
    # Standard inference settings from UTMOS score.py
    _JUDGE_ID = 288
    _DOMAIN_ID = 0

    def __init__(self, model_path: str | Path) -> None:
        try:
            import torch
            import torch.nn as nn
            import torchaudio
        except ImportError as exc:
            raise ImportError(
                "PyTorch and torchaudio are required for UTMOS: pip install torch torchaudio"
            ) from exc
        import torch
        import torch.nn as nn
        import torchaudio

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(model_path)

        # Locate checkpoint file
        if model_path.is_dir():
            candidates = sorted(model_path.rglob("epoch=*.ckpt"))
            if not candidates:
                candidates = sorted(model_path.rglob("*.ckpt"))
            if not candidates:
                raise FileNotFoundError(f"No .ckpt checkpoint found under {model_path}")
            ckpt_path = candidates[0]
        else:
            ckpt_path = model_path

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]

        # ------------------------------------------------------------------
        # 1. SSL backbone: torchaudio wav2vec2_base with weights from checkpoint
        # ------------------------------------------------------------------
        ssl_model = torchaudio.models.wav2vec2.wav2vec2_base()
        ssl_prefix = "feature_extractors.0.ssl_model."
        ssl_sd = {}
        for k, v in sd.items():
            if not k.startswith(ssl_prefix):
                continue
            fairseq_key = k[len(ssl_prefix):]
            ta_key = _map_fairseq_key(fairseq_key)
            if ta_key is not None:
                ssl_sd[ta_key] = v
        incompatible = ssl_model.load_state_dict(ssl_sd, strict=False)
        missing_parameters = [
            key for key in incompatible.missing_keys
            if key in dict(ssl_model.named_parameters())
        ]
        if missing_parameters or incompatible.unexpected_keys:
            raise RuntimeError(
                "Incomplete UTMOS wav2vec2 checkpoint conversion. "
                f"Missing parameters: {missing_parameters}; "
                f"unexpected parameters: {incompatible.unexpected_keys}"
            )
        ssl_model.eval()
        self._ssl = ssl_model.to(self._device)

        # ------------------------------------------------------------------
        # 2. Domain embedding: Embedding(3, 128)
        # ------------------------------------------------------------------
        domain_emb = nn.Embedding(3, 128)
        domain_emb.weight.data.copy_(sd["feature_extractors.1.embedding.weight"])
        domain_emb.eval()
        self._domain_emb = domain_emb.to(self._device)

        # ------------------------------------------------------------------
        # 3. Judge embedding + BiLSTM (LDConditioner)
        #    LSTM input: ssl(768) + domain(128) + judge(128) = 1024
        #    hidden_size = 512, bidirectional → output = 1024
        # ------------------------------------------------------------------
        judge_emb = nn.Embedding(3000, 128)
        judge_emb.weight.data.copy_(sd["output_layers.0.judge_embedding.weight"])
        judge_emb.eval()
        self._judge_emb = judge_emb.to(self._device)

        bilstm = nn.LSTM(
            input_size=1024, hidden_size=512, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        for suffix in (
            "weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
            "weight_ih_l0_reverse", "weight_hh_l0_reverse",
            "bias_ih_l0_reverse", "bias_hh_l0_reverse",
        ):
            getattr(bilstm, suffix).data.copy_(
                sd[f"output_layers.0.decoder_rnn.{suffix}"]
            )
        bilstm.eval()
        self._bilstm = bilstm.to(self._device)

        # ------------------------------------------------------------------
        # 4. Projection head: Linear(1024,2048) → ReLU → Dropout(0.3) → Linear(2048,1)
        # ------------------------------------------------------------------
        proj = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1),
        )
        proj[0].weight.data.copy_(sd["output_layers.1.net.0.weight"])
        proj[0].bias.data.copy_(sd["output_layers.1.net.0.bias"])
        proj[3].weight.data.copy_(sd["output_layers.1.net.3.weight"])
        proj[3].bias.data.copy_(sd["output_layers.1.net.3.bias"])
        proj.eval()
        self._proj = proj.to(self._device)

    def _forward(self, audio: np.ndarray):
        """
        Shared forward pass.

        Returns
        -------
        lstm_out : torch.Tensor of shape (1, T_frames, 1024)
        mos_raw  : torch.Tensor of shape (1, T_frames, 1) — unnormalised MOS
        """
        import torch
        wav = torch.from_numpy(audio).unsqueeze(0).to(self._device)  # (1, T_samples)
        with torch.no_grad():
            ssl_feat, _ = self._ssl(wav)                              # (1, T_frames, 768)
            T = ssl_feat.size(1)
            domain_feat = self._domain_emb(
                torch.tensor([self._DOMAIN_ID], device=self._device)  # (1, 128)
            )
            judge_feat = self._judge_emb(
                torch.tensor([self._JUDGE_ID], device=self._device)   # (1, 128)
            )
            combined = torch.cat([
                ssl_feat,
                domain_feat.unsqueeze(1).expand(-1, T, -1),
                judge_feat.unsqueeze(1).expand(-1, T, -1),
            ], dim=2)                                                  # (1, T, 1024)
            lstm_out, _ = self._bilstm(combined)                       # (1, T, 1024)
            mos_raw = self._proj(lstm_out)                             # (1, T, 1)
        return lstm_out, mos_raw

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Run UTMOS forward pass and return penultimate-layer feature vector.

        Returns
        -------
        np.ndarray of shape (1024,) — mean-pooled BiLSTM (LDConditioner) output
        """
        lstm_out, _ = self._forward(audio)
        feat = lstm_out[0].mean(0).cpu().numpy().astype(np.float32)   # (1024,)
        return feat

    def predict(self, audio: np.ndarray) -> float:
        """
        Return UTMOS MOS score rescaled to [1, 5].

        Training used NormalizeScore with org_max=5.0, org_min=1.0,
        normalize_to_max=1.0, normalize_to_min=-1.0. Inverse: score = raw * 2 + 3.
        """
        _, mos_raw = self._forward(audio)
        return float(mos_raw.mean().item() * 2.0 + 3.0)


# ---------------------------------------------------------------------------
# Extractors registry
# ---------------------------------------------------------------------------

_EXTRACTOR_CLS = {
    "dnsmos": DNSMOSExtractor,
    "nisqa": NISQAExtractor,
    "utmos": UTMOSExtractor,
}


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_features(
    model_name: str,
    model_path: str | Path,
    data_csv: str | Path,
    out_dir: str | Path,
    batch_size: int = 1,
) -> Path:
    """
    Extract penultimate-layer features for all samples in data_csv.

    Returns path to the saved .npz file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    dataset_name = df["dataset"].iloc[0] if "dataset" in df.columns else "unknown"
    out_npz = out_dir / f"features_{model_name}_{dataset_name}.npz"
    out_csv = out_dir / f"features_{model_name}_{dataset_name}.csv"

    if model_name not in _EXTRACTOR_CLS:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {list(_EXTRACTOR_CLS)}")

    print(f"[extract_features] Loading {model_name} from {model_path} ...")
    extractor = _EXTRACTOR_CLS[model_name](model_path)

    all_features = []
    all_preds = []
    valid_idx = []

    for i, row in df.iterrows():
        filepath = row["filepath"]
        try:
            audio = load_audio(filepath)
            feat = extractor.extract(audio)
            pred = extractor.predict(audio)
            all_features.append(feat)
            all_preds.append(pred)
            valid_idx.append(i)
        except Exception as exc:
            print(f"  [WARN] Skipping {filepath}: {exc}", file=sys.stderr)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(df)} samples ...")

    df_valid = df.loc[valid_idx].copy()
    df_valid["mos_pred"] = all_preds

    features_array = np.stack(all_features, axis=0).astype(np.float32)
    mos_array = df_valid["mos"].to_numpy(dtype=np.float32)
    filepaths_array = df_valid["filepath"].to_numpy()

    np.savez_compressed(
        out_npz,
        features=features_array,
        mos=mos_array,
        mos_pred=np.array(all_preds, dtype=np.float32),
        filepaths=filepaths_array,
        dataset=np.array([dataset_name]),
    )
    df_valid.to_csv(out_csv, index=False)

    print(f"[extract_features] {len(df_valid)}/{len(df)} samples → {out_npz}")
    print(f"  Feature dimension: {features_array.shape[1]}")
    return out_npz


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract penultimate-layer MOS predictor features.")
    p.add_argument("--model", required=True, choices=list(_EXTRACTOR_CLS),
                   help="Which MOS predictor to use.")
    p.add_argument("--model_path", required=True,
                   help="Path to model directory or checkpoint file.")
    p.add_argument("--data_csv", required=True,
                   help="Input CSV from load_dns.py / load_urgent.py / load_odaq.py.")
    p.add_argument("--out_dir", default="../../results/features",
                   help="Directory to save .npz and .csv feature files.")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size for inference (currently only batch_size=1 is supported).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    extract_features(
        model_name=args.model,
        model_path=args.model_path,
        data_csv=args.data_csv,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
