"""
ECAPA-TDNN Speaker Encoder for Voice Cloning

ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation –
Time Delay Neural Network) achieves state-of-the-art speaker verification.

Architecture:
  Input:  log-mel filterbank, 80 bins
  Body:   3 × SE-Res2Block (dilation 2, 3, 4) + aggregation
  Pool:   Attentive Statistics Pooling
  Output: 192-dim L2-normalised d-vector

Key equations
─────────────
SE-Res2Block:
  1. Pointwise expansion   h = ReLU(BN(Conv1(x)))
  2. Res2Net multi-scale   y_i = k_i(x_i + y_{i-1}),  i = 2…s
  3. Concatenate + Conv    h' = Conv1(concat(y))
  4. Squeeze-Excitation    s = σ(W₂ ReLU(W₁ AvgPool(h')))
                           z = h' ⊙ s
  5. Residual add          out = z + x

Attentive Statistics Pooling:
  e_t = v^T tanh(W h_t + b)         attention energy
  α_t = exp(e_t) / Σ_τ exp(e_τ)     softmax
  μ   = Σ_t α_t h_t
  σ   = √(Σ_t α_t h_t² − μ²)

GE2E training loss (Generalized End-to-End):
  s_{ji}  = cos(e_{ji}, c_j)          similarity to own centroid
  L(e_{ji}) = -s_{ji} + log Σ_k exp s_{ji,k}   (softmax loss)

References:
  Desplanques et al. "ECAPA-TDNN: Emphasized Channel Attention,
    Propagation and Aggregation in TDNN Based Speaker Verification"
    INTERSPEECH 2020
  Wan et al. "Generalized End-to-End Loss for Speaker Verification"
    ICASSP 2018
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import torchaudio


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TDNNLayer(nn.Module):
    """1-D TDNN layer (dilated Conv1d + BatchNorm + ReLU)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 1, dilation: int = 1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class Res2Block(nn.Module):
    """
    Res2Net multi-scale feature extraction.

    Splits channels into s groups; each group applies a 1-D conv and
    accumulates the output from the previous group:
        y_1 = x_1
        y_i = conv_i(x_i + y_{i-1}),  i = 2…s

    The effective receptive field grows exponentially with depth.
    """

    def __init__(self, channels: int, scale: int = 8, kernel: int = 3, dilation: int = 1):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        width = channels // scale
        pad = (kernel - 1) * dilation // 2
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel, dilation=dilation, padding=pad)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.scale, dim=1)
        ys = [chunks[0]]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            xi = chunks[i + 1] + ys[-1] if i > 0 else chunks[i + 1]
            ys.append(F.relu(bn(conv(xi))))
        return torch.cat(ys, dim=1)


class SqueezeExcitation(nn.Module):
    """
    Channel-wise recalibration:
        s = σ(W₂ ReLU(W₁ AvgPool(x)))
        y = x ⊙ s
    Bottleneck ratio r controls capacity vs. parameter count.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = self.fc(x.mean(dim=-1))   # (B, C)
        return x * s.unsqueeze(-1)


class SERes2Block(nn.Module):
    """
    SE-Res2Block — the core building block of ECAPA-TDNN.

    Pre-activation residual + Res2Net multi-scale + SE gating.
    """

    def __init__(self, channels: int, kernel: int, dilation: int, scale: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.res2 = Res2Block(channels, scale=scale, kernel=kernel, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv3 = nn.Conv1d(channels, channels, 1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.se = SqueezeExcitation(channels)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.res2(h)))
        h = self.bn3(self.conv3(h))
        h = self.se(h)
        if context is not None:
            h = h + context
        return F.relu(h + residual)


class AttentiveStatisticsPooling(nn.Module):
    """
    Temporal pooling with learned attention weights.

        e_t = w^T tanh(Wh_t + b)    (attention energy)
        α_t = softmax(e_t)
        μ   = Σ α_t h_t
        σ   = √(Σ α_t h_t² − μ²)
        out = concat(μ, σ)

    Output size = 2 × in_channels.
    """

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_channels * 3, hidden, 1),
            nn.Tanh(),
            nn.Conv1d(hidden, in_channels, 1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        t = x.size(-1)
        global_mean = x.mean(dim=-1, keepdim=True).expand_as(x)
        global_std = x.std(dim=-1, keepdim=True).expand_as(x)
        ctx = torch.cat([x, global_mean, global_std], dim=1)
        alpha = self.attn(ctx)            # (B, C, T)
        mu = (alpha * x).sum(dim=-1)
        var = (alpha * x ** 2).sum(dim=-1) - mu ** 2
        sigma = (var.clamp(min=1e-9)).sqrt()
        return torch.cat([mu, sigma], dim=1)   # (B, 2C)


# ---------------------------------------------------------------------------
# ECAPA-TDNN
# ---------------------------------------------------------------------------

class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN speaker encoder.

    Input:  (B, T, 80) log-mel spectrogram
    Output: (B, emb_dim) L2-normalised d-vector
    """

    def __init__(
        self,
        in_channels: int = 80,
        channels: int = 1024,
        emb_dim: int = 192,
        scale: int = 8,
    ):
        super().__init__()
        self.layer1 = TDNNLayer(in_channels, channels, kernel=5)

        self.layer2 = SERes2Block(channels, kernel=3, dilation=2, scale=scale)
        self.layer3 = SERes2Block(channels, kernel=3, dilation=3, scale=scale)
        self.layer4 = SERes2Block(channels, kernel=3, dilation=4, scale=scale)

        # Multi-layer feature aggregation: concatenate all SE-Res2Block outputs
        cat_ch = channels * 3
        self.layer5 = TDNNLayer(cat_ch, 1536, kernel=1)

        self.pool = AttentiveStatisticsPooling(1536, hidden=256)

        self.bn = nn.BatchNorm1d(1536 * 2)
        self.fc = nn.Linear(1536 * 2, emb_dim)
        self.bn_out = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 80) or (B, 80, T)
        Returns: (B, emb_dim) unit-normalised speaker embedding.
        """
        if x.dim() == 3 and x.shape[1] != 80:
            x = x.transpose(1, 2)          # (B, 80, T)

        h1 = self.layer1(x)

        h2 = self.layer2(h1)
        h3 = self.layer3(h2 + h1)
        h4 = self.layer4(h3 + h2 + h1)

        cat = torch.cat([h2, h3, h4], dim=1)
        h5 = self.layer5(cat)

        pooled = self.pool(h5)             # (B, 3072)
        pooled = self.bn(pooled)
        emb = self.fc(pooled)
        emb = self.bn_out(emb)
        return F.normalize(emb, p=2, dim=1)


# ---------------------------------------------------------------------------
# GE2E Loss (Generalized End-to-End)
# ---------------------------------------------------------------------------

class GE2ELoss(nn.Module):
    """
    Generalised End-to-End Speaker Verification Loss (Wan et al., 2018).

    Given a batch of N speakers × M utterances each:

        c_j = (1 / M) Σ_m e_{jm}            speaker centroid

    Softmax version:
        L(e_{ji}) = -s(e_{ji}, c_j)
                  + log Σ_k exp s(e_{ji}, c_k)

    where s(a, b) = w · cos(a, b) + b   (learnable scale/bias).
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (N, M, D)  — N speakers, M utterances each, D-dim embeddings.
        Returns scalar loss.
        """
        N, M, D = embeddings.shape
        # Centroids
        centroids = embeddings.mean(dim=1)  # (N, D)
        centroids = F.normalize(centroids, p=2, dim=1)
        embs = F.normalize(embeddings.view(N * M, D), p=2, dim=1)

        # Similarity matrix (N*M, N)
        sim = torch.matmul(embs, centroids.T)  # (N*M, N)
        sim = self.w.abs() * sim + self.b

        # Labels: each utterance belongs to its speaker
        labels = torch.arange(N, device=embeddings.device).repeat_interleave(M)
        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# SpeakerEncoder API (feature extraction + embedding)
# ---------------------------------------------------------------------------

class SpeakerEncoder(nn.Module):
    """
    Wraps ECAPA-TDNN for inference.

    Handles:
      - Log-mel spectrogram extraction from raw waveforms
      - Long-utterance segmentation & averaging
      - Returns L2-normalised 192-dim d-vectors
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        channels: int = 1024,
        emb_dim: int = 192,
        device: str = "cpu",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.device_ = device

        self.model = ECAPA_TDNN(in_channels=n_mels, channels=channels, emb_dim=emb_dim)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            f_min=20.0,
            f_max=7600.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.to(device)

    def extract_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (T,) or (1, T) at self.sample_rate → (1, n_mels, frames)."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = self.mel_transform(wav.to(self.device_))
        mel = self.amplitude_to_db(mel)
        mel = (mel + 40.0) / 40.0  # rough normalisation to ~[-1, 1]
        return mel  # (1, n_mels, T)

    @torch.no_grad()
    def embed_utterance(self, wav: torch.Tensor, segment_len: int = 160) -> torch.Tensor:
        """
        Compute a single d-vector for an utterance.

        Long utterances are split into overlapping segments; embeddings are
        mean-pooled then re-normalised — this matches the protocol used in
        the original GE2E paper.

        wav: (T,) raw 16 kHz waveform.
        Returns: (emb_dim,) unit-norm embedding.
        """
        mel = self.extract_mel(wav)  # (1, n_mels, T_frames)
        T = mel.shape[-1]

        if T <= segment_len:
            emb = self.model(mel.transpose(1, 2))  # (1, D)
            return emb.squeeze(0)

        # Sliding window
        stride = segment_len // 2
        starts = list(range(0, T - segment_len + 1, stride))
        segs = torch.stack([mel[0, :, s : s + segment_len].T for s in starts])  # (S, T, C)
        embs = self.model(segs)  # (S, D)
        mean_emb = embs.mean(dim=0)
        return F.normalize(mean_emb, p=2, dim=0)

    @torch.no_grad()
    def embed_utterance_numpy(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Accept a NumPy waveform and return NumPy embedding."""
        import torchaudio.functional as AF
        t = torch.from_numpy(wav.astype(np.float32))
        if sr != self.sample_rate:
            t = AF.resample(t, sr, self.sample_rate)
        return self.embed_utterance(t).cpu().numpy()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Direct forward — mel: (B, T, n_mels) → (B, emb_dim)."""
        return self.model(mel)
