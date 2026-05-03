"""
HiFi-GAN Neural Vocoder

Converts latent representation z (or mel-spectrogram) to high-fidelity waveforms.

Architecture
────────────
Generator:
  z (or mel) → Upsample × L → Multi-Receptive Field Fusion (MRF) → Conv → tanh

MRF = sum of K ResBlocks with different kernel sizes and dilation patterns:
  ResBlock_k(x) = x + Σ_r (dilated_conv_r ∘ LeakyReLU ∘ dilated_conv_r)(x)

Discriminators (adversarial training signal):
  Multi-Period Discriminator (MPD): K sub-discriminators, each on period-p samples.
  Multi-Scale Discriminator (MSD): 3 sub-discriminators at scales 1×, 2×, 4×.

Losses
──────
Adversarial generator:  L_adv = E[(D(ŷ) − 1)²]                     (LS-GAN)
Adversarial discrim.:   L_D   = E[(D(y)−1)² + D(ŷ)²]
Feature matching:       L_fm  = (1/KL) Σ_{k,l} ‖D_k^l(y) − D_k^l(ŷ)‖₁
Mel reconstruction:     L_mel = ‖mel(y) − mel(ŷ)‖₁

References:
  Kong et al. "HiFi-GAN: Generative Adversarial Networks for Efficient
    and High Fidelity Speech Synthesis" NeurIPS 2020.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

LRELU_SLOPE = 0.1


# ---------------------------------------------------------------------------
# Generator building blocks
# ---------------------------------------------------------------------------

class ResBlock1(nn.Module):
    """
    Residual block with 3 dilation levels.

    For each dilation d in dilations:
        h = LeakyReLU(x)
        h = Conv(h)   [dilated, padding = (k-1)*d/2]
        h = LeakyReLU(h)
        h = Conv(h)   [dilation = 1]
        x = x + h
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, dilation=d,
                padding=(kernel_size * d - d) // 2,
            ))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, dilation=1,
                padding=(kernel_size - 1) // 2,
            ))
            for _ in dilation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c in self.convs1 + self.convs2:
            nn.utils.remove_weight_norm(c)


class ResBlock2(nn.Module):
    """Simplified residual block with 2 dilation levels."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple = (1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, dilation=d,
                padding=(kernel_size * d - d) // 2,
            ))
            for d in dilation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            x = x + c(xt)
        return x

    def remove_weight_norm(self):
        for c in self.convs:
            nn.utils.remove_weight_norm(c)


# ---------------------------------------------------------------------------
# HiFi-GAN Generator
# ---------------------------------------------------------------------------

class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN V1 generator.

    z → pre-conv → [upsample → MRF] × L → post-conv → tanh → waveform

    Upsampling rates multiply together to give the total hop length:
      hop = Π upsample_rates  (e.g. 8 × 8 × 2 × 2 = 256)

    MRF at each scale computes:
      o = (1/K) Σ_k ResBlock_k(x)
    """

    def __init__(
        self,
        in_channels: int = 80,
        resblock: str = "1",
        resblock_kernel_sizes: tuple = (3, 7, 11),
        resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates: tuple = (8, 8, 2, 2),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: tuple = (16, 16, 4, 4),
        gin_channels: int = 0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        ResBlock = ResBlock1 if resblock == "1" else ResBlock2

        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        )

        self.ups = nn.ModuleList()
        in_ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(nn.utils.weight_norm(
                nn.ConvTranspose1d(in_ch, out_ch, k, stride=u, padding=(k - u) // 2)
            ))
            in_ch = out_ch

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

        if gin_channels:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            # MRF: average over ResBlocks at this scale
            xs = sum(
                self.resblocks[i * self.num_kernels + j](x)
                for j in range(self.num_kernels)
            ) / self.num_kernels

        x = F.leaky_relu(xs, LRELU_SLOPE)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        nn.utils.remove_weight_norm(self.conv_post)
        for up in self.ups:
            nn.utils.remove_weight_norm(up)
        for rb in self.resblocks:
            rb.remove_weight_norm()


# ---------------------------------------------------------------------------
# Discriminators
# ---------------------------------------------------------------------------

class DiscriminatorP(nn.Module):
    """
    Period sub-discriminator.

    Reshape waveform into (B, 1, T/p, p) and apply 2-D convolutions.
    Each period p extracts different harmonic structure.
    """

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3,
                 use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1, 32,   (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        # Pad to multiple of period
        if t % self.period:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t += pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    """MPD: 5 sub-discriminators with periods [2, 3, 5, 7, 11]."""

    def __init__(self, periods: tuple = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[List, List, List, List]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            yr, fmap_r = d(y)
            yg, fmap_g = d(y_hat)
            y_d_rs.append(yr)
            y_d_gs.append(yg)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    """Scale sub-discriminator (1-D convolutions on raw/downsampled waveform)."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1,    128,  15, 1,  padding=7)),
            norm(nn.Conv1d(128,  128,  41, 2,  padding=20, groups=4)),
            norm(nn.Conv1d(128,  256,  41, 2,  padding=20, groups=16)),
            norm(nn.Conv1d(256,  512,  41, 4,  padding=20, groups=16)),
            norm(nn.Conv1d(512,  1024, 41, 4,  padding=20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 41, 1,  padding=20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 5,  1,  padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for c in self.convs:
            x = F.leaky_relu(c(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    """MSD: 3 sub-discriminators at original, 2× and 4× downsampling."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[List, List, List, List]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            yr, fmap_r = d(y)
            yg, fmap_g = d(y_hat)
            y_d_rs.append(yr)
            y_d_gs.append(yg)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, List, List]:
    """LS-GAN discriminator loss."""
    loss = 0.0
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, List]:
    """LS-GAN generator loss."""
    loss = 0.0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses


def feature_loss(
    fmap_r: List[List[torch.Tensor]],
    fmap_g: List[List[torch.Tensor]],
) -> torch.Tensor:
    """
    Feature matching loss.

        L_fm = (1 / KL) Σ_{k,l} ‖D_k^l(y) − D_k^l(ŷ)‖₁
    """
    loss = 0.0
    for mr, mg in zip(fmap_r, fmap_g):
        for rl, gl in zip(mr, mg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def mel_spectrogram_loss(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    n_fft: int = 1024,
    num_mels: int = 80,
    sample_rate: int = 22050,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> torch.Tensor:
    """
    Mel-spectrogram reconstruction loss (L1 on log-mel).

        L_mel = ‖mel(y) − mel(ŷ)‖₁
    """
    import torchaudio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_size,
        hop_length=hop_size,
        n_mels=num_mels,
        f_min=fmin,
        f_max=fmax,
    ).to(y.device)

    def mel_log(wav):
        m = mel_transform(wav.squeeze(1))
        return torch.log(torch.clamp(m, min=1e-5))

    return F.l1_loss(mel_log(y_hat), mel_log(y))
