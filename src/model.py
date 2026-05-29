"""
VITS2: Variational Inference Text-to-Speech with Adversarial Learning

End-to-end TTS model. The core is a Conditional VAE where:
  - Posterior encoder  q_φ(z | x_mel, c_text)  — WaveNet
  - Prior              p_θ(z | c_text)          — Normalizing flows over N(0,I)
  - Decoder            p_θ(x | z)               — HiFi-GAN generator
  - Duration           p(d | c_text)             — Stochastic Duration Predictor

ELBO objective:
  L = E_q[log p_θ(x|z)] - KL[q_φ(z|x,c) ‖ p_θ(z|c)] - log p(d|c)

References:
  Kim et al. "VITS" ICML 2021
  Kim et al. "VITS2" Interspeech 2023
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Utility modules
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Layer normalisation over the channel dimension (last dim)."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)  →  normalise over C
        x = x.transpose(1, -1)  # (B, T, C)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


def convert_pad_shape(pad_shape):
    """Convert nested list to flat list for F.pad."""
    l = pad_shape[::-1]
    return [item for sublist in l for item in sublist]


# ---------------------------------------------------------------------------
# WaveNet dilated residual convolutions (posterior encoder backbone)
# ---------------------------------------------------------------------------

class WN(nn.Module):
    """
    WaveNet-style stack of dilated residual convolutions.

    Each layer computes:
        h = tanh(W_f * x + V_f * g) ⊙ σ(W_g * x + V_g * g)
        y = W_o * h + residual

    Dilation pattern: 1, 2, 4, … 2^(num_layers-1)
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * num_layers, 1)

        for i in range(num_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size - 1) * dilation // 2
            in_layer = nn.Conv1d(
                hidden_channels, 2 * hidden_channels, kernel_size,
                dilation=dilation, padding=padding,
            )
            in_layer = nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            is_last = i == (num_layers - 1)
            res_skip_channels = hidden_channels if is_last else 2 * hidden_channels
            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        num_conds = 0

        if self.gin_channels != 0 and g is not None:
            g = self.cond_layer(g)

        for i in range(self.num_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels]
            else:
                g_l = torch.zeros_like(x_in)

            acts = torch.tanh(x_in[:, : self.hidden_channels] + g_l[:, : self.hidden_channels])
            acts *= torch.sigmoid(x_in[:, self.hidden_channels :] + g_l[:, self.hidden_channels :])
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels]
                x = (x + res_acts) * x_mask
                output += res_skip_acts[:, self.hidden_channels :]
            else:
                output += res_skip_acts

        return output * x_mask

    def remove_weight_norm(self):
        for l in self.in_layers:
            nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            nn.utils.remove_weight_norm(l)


# ---------------------------------------------------------------------------
# Posterior encoder  q_φ(z | x_mel, g_speaker)
# ---------------------------------------------------------------------------

class PosteriorEncoder(nn.Module):
    """
    Encodes mel-spectrogram into latent mean and log-variance.

        μ, log σ = WaveNet(Conv1d(x_mel)) + speaker_g

    During training we sample z ~ N(μ, σ²).
    Not used at inference (only the prior is used).
    """

    def __init__(
        self,
        in_channels: int,       # mel bins (e.g. 513 for linear or 80 for mel)
        inter_channels: int,    # latent dim
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        num_layers: int = 16,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, num_layers,
                      gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, inter_channels * 2, 1)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = _sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(self.inter_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


# ---------------------------------------------------------------------------
# Residual Coupling Block (normalising flows for the prior)
# ---------------------------------------------------------------------------

class ResidualCouplingLayer(nn.Module):
    """
    Affine coupling layer:
        x₁, x₂ = split(x)
        x₂' = x₂ · exp(s(x₁)) + t(x₁)
        y = concat(x₁, x₂')
        log|det J| = Σ s(x₁)
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        assert channels % 2 == 0, "channels must be even for coupling"
        super().__init__()
        self.half = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, num_layers,
                      gin_channels=gin_channels)
        out_channels = self.half if mean_only else channels
        self.post = nn.Conv1d(hidden_channels, out_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = x[:, : self.half], x[:, self.half :]
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = stats[:, : self.half], stats[:, self.half :]
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            log_det = torch.sum(logs * x_mask, dim=[1, 2])
            return torch.cat([x0, x1], dim=1), log_det
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            return torch.cat([x0, x1], dim=1), torch.zeros(x.size(0), device=x.device)


class Flip(nn.Module):
    """Reverses the channel dimension — zero-cost bijection."""

    def forward(self, x, *args, reverse=False, **kwargs):
        return torch.flip(x, [1]), torch.zeros(x.size(0), device=x.device)


class ResidualCouplingBlock(nn.Module):
    """Stack of coupling layers separated by flips."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(ResidualCouplingLayer(
                channels, hidden_channels, kernel_size, dilation_rate,
                num_layers, gin_channels=gin_channels, mean_only=True,
            ))
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        flows = self.flows if not reverse else reversed(self.flows)
        for flow in flows:
            x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x


# ---------------------------------------------------------------------------
# Text Encoder — Transformer with relative positional encoding
# ---------------------------------------------------------------------------

class MultiHeadAttentionRP(nn.Module):
    """
    Multi-head self-attention with Shaw-style relative position bias.

        score_ij = (q_i · k_j + q_i · r_{i−j}) / √d_k

    where r_{i−j} are learnable relative position embeddings clipped to
    ±window_size.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_heads: int,
        window_size: int = 4,
        proximal_bias: bool = False,
        proximal_init: bool = True,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = channels // num_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        n_rel = 2 * window_size + 1
        self.emb_rel_k = nn.Parameter(torch.randn(1, self.head_dim, n_rel) * 0.1)
        self.emb_rel_v = nn.Parameter(torch.randn(1, self.head_dim, n_rel) * 0.1)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, _ = self._attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def _attention(self, query, key, value, mask=None):
        B, C, T_s = query.shape
        T_t = key.shape[2]
        H = self.num_heads
        D = self.head_dim

        q = query.view(B, H, D, T_s).permute(0, 1, 3, 2)  # (B,H,T,D)
        k = key.view(B, H, D, T_t).permute(0, 1, 3, 2)
        v = value.view(B, H, D, T_t).permute(0, 1, 3, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        # Relative position bias
        if self.window_size:
            seq_len = max(T_s, T_t)
            rel_pos = self._relative_positions(T_s, T_t)
            scores_rel = self._relative_position_to_absolute(
                torch.matmul(q, self.emb_rel_k.expand(B * H, -1, -1).view(B, H, D, -1))
            )
            scores = scores + scores_rel

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, v)
        output = output.permute(0, 1, 3, 2).contiguous().view(B, C, T_s)
        return output, p_attn

    def _relative_positions(self, q_len: int, k_len: int) -> torch.Tensor:
        r = torch.arange(q_len, dtype=torch.long)
        c = torch.arange(k_len, dtype=torch.long)
        dist = r.unsqueeze(1) - c.unsqueeze(0)
        return dist.clamp(-self.window_size, self.window_size) + self.window_size

    def _relative_position_to_absolute(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, _ = x.shape
        x = F.pad(x, (0, 0, 1, 0))
        x = x.view(B, H, -1)
        x = F.pad(x, (0, L - 1))
        x = x.view(B, H, L + 1, 2 * L - 1)
        return x[:, :, :L, L - 1 :]


class FFN(nn.Module):
    """Position-wise feed-forward with optional causal (or padding) conv."""

    def __init__(self, in_channels: int, out_channels: int, filter_channels: int,
                 kernel_size: int, p_dropout: float = 0.0, causal: bool = False):
        super().__init__()
        self.causal = causal
        pad = (kernel_size - 1) if causal else (kernel_size - 1) // 2
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=pad)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=pad)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x * x_mask)
        if self.causal:
            x = x[:, :, : -self.conv_1.padding[0]]
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        if self.causal:
            x = x[:, :, : -self.conv_2.padding[0]]
        return x * x_mask


class Encoder(nn.Module):
    """
    Transformer encoder stack for text conditioning.

    Each block:  LN → MultiHeadAttentionRP → Residual
                 LN → FFN → Residual
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
    ):
        super().__init__()
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_layers.append(MultiHeadAttentionRP(
                hidden_channels, hidden_channels, num_heads,
                window_size=window_size, p_dropout=p_dropout,
            ))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(
                hidden_channels, hidden_channels, filter_channels,
                kernel_size, p_dropout=p_dropout,
            ))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        ):
            y = attn(x, x, attn_mask)
            y = F.dropout(y, p=0.1, training=self.training)
            x = norm1(x + y)
            y = ffn(x, x_mask)
            y = F.dropout(y, p=0.1, training=self.training)
            x = norm2(x + y)
        return x * x_mask


class TextEncoder(nn.Module):
    """
    Phoneme → hidden representation.
    Embedding → Transformer encoder → project to (μ_text, log σ_text).
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = Encoder(
            hidden_channels, filter_channels, num_heads,
            num_layers, kernel_size, p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb(x).transpose(1, 2) * math.sqrt(self.emb.embedding_dim)
        x_mask = _sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(stats.shape[1] // 2, dim=1)
        return x, m, logs, x_mask


# ---------------------------------------------------------------------------
# Duration predictor (deterministic + stochastic)
# ---------------------------------------------------------------------------

class DurationPredictor(nn.Module):
    """Deterministic duration predictor — used for fast inference."""

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        if gin_channels:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x.detach()
        if g is not None:
            g = g.detach()
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class StochasticDurationPredictor(nn.Module):
    """
    Flow-based stochastic duration predictor.

    Models log p(d | c) = log N(f(d); 0, I) + log|det J_f|
    using a sequence of dilated convolutions + coupling layers.
    Trained with negative log-likelihood; at inference samples d ~ p(d|c).
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        num_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        filter_channels = in_channels
        self.log_flow = Log()
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        for _ in range(num_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, num_layers=3))
            self.flows.append(Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        for _ in range(num_flows):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, num_layers=3))
            self.post_flows.append(Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, num_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        x = self.pre(x)
        if g is not None:
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            assert w is not None
            h_w = self.post_pre(w)
            h_w = self.post_flows[0](h_w, x_mask, g=(x, None))
            for flow in self.post_flows[1:]:
                if hasattr(flow, 'g'):
                    h_w = flow(h_w, x_mask, g=(x, None))
                else:
                    h_w, _ = flow(h_w, x_mask)
            # NLL
            e_q = torch.randn(w.size(0), 2, w.size(2), device=x.device) * x_mask
            z_q = e_q
            for flow in self.flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x, h_w))
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + z_q ** 2) * x_mask, dim=[1, 2])
                - logdet_q
            )
            return logq.unsqueeze(-1) * x_mask[:, 0]
        else:
            flows = list(reversed(self.flows))
            z = torch.randn(x.size(0), 2, x.size(2), device=x.device) * noise_scale
            for flow in flows:
                z, _ = flow(z, x_mask, g=(x, None), reverse=True)
            z = z[:, :1]
            return z


class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, dim=[1, 2])
            return y, logdet
        else:
            return torch.exp(x) * x_mask, None


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, dim=[1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x, None


class DDSConv(nn.Module):
    """Dilated and depth-separable convolutions."""

    def __init__(self, channels, kernel_size, num_layers, p_dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(num_layers):
            dil = kernel_size ** i
            pad = (kernel_size * dil - dil) // 2
            self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size,
                                            groups=channels, dilation=dil, padding=pad))
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for c_sep, c_1x1, n1, n2 in zip(
            self.convs_sep, self.convs_1x1, self.norms_1, self.norms_2
        ):
            y = c_sep(x * x_mask)
            y = self.drop(torch.relu(n1(y)))
            y = c_1x1(y * x_mask)
            y = self.drop(torch.relu(n2(y)))
            x = x + y
        return x * x_mask


class ConvFlow(nn.Module):
    """Convolution-based coupling layer for the SDP."""

    def __init__(self, in_channels, filter_channels, kernel_size, num_layers, p_dropout=0.0,
                 num_bins=10):
        super().__init__()
        self.half = in_channels // 2
        self.num_bins = num_bins
        self.convs = DDSConv(self.half, kernel_size, num_layers, p_dropout)
        self.proj = nn.Conv1d(self.half, self.half * (num_bins * 3 - 1), 1)

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = x.split([self.half, self.half], dim=1)
        h = self.convs(x0, x_mask)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.view(b, self.half, -1, t).permute(0, 1, 3, 2)  # (B, half, T, bins)
        unnorm = h[..., : self.num_bins]
        widths = F.softmax(h[..., self.num_bins : 2 * self.num_bins], dim=-1)
        heights = F.softmax(h[..., 2 * self.num_bins : 3 * self.num_bins], dim=-1)
        # Rational-quadratic spline (simplified — use linear here for stability)
        if not reverse:
            x1, logdet = self._forward_spline(x1, widths, heights, unnorm, x_mask)
        else:
            x1, logdet = self._forward_spline(x1, widths, heights, unnorm, x_mask, reverse=True)
        return torch.cat([x0, x1], dim=1), logdet

    def _forward_spline(self, x, w, h, d, mask, reverse=False):
        # Simplified piecewise linear for stability
        cum_w = torch.cumsum(w, dim=-1)
        cum_h = torch.cumsum(h, dim=-1)
        if not reverse:
            # Bin selection
            idx = torch.sum(x.unsqueeze(-1) >= cum_w[..., :-1], dim=-1)
            logdet = torch.sum(torch.log(h.gather(-1, idx.unsqueeze(-1)).squeeze(-1) + 1e-8) * mask[:, :1], dim=[1, 2])
            return x + d[..., 0] * mask[:, :1, :1], logdet
        else:
            return x - d[..., 0] * mask[:, :1, :1], None


# ---------------------------------------------------------------------------
# Monotonic Alignment Search (MAS) — dynamic programming
# ---------------------------------------------------------------------------

def maximum_path(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Monotonic Alignment Search.

    Finds the monotonic path A* in a T_text × T_mel score matrix:
        A* = argmax_{A ∈ A_monotonic} Σ_{i,j} score[i,j] * A[i,j]

    Implemented with Viterbi-style DP.
    value: (B, T_x, T_y)
    mask:  (B, T_x, T_y)
    """
    value = value * mask
    dtype = value.dtype
    device = value.device
    b, x_len, y_len = value.shape
    value = value.cpu().float().numpy()
    mask = mask.cpu().bool().numpy()

    paths = _maximum_path_numpy(value, mask)
    return torch.from_numpy(paths).to(dtype=dtype, device=device)


def _maximum_path_numpy(value, mask):
    import numpy as np
    b, t_x, t_y = value.shape
    paths = np.zeros_like(value)
    for i in range(b):
        v = value[i]
        m = mask[i]
        dp = np.full((t_x, t_y), -1e9)
        dp[0, 0] = v[0, 0]
        for x in range(1, t_x):
            dp[x, 0] = dp[x - 1, 0] + v[x, 0]
        for y in range(1, t_y):
            dp[0, y] = -1e9  # cannot stay on phoneme 0 forever in non-trivial alignment
        for x in range(1, t_x):
            for y in range(1, t_y):
                if not m[x, y]:
                    continue
                dp[x, y] = max(dp[x - 1, y - 1], dp[x - 1, y]) + v[x, y]
        # Backtrack
        path = np.zeros((t_x, t_y))
        y = int(np.sum(m[0]) - 1)
        for x in range(t_x - 1, -1, -1):
            path[x, y] = 1
            if x > 0 and y > 0 and dp[x - 1, y - 1] > dp[x - 1, y]:
                y -= 1
        paths[i] = path
    return paths


# ---------------------------------------------------------------------------
# Main synthesizer (training mode)
# ---------------------------------------------------------------------------

class SynthesizerTrn(nn.Module):
    """
    VITS2 end-to-end TTS model.

    Forward (training):
      1. TextEncoder(phonemes)       → (m_p, logs_p, x_mask)
      2. PosteriorEncoder(mel)       → z, m_q, logs_q
      3. MAS(m_p, m_q)              → alignment A  →  z_p = flows(z, A)
      4. DurationPredictor(m_p)     → log_w
      5. Decoder(z)                 → waveform ŷ
    """

    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        resblock: str = "1",
        resblock_kernel_sizes: tuple = (3, 7, 11),
        resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates: tuple = (8, 8, 2, 2),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: tuple = (16, 16, 4, 4),
        n_speakers: int = 0,
        gin_channels: int = 0,
        use_sdp: bool = True,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp
        self.n_speakers = n_speakers

        self.enc_p = TextEncoder(
            n_vocab, inter_channels, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, p_dropout,
        )

        # Import generator from vocoder module
        from src.vocoder import HiFiGANGenerator
        self.dec = HiFiGANGenerator(
            inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
            upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16,
            gin_channels=gin_channels,
        )

        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels,
        )

        if use_sdp:
            self.dp = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels,
            )
        else:
            self.dp = DurationPredictor(
                hidden_channels, 256, 3, 0.5, gin_channels=gin_channels,
            )

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass."""
        if self.n_speakers > 1 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)
        elif g is not None:
            g = g.unsqueeze(-1) if g.dim() == 2 else g

        # Text encoder
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Posterior encoder
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)

        # Flow: z → z_p (align posterior to prior space)
        z_p = self.flow(z, y_mask, g=g)

        # Monotonic Alignment Search
        with torch.no_grad():
            # (B, T_text, T_mel) score matrix = Σ_d log N(z_p | m_p, logs_p)
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, dim=1, keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, dim=1, keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn = maximum_path(neg_cent.squeeze(1), attn_mask.squeeze(1))

        # Duration target from alignment
        w = attn.sum(dim=2, keepdim=True)  # (B, T_text, 1)

        # Duration predictor loss
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = torch.sum(l_length.float())
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, dim=[1, 2]) / torch.sum(x_mask)

        # Expand text to mel length using alignment
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # Segment z for decoder
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, segment_size=self.dec.segment_size
                                                  if hasattr(self.dec, 'segment_size') else 8192)
        o = self.dec(z_slice, g=g)

        return {
            "o": o,
            "attn": attn,
            "ids_slice": ids_slice,
            "x_mask": x_mask,
            "y_mask": y_mask,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_q": m_q,
            "logs_q": logs_q,
            "l_length": l_length,
        }

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference: text → waveform."""
        if self.n_speakers > 1 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)
        elif g is not None:
            g = g.unsqueeze(-1) if g.dim() == 2 else g

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Duration
        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)

        y_lengths = torch.clamp_min(torch.sum(w_ceil, dim=[1, 2]), 1).long()
        y_mask = _sequence_mask(y_lengths, None).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = _generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec(z * y_mask, g=g)
        return o, attn, y_mask


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sequence_mask(lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    if max_length is None:
        max_length = lengths.max()
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1)


def _generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b, _, t_y, t_x = mask.shape
    cum = torch.cumsum(duration, -1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype, device=mask.device)
    mask_ = mask.squeeze(1).bool()
    for i in range(b):
        path[i] = (cum[i, 0] - duration[i, 0]).unsqueeze(-1) <= torch.arange(t_y, device=mask.device)
        path[i] &= cum[i, 0].unsqueeze(-1) > torch.arange(t_y, device=mask.device)
    return path.unsqueeze(1)


def rand_slice_segments(
    x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, segment_size: int = 8192
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, d, t = x.shape
    if x_lengths is None:
        x_lengths = torch.full((b,), t, device=x.device)
    ids_str = (
        torch.rand(b, device=x.device) * (x_lengths - segment_size).clamp(min=0)
    ).int()
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def slice_segments(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4
) -> torch.Tensor:
    b, d, t = x.shape
    segments = torch.zeros_like(x[:, :, :segment_size])
    for i in range(b):
        idx = ids_str[i]
        segments[i] = x[i, :, idx : idx + segment_size]
    return segments


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence  KL(q ‖ p)  between posterior q and prior p.

        KL = Σ [ logs_p - logs_q - 0.5
               + 0.5 * (exp(2 logs_q) + (z_p - m_p)²) * exp(-2 logs_p) ]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_q) + ((z_p - m_p) ** 2)) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


# ===========================================================================
# Backward-compatibility models (restored).
# These classes were dropped during the VITS refactor (commit 3229a98) but are
# still imported by evaluate.py, export.py and the test-suite. Restored verbatim
# from commit 3229a98~1 so those modules import and the tests collect again.
# ===========================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(context)


class TransformerBlock(nn.Module):
    """Transformer encoder/decoder block."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class SpeakerEncoder(nn.Module):
    """Speaker embedding encoder."""
    
    def __init__(self, speaker_embedding_dim: int, num_speakers: Optional[int] = None):
        super().__init__()
        self.speaker_embedding_dim = speaker_embedding_dim
        self.num_speakers = num_speakers
        
        if num_speakers:
            self.embedding = nn.Embedding(num_speakers, speaker_embedding_dim)
        else:
            # For voice cloning, accept external embeddings
            self.projection = nn.Linear(256, speaker_embedding_dim)
    
    def forward(self, speaker_ids: Optional[torch.Tensor] = None,
                speaker_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        if speaker_ids is not None and self.num_speakers:
            return self.embedding(speaker_ids)
        elif speaker_embeddings is not None:
            return self.projection(speaker_embeddings)
        else:
            raise ValueError("Either speaker_ids or speaker_embeddings must be provided")


class AdvancedTTSModel(nn.Module):
    """
    Advanced TTS model with transformer architecture and speaker conditioning.
    Supports both multi-speaker TTS and voice cloning.
    """
    
    def __init__(self, 
                 vocab_size: int = 256,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_length: int = 1000,
                 mel_channels: int = 80,
                 speaker_embedding_dim: int = 256,
                 num_speakers: Optional[int] = None,
                 dropout: float = 0.1,
                 use_speaker_embedding: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.mel_channels = mel_channels
        self.use_speaker_embedding = use_speaker_embedding
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Speaker encoder
        if use_speaker_embedding:
            self.speaker_encoder = SpeakerEncoder(speaker_embedding_dim, num_speakers)
            self.speaker_projection = nn.Linear(speaker_embedding_dim, d_model)
        
        # Transformer encoder (text processing)
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Transformer decoder (mel generation)
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projections
        self.mel_projection = nn.Linear(d_model, mel_channels)
        self.duration_projection = nn.Linear(d_model, 1)
        self.stop_projection = nn.Linear(d_model, 1)
        
        # Prenet for decoder
        self.prenet = nn.Sequential(
            nn.Linear(mel_channels, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def encode_text(self, text_tokens: torch.Tensor, 
                   speaker_embeddings: Optional[torch.Tensor] = None,
                   speaker_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text tokens with optional speaker conditioning."""
        # Text embedding
        x = self.text_embedding(text_tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Add speaker conditioning
        if self.use_speaker_embedding:
            speaker_emb = self.speaker_encoder(speaker_ids, speaker_embeddings)
            speaker_emb = self.speaker_projection(speaker_emb)
            x = x + speaker_emb.unsqueeze(1)
        
        x = self.dropout(x)
        
        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x
    
    def decode_mel(self, encoder_output: torch.Tensor, 
                   prev_mel: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode mel spectrogram from encoded text."""
        if prev_mel is None:
            # Initialize with zeros for autoregressive generation
            prev_mel = torch.zeros(encoder_output.size(0), 1, self.mel_channels, 
                                 device=encoder_output.device)
        
        # Prenet processing
        decoder_input = self.prenet(prev_mel)
        decoder_input = self.pos_encoding(decoder_input)
        
        # Decoder layers
        for layer in self.decoder_layers:
            decoder_input = layer(decoder_input)
        
        # Output projections
        mel_output = self.mel_projection(decoder_input)
        duration_output = F.softplus(self.duration_projection(encoder_output))
        stop_output = torch.sigmoid(self.stop_projection(decoder_input))
        
        return mel_output, duration_output, stop_output
    
    def forward(self, text_tokens: torch.Tensor,
                mel_targets: Optional[torch.Tensor] = None,
                speaker_embeddings: Optional[torch.Tensor] = None,
                speaker_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for training or inference."""
        # Encode text
        encoder_output = self.encode_text(text_tokens, speaker_embeddings, speaker_ids)
        
        # Decode mel
        if mel_targets is not None:
            # Teacher forcing during training
            mel_input = F.pad(mel_targets, (0, 0, 1, 0))[:, :-1]  # Shift right
            mel_output, duration_output, stop_output = self.decode_mel(encoder_output, mel_input)
        else:
            # Autoregressive generation during inference
            mel_output, duration_output, stop_output = self.decode_mel(encoder_output)
        
        return {
            'mel_output': mel_output,
            'duration_output': duration_output,
            'stop_output': stop_output,
            'encoder_output': encoder_output
        }
    
    def inference(self, text_tokens: torch.Tensor,
                  speaker_embeddings: Optional[torch.Tensor] = None,
                  speaker_ids: Optional[torch.Tensor] = None,
                  max_decoder_steps: int = 1000) -> torch.Tensor:
        """Generate mel spectrogram autoregressively."""
        self.eval()
        
        with torch.no_grad():
            # Encode text
            encoder_output = self.encode_text(text_tokens, speaker_embeddings, speaker_ids)
            
            # Initialize
            mel_outputs = []
            prev_mel = torch.zeros(text_tokens.size(0), 1, self.mel_channels, 
                                 device=text_tokens.device)
            
            for step in range(max_decoder_steps):
                mel_output, _, stop_output = self.decode_mel(encoder_output, prev_mel)
                mel_outputs.append(mel_output)
                prev_mel = mel_output
                
                # Check stop condition
                if stop_output.squeeze() > 0.5:
                    break
            
            return torch.cat(mel_outputs, dim=1)


# Legacy SimpleTTSModel for backward compatibility
class SimpleTTSModel(nn.Module):
    """Simple TTS model for backward compatibility."""
    
    def __init__(self, input_dim=256, hidden_size=256, output_dim=80, num_layers=4, speaker_embedding=True):
        super().__init__()
        self.speaker_embedding = speaker_embedding
        self.embedding = nn.Embedding(256, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x, speaker_emb=None):
        x = self.embedding(x)
        if self.speaker_embedding and speaker_emb is not None:
            x = x + speaker_emb.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out


def create_model(model_config: Dict[str, Any], model_type: Optional[str] = None) -> nn.Module:
    """Factory function to create TTS models based on configuration.

    ``model_type`` may be passed explicitly (used by :class:`ModelFactory`);
    otherwise it is read from ``model_config['type']`` (default ``'advanced'``).
    """
    model_type = (model_type or model_config.get('type', 'advanced')).lower()

    if model_type == 'advanced':
        return AdvancedTTSModel(**model_config.get('params', {}))
    elif model_type == 'simple':
        return SimpleTTSModel(**model_config.get('params', {}))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model(checkpoint_path: str, device: str = 'cpu') -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('model_config', {})
    model = create_model(model_config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model


def save_model(model: nn.Module, path: str, model_config: Dict[str, Any], 
               optimizer_state: Optional[Dict] = None, epoch: Optional[int] = None):
    """Save model checkpoint."""
    checkpoint = {
        'state_dict': model.state_dict(),
        'model_config': model_config,
        'epoch': epoch,
        'optimizer_state': optimizer_state
    }
    torch.save(checkpoint, path)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ModelFactory:
    """Factory class for creating different TTS models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model factory with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
    
    def create_model(self, model_type: str = None) -> nn.Module:
        """
        Create a model based on configuration.
        
        Args:
            model_type: Override model type from config
            
        Returns:
            Initialized model
        """
        if model_type is None:
            model_type = self.config.get('model_type', 'tacotron2')
        
        return create_model(self.config, model_type)
    
    def load_model(self, checkpoint_path: str, device: str = 'cpu') -> nn.Module:
        """Load model from checkpoint."""
        return load_model(checkpoint_path, device)
    
    def save_model(self, model: nn.Module, path: str, 
                   optimizer_state: Optional[Dict] = None, 
                   epoch: Optional[int] = None):
        """Save model checkpoint."""
        return save_model(model, path, self.config, optimizer_state, epoch)
