# VITS2 Neural TTS + Voice Cloning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

State-of-the-art end-to-end text-to-speech system implementing **VITS2** with
**ECAPA-TDNN** zero-shot voice cloning. Synthesises natural 22 kHz speech from
raw text in a single forward pass — no autoregressive decoding, no separate
alignment model.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Mathematical Foundations](#2-mathematical-foundations)
   - 2.1 [Variational Autoencoder — ELBO](#21-variational-autoencoder--elbo)
   - 2.2 [Normalising Flows — Prior p(z | c)](#22-normalising-flows--prior-pz--c)
   - 2.3 [WaveNet Posterior Encoder](#23-wavenet-posterior-encoder)
   - 2.4 [Transformer Text Encoder](#24-transformer-text-encoder)
   - 2.5 [Stochastic Duration Predictor](#25-stochastic-duration-predictor)
   - 2.6 [Monotonic Alignment Search](#26-monotonic-alignment-search)
   - 2.7 [HiFi-GAN Vocoder](#27-hifi-gan-vocoder)
   - 2.8 [Adversarial Training Losses](#28-adversarial-training-losses)
   - 2.9 [ECAPA-TDNN Speaker Encoder](#29-ecapa-tdnn-speaker-encoder)
   - 2.10 [GE2E Speaker Verification Loss](#210-ge2e-speaker-verification-loss)
3. [Installation](#3-installation)
4. [Training](#4-training)
5. [Inference](#5-inference)
6. [Voice Cloning](#6-voice-cloning)
7. [Datasets](#7-datasets)
8. [Project Structure](#8-project-structure)
9. [References](#9-references)

---

## 1. Architecture Overview

```
                         ┌──────────────────────────────────────────────┐
  Text (phonemes)        │           VITS2 Generator                    │
  x ∈ ℤᵀˣ  ──────────►  │  TextEncoder                                 │
                         │    (Transformer + Relative PE)               │
                         │       ↓ m_p, logs_p                          │
                         │  StochasticDurationPredictor                  │
                         │    (Flow-based)  ↓ durations d               │
                         │  Monotonic Alignment Search                   │
  Mel / Linear spec ──►  │  PosteriorEncoder (WaveNet)                  │  Training only
                         │    ↓ z ~ N(m_q, σ_q²)                        │
                         │  NormalisingFlows (z → z_p, reverse at infer)│
                         │  HiFi-GAN Decoder  ↓                         │
                         └──────────────────────────────────────────────┘
                                               │ waveform ŷ
  Reference audio ──► ECAPA-TDNN ──► g ───────┘ (speaker conditioning)
  (voice cloning)       (d-vector)
```

**Key design choices vs. competing systems:**

| System | Alignment | Duration | Vocoder | Voice Cloning |
|---|---|---|---|---|
| Tacotron 2 | Attention | Autoregressive | WaveNet | ✗ |
| FastSpeech 2 | External aligner | Explicit predictor | HiFi-GAN | Limited |
| VITS | MAS + flow | SDP (flow) | HiFi-GAN (integrated) | Speaker embedding |
| **VITS2 (this)** | **MAS + improved flow** | **SDP v2** | **HiFi-GAN V1** | **ECAPA-TDNN d-vector** |

---

## 2. Mathematical Foundations

### 2.1 Variational Autoencoder — ELBO

The model is a **Conditional VAE** over waveforms `x` given text condition `c`:

```
log p_θ(x | c) ≥ E_{q_φ(z|x,c)}[ log p_θ(x|z) ]  −  KL[ q_φ(z|x,c) ‖ p_θ(z|c) ]
```

This is the **Evidence Lower BOund (ELBO)**, tight when the posterior
`q_φ(z|x,c)` equals the true posterior `p(z|x,c)`.

**Three components:**

1. **Reconstruction term** `E_q[log p_θ(x|z)]`  
   Measures how well the decoder can reconstruct the waveform from latent `z`.
   In practice computed as:  
   `L_rec = ‖mel(x) − mel(G(z))‖₁`  
   because the mel-spectrogram is a perceptually motivated representation.

2. **KL divergence** `KL[q_φ ‖ p_θ]`  
   For diagonal Gaussians with parameters `(m_q, σ_q)` and `(m_p, σ_p)`:
   ```
   KL = Σᵢ [ log σ_p,i/σ_q,i  +  (σ_q,i² + (m_q,i − m_p,i)²) / (2σ_p,i²)  −  ½ ]
      = Σᵢ [ log_σ_p − log_σ_q − ½
             + ½(exp(2 log_σ_q) + (z_p − m_p)²) exp(−2 log_σ_p) ]
   ```
   where `z_p = f(z)` is the posterior sample transformed through the flow to
   the prior space.

3. **Duration NLL** `−log p(d | c)`  
   Stochastic Duration Predictor (see §2.5).

**Total objective:**
```
L = 45 · L_mel  +  1 · L_kl  +  1 · L_dur  +  2 · L_fm  +  1 · L_adv
```
Weights follow the original VITS paper; `L_mel × 45` dominates early training
to ensure the output is on-target before GAN pressure kicks in.

---

### 2.2 Normalising Flows — Prior p(z | c)

We choose `p_θ(z | c)` to be a **flow-based distribution** rather than a
fixed `N(0, I)`. This gives the prior the capacity to match the complex
structure of the latent space.

A normalising flow is an invertible mapping `f: z ↦ z_p` such that:

```
log p_θ(z | c) = log π₀(z_p)  +  log |det ∂z_p/∂z|
```

where `π₀ = N(0, I)`. The **change-of-variables formula** gives the exact
log-likelihood.

**Residual Coupling Layer (affine coupling):**

Split `z = [z₁, z₂]` along the channel axis:

```
z₁' = z₁                          (pass-through)
z₂' = z₂ · exp(s(z₁; θ)) + t(z₁; θ)    (scale + shift)
```

Log-determinant:  `log |det J| = Σᵢ s_i(z₁)`  — only over the transformed half.

Inversion (needed at inference to go from `z_p ~ N(0,I)` back to `z`):
```
z₁ = z₁'
z₂ = (z₂' − t(z₁; θ)) · exp(−s(z₁; θ))
```

`s` and `t` are implemented as WaveNet stacks conditioned on text `c`.
**Flip layers** alternate which half is transformed, ensuring full coupling.

We stack **4 coupling + 4 flip = 8 invertible layers** forming the
`ResidualCouplingBlock`.

---

### 2.3 WaveNet Posterior Encoder

The posterior `q_φ(z | x_mel, g)` is parameterised by a **WaveNet** stack
that processes the (linear) spectrogram `x_mel`.

Each WaveNet layer computes:

```
h_l = tanh(W_f,l * x + V_f,l * g)  ⊙  σ(W_g,l * x + V_g,l * g)
x'  = W_o * h_l  +  x              (residual connection)
```

where `*` denotes dilated causal convolution with dilation `d = r^l` (r=1
here since the spectrogram has no causal structure), `g` is the speaker
embedding, and `⊙` is element-wise product.

The final hidden state is projected:
```
[μ, log σ] = Conv1d(WaveNet(x_mel))
z ~ N(μ, σ²)          (reparameterisation trick)
```

Stack: **16 WaveNet layers**, kernel size 5, hidden dim 192.

---

### 2.4 Transformer Text Encoder

The text encoder maps phoneme IDs to a continuous representation with
**relative positional encoding** (Shaw et al., 2018).

**Self-attention with relative position bias:**

```
score_ij = (q_i · k_j + q_i · r_{i−j}) / √d_k
```

where `r_{i−j}` is a learnable embedding for relative position `i−j`, clipped
to `[−w, +w]` (window size `w = 4`). This allows the model to use local
relative structure while remaining permutation-equivariant at long range.

**Multi-head attention:**
```
head_h = Attention(QW_h^Q, KW_h^K, VW_h^V)
output  = concat(head_1, …, head_H) W^O
```

**Position-wise FFN** (with depthwise conv instead of dense for efficiency):
```
FFN(x) = Conv(ReLU(Conv(x, k=3)), k=3)
```

**Encoder output** → `Conv1d` → `(m_p, log_σ_p)` (prior mean and variance).

Stack: **6 transformer layers**, 2 heads, hidden 192, filter 768.

---

### 2.5 Stochastic Duration Predictor

A deterministic predictor (MSE on log-durations) produces robotic,
monotone speech. VITS introduces a **Stochastic Duration Predictor (SDP)**
based on normalising flows:

```
log p(d | c) = log N(f_θ(d; c); 0, I)  +  log |det J_f|
```

`f_θ` is a stack of `ConvFlow` coupling layers conditioning on the text `c`.

**Advantages of flow-based duration:**
- Models the full distribution `p(d | c)`, not just the mean
- At inference, sampling `ε ~ N(0, noise_scale²)` and inverting `f_θ` gives
  varied, natural-sounding durations
- Training NLL: `L_dur = −log p(d | c)` is computed efficiently via the
  change-of-variables formula

At inference we also have a **length scale** parameter `λ`:
```
d̂ = λ · f_θ⁻¹(ε;  c)        ε ~ N(0, noise_scale_w²)
```
Setting `λ > 1` slows speech; `λ < 1` speeds it up.

---

### 2.6 Monotonic Alignment Search

MAS finds the optimal **monotonic alignment** between `T_text` phonemes and
`T_mel` spectrogram frames during training. This avoids the need for external
forced-alignment tools (like MFA).

**Score matrix:** For posterior sample `z_p = flow(z)` and prior `(m_p, σ_p)`:

```
Q_{ij} = log N(z_p,j | m_p,i, σ_p,i²)
       = −½ log(2π) − log σ_p,i
         − ½ (z_p,j − m_p,i)² / σ_p,i²
```

**DP recursion** (Viterbi):
```
dp[i, j] = Q[i, j]  +  max(dp[i−1, j−1], dp[i−1, j])
```

The constraint is that the path must be monotonic (`i` can only stay or
advance) with each phoneme taking ≥ 1 frame (in practice the SDP handles
this). Complexity: `O(T_text × T_mel)`.

After MAS the alignment matrix `A` expands the prior parameters:
```
m_p_expanded  = A^T · m_p       (matrix multiply broadcasts phoneme → frame)
σ_p_expanded  = A^T · σ_p
```

This is equivalent to **replicating** each phoneme embedding for its aligned
duration.

---

### 2.7 HiFi-GAN Vocoder

HiFi-GAN synthesises waveforms from latent `z` via **transposed
convolutions** (upsampling) and **Multi-Receptive Field Fusion (MRF)**.

**Generator architecture:**

```
z  →  Conv(7)  →  [ConvTranspose + MRF] × L  →  Conv(7)  →  Tanh  →  ŷ
```

At each of the `L = 4` upsampling stages, `ConvTranspose1d` with stride
`u_l` increases temporal resolution. The **MRF** at each stage averages `K`
ResBlocks with different kernel sizes and dilations:

```
MRF(x) = (1/K) Σ_{k=1}^K ResBlock_k(x)
```

Each ResBlock uses 3 dilation levels. For ResBlock1 with kernel `k` and
dilations `(d₁, d₂, d₃)`:
```
ResBlock(x) = x + Σ_r LeakyReLU ∘ Conv_k^{d_r} ∘ LeakyReLU ∘ Conv_k^{d_r}
```

Total upsampling: `Π u_l = 8 × 8 × 2 × 2 = 256` samples/frame.  
At 22 050 Hz and hop 256: frame rate ≈ 86 Hz.

**Discriminators** use **weight normalisation** (not spectral norm) in
the generator path and evaluate the waveform at multiple scales/periods.

---

### 2.8 Adversarial Training Losses

**Least-Squares GAN (LS-GAN)** objective is more stable than the original
GAN cross-entropy because the gradient does not vanish when the discriminator
is confident:

**Discriminator** (minimise w.r.t. D):
```
L_D = E[(D(y) − 1)²]  +  E[D(ŷ)²]
```

**Generator** (minimise w.r.t. G):
```
L_adv(G) = E[(D(G(z)) − 1)²]
```

**Feature Matching Loss** (perceptual loss over discriminator internals):
```
L_fm = (1/KL) Σ_{k=1}^K Σ_{l=1}^L ‖D_k^l(y) − D_k^l(ŷ)‖₁
```

`D_k^l` is the `l`-th intermediate activation of the `k`-th sub-discriminator.
This loss is **not adversarial** — it purely encourages G to produce
activations similar to real audio, acting as a perceptual similarity metric.

**Multi-Period Discriminator (MPD):** 5 sub-discriminators, each reshaping the
waveform into a `(T/p, p)` 2-D tensor and applying 2-D convolutions. Periods
`p ∈ {2, 3, 5, 7, 11}` are co-prime to maximise coverage.

**Multi-Scale Discriminator (MSD):** 3 sub-discriminators at resolutions
`1×, 2×, 4×` (average pooling with stride 2). The first uses spectral norm
for training stability.

---

### 2.9 ECAPA-TDNN Speaker Encoder

ECAPA-TDNN achieves EER < 0.8% on VoxCeleb1, making it state-of-the-art for
text-independent speaker verification.

**SE-Res2Block** combines three innovations:

1. **Res2Net** multi-scale feature extraction:
   Split input into `s` equal groups `{x_i}`, then:
   ```
   y_1 = x_1
   y_i = Conv_i(x_i + y_{i-1}),   i = 2, …, s
   output = concat(y_1, …, y_s)
   ```
   Scale `s = 8` gives `s` different receptive fields per block, exponentially
   increasing the effective temporal context.

2. **Squeeze-and-Excitation (SE)** channel-wise recalibration:
   ```
   s = σ(W₂ · ReLU(W₁ · GlobalAvgPool(x)))    ∈ (0,1)^C
   y = x ⊙ s
   ```
   Bottleneck `W₁ ∈ ℝ^{C/r × C}`, `W₂ ∈ ℝ^{C × C/r}` with ratio `r = 8`.

3. **Multi-layer feature aggregation:** Outputs of all three SE-Res2Blocks are
   concatenated before temporal pooling, giving the network access to all
   levels of feature abstraction simultaneously.

**Attentive Statistics Pooling:**
```
e_t = w^T · tanh(W · h_t + b)        (attention energy, scalar per frame)
α_t = exp(e_t) / Σ_τ exp(e_τ)        (softmax over time)
μ   = Σ_t α_t · h_t                  (weighted mean)
σ   = √(Σ_t α_t · h_t²  −  μ²)      (weighted std)
output = concat(μ, σ)                 ∈ ℝ^{2C}
```

The **mean** captures average spectral/prosodic properties; the **std**
captures temporal variability — together they encode more speaker identity
than mean alone.

**Final embedding:**
```
e = BN(FC(BN(concat(μ, σ))))         ∈ ℝ^{192}
ê = e / ‖e‖₂                         (L2 normalisation)
```

L2 normalisation maps all embeddings to the unit hypersphere, making cosine
similarity equal to dot product — simplifying the loss computation.

**Voice cloning** at inference:
```
g = ECAPA-TDNN(reference_mel)    ∈ ℝ^{192}
ŷ = Decoder(z, g)                where g conditions WN layers via additive bias
```

---

### 2.10 GE2E Speaker Verification Loss

The **Generalised End-to-End (GE2E) Loss** trains the speaker encoder to
produce embeddings that cluster by identity.

Given a batch of `N` speakers × `M` utterances each, embed all `N×M`
utterances and compute speaker centroids:

```
c_j = normalize( (1/M) Σ_{m=1}^M e_{jm} )
```

**Similarity matrix:**
```
S_{ji,k} = w · cos(e_{ji}, c_k) + b        (w > 0 learnable scale)
```

**Softmax GE2E loss** (preferred over contrast version):
```
L(e_{ji}) = −S_{ji,j} + log Σ_{k=1}^N exp(S_{ji,k})
```

This is the **cross-entropy loss** where class `j` is the correct speaker.
Training pushes `e_{ji}` towards its own centroid `c_j` and away from all
other centroids `c_{k≠j}`.

---

## 3. Installation

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# 2. Install PyTorch (CUDA 11.8 example)
pip install torch torchudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install remaining deps
pip install -r requirements.txt

# 4. Install espeak-ng (required for phonemizer)
#   Ubuntu:    sudo apt-get install espeak-ng
#   macOS:     brew install espeak-ng
#   Windows:   download installer from espeak-ng/releases on GitHub

# 5. Install the package (editable)
pip install -e .
```

---

## 4. Training

### Dataset preparation

The system auto-detects **LJSpeech**, **VCTK**, and **LibriTTS** formats.
For LJSpeech (recommended for single-speaker):

```bash
# Download LJSpeech (~2.6 GB)
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xf LJSpeech-1.1.tar.bz2
mv LJSpeech-1.1 data/train

# Optional: create a small validation split
# The dataset reader handles it automatically via metadata.csv
```

For VCTK (multi-speaker, 44 hours, 109 speakers):
```bash
# Place speaker dirs under data/train/ and data/val/
# e.g. data/train/p225/p225_001.wav + p225_001.txt
```

For LibriTTS (large-scale, 245 hours):
```bash
# Download train-clean-100 or train-clean-360
wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
tar -xf train-clean-100.tar.gz --strip-components=1 -C data/train
```

### Start training

```bash
python main.py train \
    --config config.yaml \
    --output-dir ./outputs \
    --device auto
```

Resume from checkpoint:
```bash
python main.py train --config config.yaml --resume outputs/checkpoint_epoch_50.pth
```

Monitor training:
```bash
tensorboard --logdir logs/
```

---

## 5. Inference

```bash
python main.py infer \
    --model outputs/checkpoint_best.pth \
    --text "The quick brown fox jumps over the lazy dog." \
    --output out.wav \
    --noise-scale 0.667 \
    --length-scale 1.0
```

Python API:
```python
from src.infer import TTSInferencer

tts = TTSInferencer.from_checkpoint("outputs/checkpoint_best.pth", device="cuda")
wav = tts.synthesize("Hello, world!")
import soundfile as sf
sf.write("hello.wav", wav, 22050)
```

---

## 6. Voice Cloning

Provide at least **3 seconds** of clean reference audio (30 s recommended).

```bash
python main.py clone \
    --model outputs/checkpoint_best.pth \
    --reference /path/to/target_speaker.wav \
    --text "This is synthesized in the cloned voice." \
    --output cloned.wav \
    --noise-scale 0.667 \
    --similarity-threshold 0.7
```

Python API:
```python
from src.clone import VoiceCloner

cloner = VoiceCloner.from_checkpoint("outputs/checkpoint_best.pth", device="cuda")
wav = cloner.clone(
    text="Hello from a cloned voice!",
    reference_path="reference.wav",
    noise_scale=0.667,
)
import soundfile as sf
sf.write("cloned.wav", wav, 22050)
```

**Improving cloning quality:**
- Use ≥ 10 s of reference audio for stable d-vector estimation
- Reference audio should be clean (no background noise, music)
- Match the language of training data; cross-lingual cloning degrades quality
- Lower `noise_scale` (e.g. 0.3) for more consistent cloning

---

## 7. Datasets

| Dataset | Speakers | Hours | Sample Rate | License |
|---|---|---|---|---|
| **LJSpeech** | 1 | 24 h | 22 kHz | Public domain |
| **VCTK** | 109 | 44 h | 48 kHz | CC-BY 4.0 |
| **LibriTTS** | 2456 | 245 h | 24 kHz | CC-BY 4.0 |
| **HiFi-TTS** | 10 | 292 h | 44 kHz | CC-BY 4.0 |
| **Common Voice** | 17 000+ | 7 000+ h | 16 kHz | CC-0 / CC-BY |

**Recommended training recipe:**
1. Pre-train on LJSpeech (single speaker) until stable (≈100 epochs)
2. Fine-tune on VCTK or LibriTTS for multi-speaker capability
3. Use ECAPA-TDNN trained on VoxCeleb2 for the speaker encoder

---

## 8. Project Structure

```
Speech_synthesis/
├── src/
│   ├── model.py            # VITS2: TextEncoder, WN, ResidualCouplingBlock,
│   │                       #        StochasticDurationPredictor, SynthesizerTrn
│   ├── speaker_encoder.py  # ECAPA-TDNN, SE-Res2Block, AttentiveStatsPool, GE2ELoss
│   ├── vocoder.py          # HiFi-GAN Generator, MPD, MSD, loss functions
│   ├── train.py            # VITS2Trainer: G/D alternating updates, AMP
│   ├── clone.py            # VoiceCloner: reference embed → conditioned synthesis
│   ├── infer.py            # TTSInferencer: text → phonemes → waveform
│   ├── data.py             # TTSDataset: LJSpeech/VCTK/LibriTTS + collate_fn
│   ├── text_processor.py   # G2P, phoneme normalisation, symbol tables
│   ├── evaluate.py         # PESQ, STOI, Mel-CD, RTF benchmarks
│   ├── preprocess.py       # Dataset format detection and preprocessing
│   ├── export.py           # TorchScript / ONNX export
│   ├── gradio_interface.py # Web demo
│   └── utils.py            # EarlyStopping, GradientClipper, etc.
├── tests/                  # Pytest test suite
├── notebooks/              # Jupyter analysis notebooks
├── config.yaml             # Full VITS2 hyperparameter config
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Build metadata
└── main.py                 # CLI entry point
```

---

## 9. References

1. **VITS** — Kim et al. "Conditional Variational Autoencoder with Adversarial
   Learning for End-to-End Text-to-Speech." ICML 2021.
   [arxiv:2106.06103](https://arxiv.org/abs/2106.06103)

2. **VITS2** — Kim et al. "VITS2: Improving Quality and Efficiency of
   Single-Stage Text-to-Speech with Adversarial Learning and Architecture
   Design." Interspeech 2023.
   [arxiv:2307.16430](https://arxiv.org/abs/2307.16430)

3. **HiFi-GAN** — Kong et al. "HiFi-GAN: Generative Adversarial Networks for
   Efficient and High Fidelity Speech Synthesis." NeurIPS 2020.
   [arxiv:2010.05646](https://arxiv.org/abs/2010.05646)

4. **ECAPA-TDNN** — Desplanques et al. "ECAPA-TDNN: Emphasized Channel
   Attention, Propagation and Aggregation in TDNN Based Speaker Verification."
   Interspeech 2020.
   [arxiv:2005.07143](https://arxiv.org/abs/2005.07143)

5. **GE2E Loss** — Wan et al. "Generalized End-to-End Loss for Speaker
   Verification." ICASSP 2018.
   [arxiv:1710.10467](https://arxiv.org/abs/1710.10467)

6. **Res2Net** — Gao et al. "Res2Net: A New Multi-scale Backbone Architecture."
   IEEE TPAMI 2019.
   [arxiv:1904.01169](https://arxiv.org/abs/1904.01169)

7. **Relative Position Representations** — Shaw et al. "Self-Attention with
   Relative Position Representations." NAACL 2018.
   [arxiv:1803.02155](https://arxiv.org/abs/1803.02155)

8. **WaveNet** — van den Oord et al. "WaveNet: A Generative Model for Raw
   Audio." 2016. [arxiv:1609.03499](https://arxiv.org/abs/1609.03499)

9. **LS-GAN** — Mao et al. "Least Squares Generative Adversarial Networks."
   ICCV 2017. [arxiv:1611.04076](https://arxiv.org/abs/1611.04076)

10. **Normalising Flows** — Rezende & Mohamed. "Variational Inference with
    Normalizing Flows." ICML 2015.
    [arxiv:1505.05770](https://arxiv.org/abs/1505.05770)
