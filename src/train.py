"""
VITS2 Training Loop with GAN + CVAE objectives.

Total generator loss:
  L_G = λ_mel · L_mel + λ_kl · L_kl + λ_dur · L_dur
      + λ_adv · L_adv(G) + λ_fm · L_fm

Total discriminator loss:
  L_D = L_adv(D_MPD) + L_adv(D_MSD)

Optimisers:
  - AdamW with β=(0.8, 0.99) and weight-decay 0.01
  - Exponential LR decay γ=0.999⁸ per epoch (following VITS paper)
"""
import math
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # type: ignore
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

from src.model import SynthesizerTrn, kl_loss, slice_segments
from src.vocoder import (
    HiFiGANGenerator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    generator_loss,
    feature_loss,
    mel_spectrogram_loss,
)
from src.data import TTSDataset, collate_fn
from src.text_processor import TextProcessor
from src.logging_config import get_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_optimisers(net_g, net_d_mpd, net_d_msd, lr=2e-4):
    optim_g = torch.optim.AdamW(
        net_g.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=0.01
    )
    optim_d = torch.optim.AdamW(
        list(net_d_mpd.parameters()) + list(net_d_msd.parameters()),
        lr=lr, betas=(0.8, 0.99), weight_decay=0.01,
    )
    return optim_g, optim_d


def build_schedulers(optim_g, optim_d, gamma=0.999**8):
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=gamma)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=gamma)
    return sched_g, sched_d


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class VITS2Trainer:
    """
    End-to-end VITS2 trainer.

    Manages:
      - Generator (SynthesizerTrn) and discriminator networks
      - Alternating G / D updates each step
      - Mixed-precision training
      - Checkpoint save / resume
      - TensorBoard logging
    """

    def __init__(self, config: DictConfig):
        self.cfg = config
        self.device = self._setup_device()
        self.logger = get_logger(dict(config.get("system", {})))

        self._build_models()
        self._build_data()
        self._build_optim()
        self._setup_monitoring()

        self.epoch = 0
        self.step = 0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    def _setup_device(self) -> torch.device:
        d = self.cfg.system.get("device", "auto")
        if d == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(d)

    def _build_models(self):
        c = self.cfg.model
        hp = c.get("params", {})

        self.net_g = SynthesizerTrn(
            n_vocab=hp.get("n_vocab", 512),
            spec_channels=hp.get("spec_channels", 513),
            segment_size=hp.get("segment_size", 8192),
            inter_channels=hp.get("inter_channels", 192),
            hidden_channels=hp.get("hidden_channels", 192),
            filter_channels=hp.get("filter_channels", 768),
            n_heads=hp.get("n_heads", 2),
            n_layers=hp.get("n_layers", 6),
            kernel_size=hp.get("kernel_size", 3),
            p_dropout=hp.get("p_dropout", 0.1),
            resblock=hp.get("resblock", "1"),
            resblock_kernel_sizes=tuple(hp.get("resblock_kernel_sizes", [3, 7, 11])),
            resblock_dilation_sizes=tuple(
                tuple(d) for d in hp.get("resblock_dilation_sizes", [[1,3,5],[1,3,5],[1,3,5]])
            ),
            upsample_rates=tuple(hp.get("upsample_rates", [8, 8, 2, 2])),
            upsample_initial_channel=hp.get("upsample_initial_channel", 512),
            upsample_kernel_sizes=tuple(hp.get("upsample_kernel_sizes", [16, 16, 4, 4])),
            n_speakers=hp.get("n_speakers", 0),
            gin_channels=hp.get("gin_channels", 256),
            use_sdp=hp.get("use_sdp", True),
        ).to(self.device)

        self.net_d_mpd = MultiPeriodDiscriminator().to(self.device)
        self.net_d_msd = MultiScaleDiscriminator().to(self.device)

        n_params_g = sum(p.numel() for p in self.net_g.parameters() if p.requires_grad)
        self.logger.logger.info(f"Generator parameters: {n_params_g / 1e6:.1f}M")

    def _build_data(self):
        tp = TextProcessor(
            language=self.cfg.text_processing.get("language", "en"),
            phoneme_backend=self.cfg.text_processing.get("phoneme_backend", "espeak"),
        )
        train_ds = TTSDataset(
            self.cfg.data.train_path,
            sample_rate=self.cfg.data.sample_rate,
            text_processor=tp,
            config=self.cfg.data,
            subset="train",
        )
        val_ds = TTSDataset(
            self.cfg.data.val_path,
            sample_rate=self.cfg.data.sample_rate,
            text_processor=tp,
            config=self.cfg.data,
            subset="val",
        )
        workers = self.cfg.system.get("max_workers", 4)
        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_ds, batch_size=self.cfg.training.batch_size,
            shuffle=True, num_workers=workers, pin_memory=pin,
            collate_fn=collate_fn, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.cfg.training.batch_size,
            shuffle=False, num_workers=workers, pin_memory=pin,
            collate_fn=collate_fn,
        )
        self.logger.logger.info(
            f"Train: {len(train_ds)} | Val: {len(val_ds)} samples"
        )

    def _build_optim(self):
        lr = self.cfg.training.get("learning_rate", 2e-4)
        self.optim_g, self.optim_d = build_optimisers(
            self.net_g, self.net_d_mpd, self.net_d_msd, lr=lr
        )
        self.sched_g, self.sched_d = build_schedulers(self.optim_g, self.optim_d)
        self.use_amp = self.cfg.performance.get("enable_fp16", False) and self.device.type == "cuda"
        self.scaler_g = GradScaler(enabled=self.use_amp)
        self.scaler_d = GradScaler(enabled=self.use_amp)

    def _setup_monitoring(self):
        out = Path(self.cfg.training.get("output_dir", "./outputs"))
        out.mkdir(parents=True, exist_ok=True)
        self.output_dir = out
        log_dir = self.cfg.monitoring.get("tensorboard_log_dir", "./logs")
        tb_enabled = self.cfg.monitoring.get("enable_tensorboard", True) and SummaryWriter is not None
        self.writer = SummaryWriter(log_dir) if tb_enabled else None

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: Dict) -> Dict[str, float]:
        x = batch["text_tokens"].to(self.device)
        x_lengths = batch["text_lengths"].to(self.device)
        spec = batch["spectrogram"].to(self.device)
        spec_lengths = batch["spec_lengths"].to(self.device)
        wav = batch["wav"].to(self.device)
        sid = batch.get("speaker_id")
        if sid is not None:
            sid = sid.to(self.device)
        g = batch.get("speaker_embedding")
        if g is not None:
            g = g.to(self.device)

        wav = wav.unsqueeze(1)  # (B, 1, T)

        # ---- Discriminator update ----
        with autocast(enabled=self.use_amp):
            out = self.net_g(x, x_lengths, spec, spec_lengths, sid=sid, g=g)
            y_hat = out["o"]  # (B, 1, T_seg)

            # Slice real waveform to same segment
            y_mel = slice_segments(wav, out["ids_slice"] * self.cfg.data.get("hop_length", 256),
                                   segment_size=y_hat.shape[-1])

            y_d_rs, y_d_gs, _, _ = self.net_d_mpd(y_mel, y_hat.detach())
            y_d_rs2, y_d_gs2, _, _ = self.net_d_msd(y_mel, y_hat.detach())

            loss_d, _, _ = discriminator_loss(y_d_rs + y_d_rs2, y_d_gs + y_d_gs2)

        self.optim_d.zero_grad()
        self.scaler_d.scale(loss_d).backward()
        self.scaler_d.unscale_(self.optim_d)
        torch.nn.utils.clip_grad_norm_(
            list(self.net_d_mpd.parameters()) + list(self.net_d_msd.parameters()), 1.0
        )
        self.scaler_d.step(self.optim_d)
        self.scaler_d.update()

        # ---- Generator update ----
        with autocast(enabled=self.use_amp):
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.net_d_mpd(y_mel, y_hat)
            y_d_rs2, y_d_gs2, fmap_rs2, fmap_gs2 = self.net_d_msd(y_mel, y_hat)

            loss_mel = mel_spectrogram_loss(
                y_mel, y_hat,
                sample_rate=self.cfg.data.get("sample_rate", 22050),
                hop_size=self.cfg.data.get("hop_length", 256),
            )
            loss_kl = kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["z_mask"])
            loss_dur = out["l_length"] / max(x.shape[0], 1)

            loss_fm = feature_loss(fmap_rs + fmap_rs2, fmap_gs + fmap_gs2)
            loss_adv, _ = generator_loss(y_d_gs + y_d_gs2)

            loss_g = (
                loss_mel * 45.0
                + loss_kl * 1.0
                + loss_dur * 1.0
                + loss_fm * 2.0
                + loss_adv * 1.0
            )

        self.optim_g.zero_grad()
        self.scaler_g.scale(loss_g).backward()
        self.scaler_g.unscale_(self.optim_g)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 1.0)
        self.scaler_g.step(self.optim_g)
        self.scaler_g.update()

        return {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "loss_mel": loss_mel.item(),
            "loss_kl": loss_kl.item(),
            "loss_dur": loss_dur.item() if isinstance(loss_dur, torch.Tensor) else loss_dur,
            "loss_fm": loss_fm.item(),
            "loss_adv": loss_adv.item(),
        }

    # ------------------------------------------------------------------
    def _val_epoch(self) -> float:
        self.net_g.eval()
        total_mel = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["text_tokens"].to(self.device)
                x_lengths = batch["text_lengths"].to(self.device)
                spec = batch["spectrogram"].to(self.device)
                spec_lengths = batch["spec_lengths"].to(self.device)
                wav = batch["wav"].to(self.device).unsqueeze(1)
                sid = batch.get("speaker_id")
                if sid is not None:
                    sid = sid.to(self.device)
                g = batch.get("speaker_embedding")
                if g is not None:
                    g = g.to(self.device)

                out = self.net_g(x, x_lengths, spec, spec_lengths, sid=sid, g=g)
                y_hat = out["o"]
                y_slice = slice_segments(
                    wav, out["ids_slice"] * self.cfg.data.get("hop_length", 256),
                    segment_size=y_hat.shape[-1],
                )
                total_mel += mel_spectrogram_loss(y_slice, y_hat,
                                                   sample_rate=self.cfg.data.get("sample_rate", 22050),
                                                   hop_size=self.cfg.data.get("hop_length", 256)).item()
                n += 1
        self.net_g.train()
        return total_mel / max(n, 1)

    # ------------------------------------------------------------------
    def save_checkpoint(self, tag: str = "latest"):
        ckpt = {
            "epoch": self.epoch,
            "step": self.step,
            "net_g": self.net_g.state_dict(),
            "net_d_mpd": self.net_d_mpd.state_dict(),
            "net_d_msd": self.net_d_msd.state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        p = self.output_dir / f"checkpoint_{tag}.pth"
        torch.save(ckpt, p)
        self.logger.logger.info(f"Saved checkpoint → {p}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net_g.load_state_dict(ckpt["net_g"])
        self.net_d_mpd.load_state_dict(ckpt["net_d_mpd"])
        self.net_d_msd.load_state_dict(ckpt["net_d_msd"])
        self.optim_g.load_state_dict(ckpt["optim_g"])
        self.optim_d.load_state_dict(ckpt["optim_d"])
        self.epoch = ckpt.get("epoch", 0)
        self.step = ckpt.get("step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.logger.logger.info(f"Resumed from {path} (epoch {self.epoch})")

    # ------------------------------------------------------------------
    def train(self):
        self.logger.logger.info(
            f"Starting VITS2 training on {self.device} for "
            f"{self.cfg.training.epochs} epochs."
        )
        t0 = time.time()

        for epoch in range(self.epoch, self.cfg.training.epochs):
            self.epoch = epoch
            self.net_g.train()

            epoch_losses: Dict[str, list] = {}
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
            for batch in pbar:
                losses = self._train_step(batch)
                for k, v in losses.items():
                    epoch_losses.setdefault(k, []).append(v)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()
                                  if k in ("loss_g", "loss_d", "loss_mel")})
                self.step += 1

                if self.writer and self.step % 100 == 0:
                    for k, v in losses.items():
                        self.writer.add_scalar(f"train/{k}", v, self.step)

            avg = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
            self.logger.logger.info(
                f"[E{epoch+1}] G={avg['loss_g']:.4f}  D={avg['loss_d']:.4f}  "
                f"mel={avg['loss_mel']:.4f}  kl={avg['loss_kl']:.4f}"
            )

            # Validation
            val_mel = self._val_epoch()
            if self.writer:
                self.writer.add_scalar("val/loss_mel", val_mel, epoch)

            self.sched_g.step()
            self.sched_d.step()

            # Checkpointing
            if (epoch + 1) % self.cfg.training.get("save_every", 5) == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")
            if val_mel < self.best_val_loss:
                self.best_val_loss = val_mel
                self.save_checkpoint("best")

        self.save_checkpoint("final")
        if self.writer:
            self.writer.close()
        elapsed = (time.time() - t0) / 3600
        self.logger.logger.info(f"Training finished in {elapsed:.2f} h.")


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def add_args(parser):
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")


def run(args, config):
    trainer = VITS2Trainer(config)
    if getattr(args, "resume", None):
        trainer.load_checkpoint(args.resume)
    trainer.train()
