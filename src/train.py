
"""
Advanced training system for TTS models with comprehensive monitoring and optimization.
Supports multiple model architectures, advanced optimization, and real-time monitoring.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

# Local imports
from src.model import AdvancedTTSModel, SimpleTTSModel, create_model, save_model, count_parameters
from src.data import TTSDataset, create_dataloader
from src.text_processor import TextProcessor
from src.logging_config import get_logger, log_performance
from src.utils import EarlyStopping, LearningRateScheduler, GradientClipper


class TTSTrainer:
    """Advanced trainer for TTS models with comprehensive features."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = self._setup_device()
        self.logger = get_logger(config.get('system', {}))
        
        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_optimization()
        self._setup_monitoring()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        self.logger.log_model_info(
            self.config.model.type,
            count_parameters(self.model),
            str(self.device)
        )
    
    def _setup_device(self) -> torch.device:
        """Setup computing device."""
        device_config = self.config.system.device
        
        if device_config == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                self.logger.logger.info("Using Apple Metal Performance Shaders (MPS)")
            else:
                device = torch.device("cpu")
                self.logger.logger.info("Using CPU")
        else:
            device = torch.device(device_config)
            self.logger.logger.info(f"Using configured device: {device}")
        
        return device
    
    def _setup_model(self):
        """Initialize model."""
        self.model = create_model(self.config.model)
        self.model.to(self.device)
        
        # Enable mixed precision if configured
        self.use_amp = self.config.performance.get('enable_fp16', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _setup_data(self):
        """Setup data loaders."""
        self.text_processor = TextProcessor(
            language=self.config.text_processing.language,
            phoneme_backend=self.config.text_processing.phoneme_backend
        )
        
        # Create datasets
        train_dataset = TTSDataset(
            self.config.data.train_path,
            sample_rate=self.config.data.sample_rate,
            text_processor=self.text_processor,
            config=self.config.data
        )
        
        val_dataset = TTSDataset(
            self.config.data.val_path,
            sample_rate=self.config.data.sample_rate,
            text_processor=self.text_processor,
            config=self.config.data
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.system.get('max_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.system.get('max_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.logger.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.logger.info(f"Validation samples: {len(val_dataset)}")
    
    def _setup_optimization(self):
        """Setup optimizer and scheduler."""
        # Optimizer
        if self.config.training.get('optimizer', 'Adam') == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.get('weight_decay', 1e-6)
            )
        elif self.config.training.get('optimizer') == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.get('weight_decay', 1e-6)
            )
        
        # Learning rate scheduler
        scheduler_config = self.config.training.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'ExponentialLR')
        
        if scheduler_type == 'ExponentialLR':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config.get('gamma', 0.95)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5)
            )
        elif scheduler_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        
        # Gradient clipping
        self.grad_clipper = GradientClipper(self.config.training.get('grad_clip_norm', 1.0))
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.get('early_stopping_patience', 20),
            min_delta=1e-4
        )
    
    def _setup_monitoring(self):
        """Setup monitoring and logging."""
        # Create output directory
        self.output_dir = Path(self.config.training.get('output_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard
        if self.config.monitoring.get('enable_tensorboard', True):
            log_dir = self.config.monitoring.get('tensorboard_log_dir', './logs')
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Weights & Biases
        self.use_wandb = self.config.monitoring.get('enable_wandb', False)
        if self.use_wandb:
            import wandb
            wandb.init(
                project=self.config.monitoring.get('wandb_project', 'speech-synthesis'),
                config=self.config
            )
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Dictionary of loss values
        """
        losses = {}
        
        # Mel spectrogram loss
        if 'mel_output' in outputs and 'mel_target' in targets:
            mel_loss = F.mse_loss(outputs['mel_output'], targets['mel_target'])
            losses['mel_loss'] = mel_loss
        
        # Duration loss
        if 'duration_output' in outputs and 'duration_target' in targets:
            duration_loss = F.mse_loss(outputs['duration_output'], targets['duration_target'])
            losses['duration_loss'] = duration_loss
        
        # Stop token loss
        if 'stop_output' in outputs and 'stop_target' in targets:
            stop_loss = F.binary_cross_entropy_with_logits(
                outputs['stop_output'], targets['stop_target']
            )
            losses['stop_loss'] = stop_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses
    
    @log_performance
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        batch['text_tokens'],
                        mel_targets=batch.get('mel_spectrogram'),
                        speaker_embeddings=batch.get('speaker_embedding')
                    )
                    losses = self.compute_loss(outputs, batch)
                
                # Backward pass with scaling
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.unscale_(self.optimizer)
                self.grad_clipper(self.model.parameters())
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    batch['text_tokens'],
                    mel_targets=batch.get('mel_spectrogram'),
                    speaker_embeddings=batch.get('speaker_embedding')
                )
                losses = self.compute_loss(outputs, batch)
                
                # Backward pass
                losses['total_loss'].backward()
                self.grad_clipper(self.model.parameters())
                self.optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Accumulate losses
            for loss_name, loss_value in losses.items():
                if loss_name not in epoch_losses:
                    epoch_losses[loss_name] = []
                epoch_losses[loss_name].append(loss_value.item())
            
            # Log to tensorboard
            if self.writer and self.global_step % 100 == 0:
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f'train/{loss_name}', loss_value.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', 
                                     self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    @log_performance
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    batch['text_tokens'],
                    mel_targets=batch.get('mel_spectrogram'),
                    speaker_embeddings=batch.get('speaker_embedding')
                )
                losses = self.compute_loss(outputs, batch)
                
                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    if loss_name not in epoch_losses:
                        epoch_losses[loss_name] = []
                    epoch_losses[loss_name].append(loss_value.item())
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_data = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint_data, checkpoint_path)
        
        # Best model checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint_data, best_path)
            self.logger.logger.info(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
        
        # Keep only last N checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints to save disk space."""
        checkpoints = list(self.output_dir.glob("checkpoint_epoch_*.pth"))
        if len(checkpoints) > keep_last:
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
    def train(self):
        """Main training loop."""
        self.logger.logger.info("Starting training...")
        start_time = time.time()
        
        try:
            for epoch in range(self.config.training.epochs):
                self.current_epoch = epoch
                
                # Training
                train_losses = self.train_epoch()
                
                # Validation
                if epoch % self.config.training.get('validate_every', 1) == 0:
                    val_losses = self.validate_epoch()
                else:
                    val_losses = {}
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_losses.get('total_loss', train_losses['total_loss']))
                    else:
                        self.scheduler.step()
                
                # Logging
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': train_losses.get('total_loss', 0),
                    'val_loss': val_losses.get('total_loss', 0),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                self.training_history.append(epoch_info)
                self.logger.log_training_metrics(epoch + 1, epoch_info)
                
                # Tensorboard logging
                if self.writer:
                    for loss_name, loss_value in train_losses.items():
                        self.writer.add_scalar(f'epoch/train_{loss_name}', loss_value, epoch)
                    for loss_name, loss_value in val_losses.items():
                        self.writer.add_scalar(f'epoch/val_{loss_name}', loss_value, epoch)
                
                # Weights & Biases logging
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        **{f'train/{k}': v for k, v in train_losses.items()},
                        **{f'val/{k}': v for k, v in val_losses.items()},
                        'epoch': epoch + 1,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                
                # Save checkpoints
                current_val_loss = val_losses.get('total_loss', train_losses['total_loss'])
                is_best = current_val_loss < self.best_val_loss
                
                if is_best:
                    self.best_val_loss = current_val_loss
                
                if (epoch + 1) % self.config.training.save_every == 0:
                    self.save_checkpoint(is_best)
                
                # Early stopping
                if val_losses and self.early_stopping(val_losses['total_loss']):
                    self.logger.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        except KeyboardInterrupt:
            self.logger.logger.info("Training interrupted by user")
        
        except Exception as e:
            self.logger.log_error_with_context(e, {"epoch": self.current_epoch})
            raise
        
        finally:
            # Final save
            self.save_checkpoint(is_best=False)
            
            # Close monitoring
            if self.writer:
                self.writer.close()
            
            if self.use_wandb:
                import wandb
                wandb.finish()
            
            total_time = time.time() - start_time
            self.logger.logger.info(f"Training completed in {total_time / 3600:.2f} hours")


def add_args(parser):
    """Add training arguments to parser."""
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')


def run(args):
    """Run training with command line arguments."""
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override with command line arguments
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.device != 'auto':
        config.system.device = args.device
    
    # Create trainer
    trainer = TTSTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train TTS model")
    add_args(parser)
    args = parser.parse_args()
    run(args)
        print(f"Epoch {epoch+1}/{config.training.epochs} done.")

def add_args(parser):
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output', type=str, default='models/', help='Output directory for checkpoints')

def run(args):
    train_loop(args.config, args.data, args.output)
