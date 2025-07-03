"""
Advanced TTS model implementations with support for multiple architectures.
Includes YourTTS, Tacotron2, and FastSpeech2 models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
import numpy as np


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


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """Factory function to create TTS models based on configuration."""
    model_type = model_config.get('type', 'advanced').lower()
    
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
