import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import math
import copy
from einops import rearrange
from typing import Optional, Tuple
import numpy as np


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size: int) -> torch.Tensor:
    """Mask out subsequent positions for causal attention."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Embedding(nn.Module):
    """Input embedding layer that processes spectrograms."""
    
    def __init__(self, dim_in: int = 80, dim_out: int = 128, 
                 units_in: int = 9, units_out: int = 18, drop_rate: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out * 2)
        self.conv = nn.Conv1d(units_in, units_out, kernel_size=5, stride=2, padding=2)
        self.dropout = nn.Dropout(drop_rate)
        self.norm = nn.LayerNorm(dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, units_in, dim_in)
        x = self.linear(x)  # (batch, units_in, dim_out * 2)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (batch, dim_out * 2, units_in)
        x = self.conv(x)  # (batch, units_out, new_length)
        x = x.transpose(1, 2)  # (batch, new_length, units_out)
        x = self.dropout(x)
        return x.reshape(x.shape[0], x.shape[1], -1)


class DepthWiseConv2d(nn.Module):
    """Depthwise separable convolution for efficient processing."""
    
    def __init__(self, dim_in: int, dim_out: int, kernel_size: int, 
                 padding: int, stride: int = 1, bias: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            dim_in, dim_in, kernel_size=kernel_size,
            padding=padding, groups=dim_in, stride=stride, bias=bias
        )
        self.norm = nn.BatchNorm2d(dim_in)
        self.pointwise = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pointwise(x)
        return x


class FeedForward(nn.Module):
    """Position-wise feed-forward network with depthwise convolutions."""
    
    def __init__(self, units: int = 16, dim: int = 128, P: int = 16, 
                 ratio: int = 4, drop_rate: float = 0.3):
        super().__init__()
        self.P = P
        dim_in = int(units * dim / P ** 2)
        
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * ratio, 1),
            nn.GELU(),
            DepthWiseConv2d(dim_in * ratio, dim_in * ratio, 3, 1),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(dim_in * ratio, dim_in, 1),
            nn.Dropout(drop_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = int(x.shape[1] / self.P)
        x = rearrange(x, 'b (c p1) (d p2) -> b (c d) p1 p2',
                      p1=self.P, p2=self.P)
        x = self.net(x)
        x = rearrange(x, 'b (c d) p1 p2 -> b (c p1) (d p2)', c=c)
        return x


class MultiHeadedAttention(nn.Module):
    """Multi-headed attention with positional encoding and depthwise convolutions."""
    
    def __init__(self, h: int = 8, units: int = 16, d_model: int = 392, 
                 drop_rate: float = 0.3, P: int = 16):
        super().__init__()
        assert d_model % h == 0
        
        self.P = P
        self.h = h
        self.d_k = d_model // h
        dim_in = int(units * d_model / P ** 2)
        
        self.wq = DepthWiseConv2d(dim_in, dim_in, 3, 1, 2, False)
        self.wk = DepthWiseConv2d(dim_in, dim_in, 3, 1, 2, False)
        self.wv = DepthWiseConv2d(dim_in, dim_in, 3, 1, 2, False)
        
        self.dropout = nn.Dropout(drop_rate)
        self.position = nn.Parameter(torch.randn(units//P, d_model//P, d_model//P))
        
        self.output_linear = nn.Sequential(
            nn.Conv1d(units//2, units, 1),
            nn.Linear(d_model // 2, d_model),
            nn.Dropout(drop_rate)
        )

    def scaled_dot_product_attention(self, query: torch.Tensor, key: torch.Tensor, 
                                   value: torch.Tensor, mask: Optional[torch.Tensor] = None,
                                   position: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention."""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        if position is not None:
            scores += position.unsqueeze(0)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        output = torch.matmul(p_attn, value)
        
        return output, p_attn

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = key.shape[1] // self.P
        
        # Reshape to 2D patches
        query, key, value = [
            rearrange(x, 'b (c p1) (d p2) -> b (c d) p1 p2', p1=self.P, p2=self.P) 
            for x in (query, key, value)
        ]

        if mask is not None:
            mask = mask.unsqueeze(1)

        # Apply convolutions
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        # Reshape for attention
        query = rearrange(query, 'b (c d) p1 p2 -> b c d (p1 p2)', c=c)
        key = rearrange(key, 'b (c d) p1 p2 -> b c d (p1 p2)', c=c)
        value = rearrange(value, 'b (c d) p1 p2 -> b c d (p1 p2)', c=c)

        # Apply attention
        x, attn = self.scaled_dot_product_attention(
            query, key, value, mask=mask, position=self.position
        )

        # Reshape and apply output projection
        x = rearrange(x, 'b c d (p1 p2) -> b (c p1) (d p2)', p1=self.P // 2)
        x = self.output_linear(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm architecture."""
    
    def __init__(self, h: int = 8, d_model: int = 392, units: int = 16, 
                 P: int = 16, drop_rate: float = 0.3):
        super().__init__()
        self.attention = MultiHeadedAttention(h, units, d_model, drop_rate, P)
        self.feed_forward = FeedForward(units, d_model, P, 4, drop_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        x_norm = self.norm1(x)
        x = x + self.dropout(self.attention(x_norm, x_norm, x_norm, mask))
        
        x_norm = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x_norm))
        
        return x


class Encoder(nn.Module):
    """Multi-layer transformer encoder."""
    
    def __init__(self, h: int = 8, d_model: int = 392, units: int = 16, 
                 P: int = 16, drop_rate: float = 0.3, layers: int = 2):
        super().__init__()
        self.layers = clones(
            TransformerBlock(h, d_model, units, P, drop_rate), layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ClassificationHead(nn.Module):
    """Classification head for VAD prediction."""
    
    def __init__(self, dim_in: int = 56, ratio: int = 2, units: int = 16, 
                 P: int = 16, drop_rate: float = 0.3):
        super().__init__()
        self.P = P
        c_units = int(dim_in * units / P**2)
        
        self.conv = DepthWiseConv2d(c_units, c_units, 5, 2, 2)
        
        c_linear = dim_in * units // (1 * 2**2 * P // 2)
        self.classifier = nn.Sequential(
            nn.Linear(c_linear, c_linear * ratio),
            nn.BatchNorm1d(P // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(c_linear * ratio, 1),
        )
        self.norm = nn.LayerNorm(dim_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = rearrange(x, 'b (c p1) (d p2) -> b (c d) p1 p2',
                      p1=self.P, p2=self.P)
        x = self.conv(x)
        x = rearrange(x, 'b c p1 p2 -> b p1 (c p2)')
        x = self.classifier(x)
        return x.squeeze(-1)


class VADModel(pl.LightningModule):
    """Voice Activity Detection model with Transformer architecture."""
    
    def __init__(self, 
                 dim_in: int = 80,
                 d_model: int = 56,
                 units_in: int = 8,
                 units: int = 16,
                 P: int = 16,
                 layers: int = 2,
                 heads: int = 8,
                 drop_rate: float = 0.3,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 reg_weight: float = 1e-5,
                 warmup_steps: int = 1000):
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding = Embedding(
            dim_in=dim_in, dim_out=d_model, 
            units_in=units_in, units_out=units, 
            drop_rate=drop_rate
        )
        
        self.encoder = Encoder(
            h=heads, d_model=d_model, units=units, 
            P=P, drop_rate=drop_rate, layers=layers
        )
        
        self.classifier = ClassificationHead(
            dim_in=d_model, ratio=2, units=units, 
            P=P, drop_rate=drop_rate
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Kaiming initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the VAD model."""
        x = self.embedding(x)
        x = self.encoder(x, mask)
        x = self.classifier(x)
        return x

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss with L2 regularization."""
        # Binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets)
        
        # L2 regularization
        l2_reg = sum(torch.norm(param, 2) for name, param in self.named_parameters() 
                    if 'weight' in name)
        
        total_loss = bce_loss + self.hparams.reg_weight * l2_reg
        
        return total_loss, bce_loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, targets = batch
        predictions = self(x)
        
        total_loss, bce_loss = self.compute_loss(predictions, targets)
        
        # Compute accuracy
        probs = torch.sigmoid(predictions)
        preds = (probs > 0.5).float()
        acc = (preds == targets).float().mean()
        
        # Log metrics
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_bce', bce_loss)
        self.log('train_acc', acc, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, targets = batch
        predictions = self(x)
        
        total_loss, bce_loss = self.compute_loss(predictions, targets)
        
        # Compute accuracy
        probs = torch.sigmoid(predictions)
        preds = (probs > 0.5).float()
        acc = (preds == targets).float().mean()
        
        # Log metrics
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_bce', bce_loss)
        self.log('val_acc', acc, prog_bar=True)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=1e-8
        )
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / self.hparams.warmup_steps
            else:
                progress = (step - self.hparams.warmup_steps) / (self.trainer.max_steps - self.hparams.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        predictions = self(x)
        probabilities = torch.sigmoid(predictions)
        return probabilities