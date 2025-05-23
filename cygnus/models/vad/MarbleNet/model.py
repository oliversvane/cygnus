import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl

from cygnus.visualize.segment_prediction import prediction_visual

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class SeparableConv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MaskedConv1d(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x, length=None):
        if length is not None:
            max_len = x.size(2)
            mask = torch.arange(max_len, device=x.device).expand(
                x.size(0), max_len
            ) < length.unsqueeze(1)
            x = x * mask.unsqueeze(1).float()
        return self.conv(x)


class MarbleNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        dropout=0.0,
        activation="swish",
        normalization="batch",
        se=True,
        se_reduction_ratio=16,
    ):
        super().__init__()

        self.use_se = se
        self.use_residual = stride == 1 and in_channels == out_channels

        padding = (kernel_size - 1) * dilation // 2

        # Depthwise separable convolution
        self.separable_conv = SeparableConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        # Normalization
        if normalization == "batch":
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        elif normalization == "layer":
            self.norm = nn.GroupNorm(1, out_channels)
        else:
            self.norm = nn.Identity()

        # Activation
        if activation == "swish":
            self.activation = Swish()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Squeeze-and-Excite
        if self.use_se:
            self.se = SqueezeExcite(out_channels, se_reduction_ratio)

        # Residual projection if needed
        if not self.use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv1d(
                in_channels, out_channels, 1, stride=stride, bias=False
            )
            self.residual_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)

    def forward(self, x, length=None):
        residual = x

        # Main path
        out = self.separable_conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        if self.use_se:
            out = self.se(out)

        # Residual connection
        if self.use_residual:
            out = out + residual
        elif hasattr(self, "residual_conv"):
            residual = self.residual_conv(residual)
            residual = self.residual_norm(residual)
            out = out + residual

        return out


class MarbleNet(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        B=3,
        R=2,
        C=128,
        feat_in=64,
        activation="swish",
        normalization="batch",
        dropout=0.0,
        se=True,
        se_reduction_ratio=16,
        kernel_size_factor=1.0,
        lr=1e-3,
        weight_decay=1e-3,
    ):
        """
        NVIDIA MarbleNet implementation

        Args:
            num_classes: Number of output classes
            B: Number of blocks
            R: Number of sub-blocks (repetitions) per block
            C: Number of channels in the first block
            feat_in: Number of input features (spectrogram features)
            activation: Activation function ('swish', 'relu')
            normalization: Normalization type ('batch', 'layer')
            dropout: Dropout probability
            se: Whether to use Squeeze-and-Excite
            se_reduction_ratio: SE reduction ratio
            kernel_size_factor: Factor to scale kernel sizes
            lr: Learning rate
            weight_decay: Weight decay
        """
        super().__init__()
        self.save_hyperparameters()

        self.B = B
        self.R = R
        self.C = C
        self.num_classes = num_classes

        # Input preprocessing - convert spectrogram to initial channels
        self.preprocessor = nn.Sequential(
            nn.Conv1d(feat_in, C, kernel_size=1, bias=False),
            nn.BatchNorm1d(C, eps=1e-3, momentum=0.1),
            Swish() if activation == "swish" else nn.ReLU(),
        )

        # MarbleNet blocks
        self.blocks = nn.ModuleList()
        channels = [C * (2**i) for i in range(B)]  # [C, 2C, 4C, ...]
        kernel_sizes = [
            int(11 + 2 * i * kernel_size_factor) for i in range(B)
        ]  # [11, 13, 15, ...]

        in_channels = C
        for b in range(B):
            out_channels = channels[b]
            kernel_size = kernel_sizes[b]

            for r in range(R):
                # First sub-block in each block (except first block) has stride=2 for downsampling
                stride = 2 if (r == 0 and b > 0) else 1

                # Channel change happens in first sub-block of each block
                block_in_channels = in_channels if r == 0 else out_channels

                self.blocks.append(
                    MarbleNetBlock(
                        block_in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        dilation=1,
                        dropout=dropout,
                        activation=activation,
                        normalization=normalization,
                        se=se,
                        se_reduction_ratio=se_reduction_ratio,
                    )
                )

            in_channels = out_channels

        final_channels = channels[-1]

        # Classification head - exactly as in MarbleNet
        self.classifier = nn.Sequential(
            nn.Conv1d(final_channels, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(final_channels, eps=1e-3, momentum=0.1),
            Swish() if activation == "swish" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(final_channels, self.num_classes, kernel_size=1, bias=True),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Initialize weights
        self.apply(self._init_weights)
        self.training_step_loss = []
        self.training_step_acc = []
        self.validation_step_loss = []
        self.validation_step_acc = []

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

    def forward(self, x, length=None):
        """
        Args:
            x: Input tensor of shape (B, feat_in, T) for spectrograms
               For 2D inputs (B, 1, H, W), will be reshaped to (B, H, W)
            length: Optional sequence lengths for masking
        """
        # Handle 2D input by treating height as features and width as time
        if x.dim() == 4:  # (B, 1, H, W) -> (B, H, W)
            x = x.squeeze(1)

        # Preprocessing
        x = self.preprocessor(x)

        # MarbleNet blocks
        for block in self.blocks:
            x = block(x, length)

        # Classification head
        x = self.classifier(x)

        # Global pooling
        x = self.global_pool(x)  # (B, num_classes, 1)
        x = x.squeeze(-1)  # (B, num_classes)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
        )

        # Cosine annealing scheduler (common in MarbleNet training)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        if self.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y.float())
            preds = torch.sigmoid(logits.squeeze()) > 0.5
            acc = (preds == y.bool()).float().mean()
        else:
            loss = F.cross_entropy(logits, y.float())
            acc = ((logits > 0.5) == y.bool()).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.training_step_loss.append(loss)
        self.log("train_acc", acc, prog_bar=True)
        self.training_step_acc.append(acc)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        if self.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y.float())
            preds = torch.sigmoid(logits.squeeze()) > 0.5
            acc = (preds == y.bool()).float().mean()
        else:
            loss = F.cross_entropy(logits, y.float())
            acc = ((logits > 0.5) == y.bool()).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.validation_step_loss.append(loss)
        self.log("val_acc", acc, prog_bar=True)
        self.validation_step_acc.append(acc)



        
        
        return {"val_loss": loss, "val_acc": acc}

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_loss).mean()
        avg_acc = torch.stack(self.training_step_acc).mean()
        self.log("epoch_train_loss", avg_loss, prog_bar=True)
        self.log("epoch_train_acc", avg_acc, prog_bar=True)
        self.training_step_loss.clear()
        self.training_step_acc.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_loss).mean()
        avg_acc = torch.stack(self.validation_step_acc).mean()
        self.log("epoch_val_loss", avg_loss, prog_bar=True)
        self.log("epoch_val_acc", avg_acc, prog_bar=True)
        self.validation_step_loss.clear()
        self.validation_step_acc.clear()
        # Take one batch
        val_loader = self.trainer.datamodule.val_dataloader()
        x, y = next(iter(val_loader))
        x, y = x.to(self.device), y.to(self.device)

        # Get predictions
        logits = self(x)
        preds = torch.sigmoid(logits).detach().cpu().squeeze()

        # Prepare data for the first sample
        true_labels = y[0].cpu().squeeze()
        pred_labels = preds[0].squeeze()

        # Plot actual vs predicted
        plt.figure(figsize=(12, 3))
        plt.plot(true_labels, label="Actual", linewidth=1)
        plt.plot(pred_labels, label="Predicted", alpha=0.7, linestyle='--')
        plt.title("Speech Presence Prediction")
        plt.xlabel("Time (ms)")
        plt.ylabel("Speech Presence")
        plt.legend()
        plt.tight_layout()

        # Log to TensorBoard
        self.logger.experiment.add_figure("Speech Prediction vs Actual", plt.gcf(), self.current_epoch)
        plt.close()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
