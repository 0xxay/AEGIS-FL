import copy
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class MLPIDSModel(nn.Module):
    """
    Enhanced Multi-Layer Perceptron (MLP) for Intrusion Detection.
    Features LayerNorm (compatible with Differential Privacy), LeakyReLU activations,
    and calibrated dropout for high generalization in federated environments.
    """
    def __init__(self, input_dim: int = 42, num_classes: int = 2, hidden_dim1: int = 128, hidden_dim2: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.10),

            nn.Linear(hidden_dim2, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Returns deep copy of model state dictionary."""
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """Sets model parameters from state dictionary."""
        self.load_state_dict(weights)


class ResidualBlock(nn.Module):
    """Residual Feed-Forward Block with Skip Connection and LayerNorm."""
    def __init__(self, dim: int, dropout: float = 0.10):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResNetIDSModel(nn.Module):
    """
    1D Deep Residual Network for Intrusion Detection.
    Uses skip connections to preserve gradient flow during multi-round federated aggregation
    and prevent gradient degradation under DP noise.
    """
    def __init__(self, input_dim: int = 42, num_classes: int = 2, hidden_dim: int = 128, num_blocks: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(hidden_dim, dropout=0.10))
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.res_blocks(h)
        return self.head(h)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        self.load_state_dict(weights)


class AttentionIDSModel(nn.Module):
    """
    Feature-Attention Tabular Network.
    Computes self-attention across latent feature representations to model complex cross-attribute
    attack patterns (e.g. packet rates interacting with TTL and payload sizes).
    """
    def __init__(self, input_dim: int = 42, num_classes: int = 2, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Project inputs to latent embedding
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.1)
        )

        # Multi-Head Attention layer
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.10)
        self.norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x).unsqueeze(1) # (batch, 1, embed_dim)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm(h + attn_out).squeeze(1) # (batch, embed_dim)
        return self.mlp(h)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        self.load_state_dict(weights)


class AutoencoderIDSModel(nn.Module):
    """
    Deep Autoencoder for Unsupervised Intrusion Detection.
    Trained on benign network traffic. Anomalies trigger high reconstruction error.
    """
    def __init__(self, input_dim: int = 42, bottleneck_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, bottleneck_dim),
            nn.LeakyReLU(0.1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        self.load_state_dict(weights)


def create_ids_model(
    model_type: str = "mlp",
    input_dim: int = 42,
    num_classes: int = 2
) -> nn.Module:
    """
    Model factory instantiating deep neural network architectures for intrusion detection.
    Supports 'mlp', 'resnet', 'attention', and 'autoencoder'.
    """
    m = model_type.lower().strip()
    if m in ["mlp", "deep_mlp", "multilayer_perceptron"]:
        return MLPIDSModel(input_dim=input_dim, num_classes=num_classes)
    elif m in ["resnet", "res_net", "residual"]:
        return ResNetIDSModel(input_dim=input_dim, num_classes=num_classes)
    elif m in ["attention", "transformer", "self_attention"]:
        return AttentionIDSModel(input_dim=input_dim, num_classes=num_classes)
    elif m in ["autoencoder", "ae"]:
        return AutoencoderIDSModel(input_dim=input_dim)
    else:
        # Default fallback
        return MLPIDSModel(input_dim=input_dim, num_classes=num_classes)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Trains the model for one local epoch and returns average loss."""
    model.train()
    total_loss = 0.0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates model performance returning accuracy, loss, precision, recall, F1, and confusion matrix.
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm
    }
