import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


from . import get_storage_root

ROOT_DIR = get_storage_root()
CHECKPOINT_DIR = ROOT_DIR / "results" / "checkpoints"


class MLPIDSModel(nn.Module):
    """
    Enhanced Multi-Layer Perceptron (MLP) for Intrusion Detection.
    Uses LayerNorm (Opacus per-sample DP-SGD compatible) and LeakyReLU activations.
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
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
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
    Preserves gradient flow across federated rounds and under DP-SGD noise.
    """

    def __init__(self, input_dim: int = 42, num_classes: int = 2, hidden_dim: int = 128, num_blocks: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout=0.10) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.res_blocks(h)
        return self.head(h)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        self.load_state_dict(weights)


class AttentionIDSModel(nn.Module):
    """
    FT-Transformer-inspired Feature-Token Tabular Self-Attention Network.
    Tokenizes network flow attributes into `num_tokens=8` distinct feature-subspace tokens
    of dimension `token_dim=32` and performs true multi-head scaled dot-product attention
    across the 8 tokens (8x8 interaction matrix) using pure Linear/LayerNorm layers
    for full Opacus per-sample DP-SGD compatibility.
    """

    def __init__(
        self,
        input_dim: int = 42,
        num_classes: int = 2,
        num_tokens: int = 8,
        token_dim: int = 32,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads

        self.tokenizer = nn.Sequential(
            nn.Linear(input_dim, num_tokens * token_dim),
            nn.LayerNorm(num_tokens * token_dim),
            nn.LeakyReLU(0.1),
        )
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)
        self.norm1 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.10),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.head = nn.Sequential(
            nn.Linear(num_tokens * token_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, num_classes),
        )

    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = x.shape[0]
        tokens = self.tokenizer(x).view(bsz, self.num_tokens, self.token_dim)  # (B, T=8, D=32)

        q = self.q_proj(tokens).view(bsz, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(tokens).view(bsz, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).view(bsz, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, 8, 8)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(bsz, self.num_tokens, self.token_dim)

        h = self.norm1(tokens + self.out_proj(context))
        h = self.norm2(h + self.ffn(h))
        logits = self.head(h.reshape(bsz, self.num_tokens * self.token_dim))
        return logits, attn_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_attention(x)
        return logits

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        self.load_state_dict(weights)


class AutoencoderIDSModel(nn.Module):
    """
    Deep Autoencoder for Unsupervised Zero-Day Intrusion Detection.
    Trains strictly on benign flows via MSE reconstruction loss; carries a federated
    `anomaly_threshold` buffer inside `state_dict` so threshold consensus aggregates cleanly.
    """

    def __init__(self, input_dim: int = 42, bottleneck_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, bottleneck_dim),
            nn.LeakyReLU(0.1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, input_dim),
        )
        self.register_buffer("anomaly_threshold", torch.tensor(0.85, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Maps reconstruction MSE relative to calibrated threshold into 2-class pseudo-logits."""
        reconstructed = self.forward(x)
        mse = torch.mean((x - reconstructed) ** 2, dim=1)
        thresh = torch.clamp(self.anomaly_threshold, min=1e-4)
        score = (mse - thresh) / (0.35 * thresh)
        return torch.stack([-score, score], dim=1)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        self.load_state_dict(weights)


def create_ids_model(
    model_type: str = "mlp",
    input_dim: int = 42,
    num_classes: int = 2,
) -> nn.Module:
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
        return MLPIDSModel(input_dim=input_dim, num_classes=num_classes)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_module", model)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Trains supervised classifiers with CrossEntropyLoss or Autoencoder with benign-only MSELoss."""
    model.train()
    base_model = _unwrap_model(model)
    is_ae = isinstance(base_model, AutoencoderIDSModel)
    total_loss = 0.0
    benign_errors: List[torch.Tensor] = []
    steps = 0

    for features, targets in loader:
        if features.shape[0] == 0:
            continue
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        if is_ae:
            benign_mask = targets == 0
            train_x = features[benign_mask] if int(benign_mask.sum().item()) >= 2 else features
            recon = model(train_x)
            loss = F.mse_loss(recon, train_x)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                per_sample_err = torch.mean((train_x - recon.detach()) ** 2, dim=1)
                benign_errors.append(per_sample_err.cpu())
        else:
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        steps += 1

    if is_ae and benign_errors:
        all_err = torch.cat(benign_errors)
        q90 = float(torch.quantile(all_err, 0.90).item())
        base_model.anomaly_threshold.fill_(max(0.05, q90))

    return total_loss / max(1, steps)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluates model returning accuracy, precision, recall, F1, false positive rate (FPR),
    ROC-AUC, loss, and confusion matrix for both binary and multi-class IDS models.
    """
    model.eval()
    base_model = _unwrap_model(model)
    is_ae = isinstance(base_model, AutoencoderIDSModel)

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_attack_probs: List[float] = []
    all_prob_vectors: List[np.ndarray] = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            if is_ae:
                recon = base_model(features)
                loss = F.mse_loss(recon, features)
                logits = base_model.predict_logits(features)
                targets_eval = (targets > 0).long()
            else:
                logits = base_model(features)
                targets_eval = targets
                loss = criterion(logits, targets_eval)

            total_loss += float(loss.item())
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets_eval.cpu().numpy().tolist())
            if probs.shape[1] == 2:
                all_attack_probs.extend(probs[:, 1].cpu().numpy().tolist())
            else:
                all_prob_vectors.append(probs.cpu().numpy())

    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Compute SOC False Positive Rate (Benign class 0 misclassified as Attack > 0)
    benign_mask = y_true == 0
    if int(benign_mask.sum()) > 0:
        fpr = float(((y_pred > 0) & benign_mask).sum() / benign_mask.sum())
    else:
        fpr = 0.0

    # Compute ROC-AUC
    roc_auc = 0.0
    try:
        if all_attack_probs and len(np.unique(y_true)) == 2:
            roc_auc = float(roc_auc_score(y_true, np.array(all_attack_probs)))
        elif all_prob_vectors and len(np.unique(y_true)) > 2:
            prob_mat = np.vstack(all_prob_vectors)
            roc_auc = float(roc_auc_score(y_true, prob_mat, multi_class="ovr", average="weighted"))
    except Exception:
        roc_auc = acc

    return {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": round(fpr, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
    }


def save_model_checkpoint(
    model: nn.Module,
    dataset_name: str,
    model_type: str,
    input_dim: int,
    num_classes: int = 2,
    class_names: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persists trained federated global model + schema metadata to disk for live SOC inference."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    base_model = _unwrap_model(model)
    payload = {
        "dataset_name": dataset_name.lower(),
        "model_type": model_type.lower(),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "class_names": class_names or (["Normal", "Attack"] if num_classes == 2 else [f"Class {i}" for i in range(num_classes)]),
        "metrics": metrics or {},
        "state_dict": {k: v.detach().cpu() for k, v in base_model.state_dict().items()},
    }
    ckpt_path = CHECKPOINT_DIR / f"global_{dataset_name.lower()}_{model_type.lower()}.pt"
    latest_path = CHECKPOINT_DIR / "latest_global.pt"
    torch.save(payload, ckpt_path)
    torch.save(payload, latest_path)
    return ckpt_path


def predict_flows_from_dataframe(
    df_flows,
    checkpoint_path: Optional[Path] = None,
    mia_defense_eps: Optional[float] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Runs real-time intrusion inference on a raw or normalized DataFrame of network flows
    using a saved federated global checkpoint and its zero-leakage preprocessor.
    """
    import pandas as pd
    from .privacy import PrivacyEngineDP

    target_ckpt = checkpoint_path or (CHECKPOINT_DIR / "latest_global.pt")
    if not target_ckpt.exists():
        raise FileNotFoundError(f"No trained global checkpoint found at {target_ckpt}. Run an experiment first.")

    ckpt = torch.load(target_ckpt, map_location="cpu", weights_only=False)
    ds_name = ckpt["dataset_name"]
    proc_path = ROOT_DIR / "data" / "processed" / ds_name / "preprocessor.joblib"
    if not proc_path.exists():
        raise FileNotFoundError(f"Preprocessor missing at {proc_path}.")

    transformer = joblib.load(proc_path)
    X_proc = transformer.transform(df_flows)
    x_tensor = torch.tensor(X_proc, dtype=torch.float32)

    model = create_ids_model(
        model_type=ckpt["model_type"],
        input_dim=ckpt["input_dim"],
        num_classes=ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dp_guard = PrivacyEngineDP()
    with torch.no_grad():
        logits = model.predict_logits(x_tensor) if isinstance(model, AutoencoderIDSModel) else model(x_tensor)
        if mia_defense_eps is not None and mia_defense_eps > 0:
            logits = dp_guard.defend_inference(logits, epsilon_local=mia_defense_eps)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)

    class_names = ckpt["class_names"]
    if len(class_names) == 2:
        attack_prob = probs[:, 1]
    else:
        attack_prob = 1.0 - probs[:, 0]

    feat_names = getattr(transformer, "feature_names_out", [f"f_{i}" for i in range(X_proc.shape[1])])
    top_feat_idx = np.argmax(np.abs(X_proc), axis=1)
    top_triggers = [f"{feat_names[idx]} (z={X_proc[row_i, idx]:+.2f})" for row_i, idx in enumerate(top_feat_idx)]

    out_df = pd.DataFrame({
        "Flow_Index": np.arange(len(df_flows)),
        "Verdict": ["INTRUSION" if p > 0 else "BENIGN" for p in preds],
        "Predicted_Class": [class_names[p] if p < len(class_names) else str(p) for p in preds],
        "Threat_Confidence": np.round(attack_prob, 4),
        "Primary_Anomaly_Signal": top_triggers,
    })
    return out_df, ckpt
