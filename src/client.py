from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .models import create_ids_model, train_epoch
from .privacy import PrivacyEngineDP, LocalDPEngine, PedersenCommitment
from .attacks import AdversarialAttacks


ROOT_DIR = Path(__file__).resolve().parents[1]

CLIENT_IDENTITIES = {
    0: {"name": "Client 1 - Bank", "sector": "Financial Services", "icon": "🏦"},
    1: {"name": "Client 2 - Hospital", "sector": "Healthcare / IoMT", "icon": "🏥"},
    2: {"name": "Client 3 - Enterprise", "sector": "Corporate IT", "icon": "🏢"},
    3: {"name": "Client 4 - Government", "sector": "Defense / Gov Subnet", "icon": "🏛️"},
    4: {"name": "Client 5 - Cloud Provider", "sector": "Public Cloud Infrastructure", "icon": "☁️"},
}


class InstitutionalSuperNode:
    """
    Institutional SuperNode executing edge training inside its sovereign perimeter.
    Supports:
    - Real Opacus per-sample DP-SGD (`use_opacus=True`) with tight Poisson-subsampled RDP accounting
    - Energy-preserving Top-k gradient sparsification
    - Local DP via Randomized Response on labels (binary & multi-class)
    - Verifiable SHA-256 + Homomorphic Pedersen commitments over update norms
    - Full adversarial simulation (`label_flipping`, `model_scaling`, `adaptive_poisoning`, `backdoor`)
    """

    def __init__(
        self,
        client_id: int,
        dataset_name: str = "unsw",
        input_dim: int = 42,
        num_classes: int = 2,
        multiclass: bool = False,
        model_type: str = "mlp",
        dp_enabled: bool = True,
        use_opacus: bool = True,
        clipping_norm: float = 1.0,
        noise_multiplier: float = 0.05,
        sparsify: bool = False,
        sparsity_ratio: float = 0.05,
        local_dp: bool = False,
        local_dp_epsilon: float = 2.0,
        pedersen_commitment: bool = True,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.identity = CLIENT_IDENTITIES.get(
            client_id, {"name": f"Client {client_id}", "sector": "Unknown", "icon": "💻"}
        )
        self.name = self.identity["name"]
        self.sector = self.identity["sector"]
        self.icon = self.identity["icon"]

        self.dataset_name = dataset_name.lower()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.multiclass = multiclass
        self.model_type = model_type
        self.device = torch.device(device)

        self.model = create_ids_model(model_type=model_type, input_dim=input_dim, num_classes=num_classes).to(self.device)

        self.dp_enabled = dp_enabled
        self.use_opacus = use_opacus
        self.dp_engine = PrivacyEngineDP(
            clipping_norm=clipping_norm,
            noise_multiplier=noise_multiplier if dp_enabled else 0.0,
        )
        self.sparsify = sparsify
        self.sparsity_ratio = sparsity_ratio

        self.local_dp = local_dp
        self.local_dp_epsilon = local_dp_epsilon

        self.pedersen_commitment = pedersen_commitment
        self.pedersen_engine = PedersenCommitment()

        self.is_compromised = False
        self.attack_type = "none"  # "none", "label_flipping", "model_scaling", "adaptive_poisoning", "backdoor"
        self.attack_scale = -25.0

        self.train_loader: Optional[DataLoader] = None
        self.batch_size: int = 64
        self.num_samples = 0
        self.num_attacks = 0
        self.num_normals = 0

    def load_data(self, batch_size: int = 64):
        """Loads private dataset partition from `data/clients/{dataset}/client_{id}.npz`."""
        self.batch_size = batch_size
        client_file = ROOT_DIR / "data" / "clients" / self.dataset_name / f"client_{self.client_id}.npz"
        if not client_file.exists():
            raise FileNotFoundError(f"Partition missing for {self.name}: {client_file}")

        data = np.load(client_file)
        X = torch.tensor(data["X"], dtype=torch.float32)
        y_bin = torch.tensor(data["y"], dtype=torch.long)
        if self.multiclass and "y_multi" in data:
            y = torch.tensor(data["y_multi"], dtype=torch.long)
        else:
            y = y_bin

        self.num_samples = len(y)
        self.num_normals = int((y_bin == 0).sum().item())
        self.num_attacks = int((y_bin == 1).sum().item())

        if self.local_dp and not self.is_compromised:
            y = LocalDPEngine.randomized_response_labels(
                y, epsilon=self.local_dp_epsilon, num_classes=self.num_classes
            )

        if self.is_compromised:
            if self.attack_type == "label_flipping":
                y = AdversarialAttacks.apply_label_flipping(y, flip_rate=0.95, num_classes=self.num_classes)
            elif self.attack_type == "backdoor":
                X, y = AdversarialAttacks.apply_targeted_backdoor(
                    X, y, trigger_value=4.5, target_label=0, poison_fraction=0.55
                )

        dataset = TensorDataset(X, y)
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_local_round(
        self,
        global_weights: Dict[str, torch.Tensor],
        epochs: int = 1,
        lr: float = 0.001,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Executes local edge training, applies DP/sparsification/commitments, and returns (dW, telemetry)."""
        if self.train_loader is None:
            self.load_data(batch_size=self.batch_size)

        self.model.set_weights(global_weights)
        criterion = nn.CrossEntropyLoss()

        used_per_sample_dp = False
        if self.dp_enabled and self.use_opacus and not (self.is_compromised and self.attack_type in ["model_scaling", "adaptive_poisoning"]):
            try:
                avg_loss, _ = self.dp_engine.train_with_opacus(
                    model=self.model,
                    train_loader=self.train_loader,
                    epochs=epochs,
                    lr=lr,
                    criterion=criterion,
                    device=self.device,
                )
                used_per_sample_dp = True
            except Exception:
                used_per_sample_dp = False

        if not used_per_sample_dp:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            train_loss = 0.0
            for _ in range(epochs):
                train_loss += train_epoch(self.model, self.train_loader, optimizer, criterion, self.device)
            avg_loss = train_loss / max(1, epochs)

        local_weights = self.model.get_weights()
        raw_update: Dict[str, torch.Tensor] = {
            key: (local_weights[key] - global_weights[key].cpu() if local_weights[key].is_floating_point() else local_weights[key])
            for key in global_weights.keys()
        }

        if self.is_compromised:
            if self.attack_type in ["model_scaling", "sign_flipping"]:
                raw_update = AdversarialAttacks.apply_model_poisoning(raw_update, scale_factor=self.attack_scale)
            elif self.attack_type == "adaptive_poisoning":
                raw_update = AdversarialAttacks.apply_adaptive_norm_poisoning(
                    raw_update, target_norm=self.dp_engine.clipping_norm
                )

        if self.dp_enabled and not used_per_sample_dp and not (self.is_compromised and self.attack_type == "model_scaling"):
            protected_update, pre_clip_norm = self.dp_engine.perturb_update(
                raw_update, sparsify=self.sparsify, sparsity_ratio=self.sparsity_ratio
            )
        else:
            protected_update = (
                self.dp_engine.sparsify_update(raw_update, sparsity_ratio=self.sparsity_ratio)
                if self.sparsify
                else raw_update
            )
            pre_clip_norm = math.sqrt(
                sum(torch.sum(t ** 2).item() for t in protected_update.values() if t.is_floating_point())
            )

        # Compute exact post-DP L2 norm that the client commits to before server submission
        committed_norm = round(
            math.sqrt(sum(torch.sum(t ** 2).item() for t in protected_update.values() if t.is_floating_point())),
            6,
        )

        commitment_hash, blinding_factor = None, None
        homomorphic_commitment, homomorphic_blinding = None, None
        if self.pedersen_commitment:
            commitment_hash, blinding_factor = self.pedersen_engine.commit(
                np.array([committed_norm], dtype=np.float64)
            )
            homomorphic_commitment, homomorphic_blinding = self.pedersen_engine.commit_homomorphic(committed_norm)

        privacy_stats = self.dp_engine.get_privacy_budget()
        telemetry = {
            "client_id": self.client_id,
            "client_name": self.name,
            "train_loss": round(avg_loss, 4),
            "pre_clip_norm": round(pre_clip_norm, 4),
            "committed_norm": committed_norm,
            "num_samples": self.num_samples,
            "dp_mode": "opacus_per_sample" if used_per_sample_dp else ("client_clip" if self.dp_enabled else "disabled"),
            "epsilon": privacy_stats["epsilon"],
            "rdp_epsilon": privacy_stats.get("rdp_epsilon", privacy_stats["epsilon"]),
            "commitment_hash": commitment_hash,
            "blinding_factor": blinding_factor,
            "homomorphic_commitment": homomorphic_commitment,
            "homomorphic_blinding": homomorphic_blinding,
        }

        return protected_update, telemetry
