from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .models import MLPIDSModel, create_ids_model, train_epoch
from .privacy import PrivacyEngineDP, LocalDPEngine, PedersenCommitment
from .attacks import AdversarialAttacks


ROOT_DIR = Path(__file__).resolve().parents[1]

CLIENT_IDENTITIES = {
    0: {"name": "Client 1 - Bank", "sector": "Financial Services", "icon": "🏦"},
    1: {"name": "Client 2 - Hospital", "sector": "Healthcare / IoMT", "icon": "🏥"},
    2: {"name": "Client 3 - Enterprise", "sector": "Corporate IT", "icon": "🏢"},
    3: {"name": "Client 4 - Government", "sector": "Defense / Gov Subnet", "icon": "🏛️"},
    4: {"name": "Client 5 - Cloud Provider", "sector": "Public Cloud Infrastructure", "icon": "☁️"}
}


class InstitutionalSuperNode:
    """
    Simulated Institutional SuperNode executing edge training.
    Enforces that raw network data strictly never leaves the institutional perimeter.
    """
    def __init__(
        self,
        client_id: int,
        dataset_name: str = "unsw",
        input_dim: int = 42,
        model_type: str = "mlp",
        dp_enabled: bool = True,
        clipping_norm: float = 1.0,
        noise_multiplier: float = 0.05,
        sparsify: bool = False,
        sparsity_ratio: float = 0.05,
        local_dp: bool = False,
        local_dp_epsilon: float = 2.0,
        pedersen_commitment: bool = True,
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.identity = CLIENT_IDENTITIES.get(client_id, {"name": f"Client {client_id}", "sector": "Unknown", "icon": "💻"})
        self.name = self.identity["name"]
        self.sector = self.identity["sector"]
        self.icon = self.identity["icon"]

        self.dataset_name = dataset_name.lower()
        self.input_dim = input_dim
        self.model_type = model_type
        self.device = torch.device(device)

        # Local private model
        self.model = create_ids_model(model_type=model_type, input_dim=input_dim).to(self.device)

        # Differential Privacy Engine
        self.dp_enabled = dp_enabled
        self.dp_engine = PrivacyEngineDP(
            clipping_norm=clipping_norm,
            noise_multiplier=noise_multiplier if dp_enabled else 0.0
        )
        self.sparsify = sparsify
        self.sparsity_ratio = sparsity_ratio

        # Local Differential Privacy (Label Privacy via Randomized Response)
        self.local_dp = local_dp
        self.local_dp_epsilon = local_dp_epsilon

        # Cryptographic Commitment Scheme
        self.pedersen_commitment = pedersen_commitment
        self.pedersen_engine = PedersenCommitment()

        # Adversarial Attack Configuration
        self.is_compromised = False
        self.attack_type = "none" # "none", "label_flipping", "model_scaling", "backdoor"
        self.attack_scale = -5.0

        # Data placeholders
        self.train_loader: Optional[DataLoader] = None
        self.num_samples = 0
        self.num_attacks = 0
        self.num_normals = 0

    def load_data(self, batch_size: int = 64):
        """Loads private dataset partition from data/clients/{dataset}/client_{id}.npz."""
        client_file = ROOT_DIR / "data" / "clients" / self.dataset_name / f"client_{self.client_id}.npz"
        if not client_file.exists():
            raise FileNotFoundError(f"Partition missing for {self.name}: {client_file}")

        data = np.load(client_file)
        X = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.long)

        self.num_samples = len(y)
        self.num_normals = int((y == 0).sum().item())
        self.num_attacks = int((y == 1).sum().item())

        # If Local DP is active on honest node, apply Randomized Response to private labels
        if self.local_dp and not self.is_compromised:
            y = LocalDPEngine.randomized_response_labels(y, epsilon=self.local_dp_epsilon)

        # If compromised with data poisoning, flip labels during ingestion
        if self.is_compromised and self.attack_type == "label_flipping":
            y = AdversarialAttacks.apply_label_flipping(y, flip_rate=0.90)

        dataset = TensorDataset(X, y)
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_local_round(
        self,
        global_weights: Dict[str, torch.Tensor],
        epochs: int = 1,
        lr: float = 0.001
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Executes local epoch training starting from current global model weights.
        Returns differential update (ΔW) and training telemetry.
        """
        if self.train_loader is None:
            self.load_data()

        # Step 1: Initialize local network with received global model parameters
        self.model.set_weights(global_weights)

        # Step 2: Local training using Adam
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_loss = 0.0
        for _ in range(epochs):
            loss = train_epoch(self.model, self.train_loader, optimizer, criterion, self.device)
            train_loss += loss
        avg_loss = train_loss / max(1, epochs)

        # Step 3: Compute raw weight differential ΔW = W_local - W_global
        local_weights = self.model.get_weights()
        raw_update: Dict[str, torch.Tensor] = {}
        for key in global_weights.keys():
            raw_update[key] = local_weights[key] - global_weights[key].cpu()

        # Step 4: If compromised with model poisoning, tamper with weight differential
        if self.is_compromised and self.attack_type in ["model_scaling", "sign_flipping"]:
            raw_update = AdversarialAttacks.apply_model_poisoning(raw_update, scale_factor=self.attack_scale)

        # Step 5: Differential Privacy Clipping and Perturbation (with optional Top-k Sparsification)
        # Honest nodes enforce DP clipping; compromised nodes attempt to bypass it to amplify poisoning
        if self.dp_enabled and not (self.is_compromised and self.attack_type == "model_scaling"):
            protected_update, pre_clip_norm = self.dp_engine.perturb_update(
                raw_update,
                sparsify=self.sparsify,
                sparsity_ratio=self.sparsity_ratio
            )
        else:
            if self.sparsify:
                protected_update = self.dp_engine.sparsify_update(raw_update, sparsity_ratio=self.sparsity_ratio)
            else:
                protected_update = raw_update
            pre_clip_norm = float(torch.sqrt(sum(torch.sum(t ** 2) for t in raw_update.values() if t.is_floating_point())).item())

        # Step 6: Optional Cryptographic Commitment (Pedersen Commitment)
        commitment_hash, blinding_factor = None, None
        if self.pedersen_commitment:
            # Commit to the scalar pre_clip_norm
            commitment_hash, blinding_factor = self.pedersen_engine.commit(np.array([pre_clip_norm], dtype=np.float64))

        privacy_stats = self.dp_engine.get_privacy_budget()
        telemetry = {
            "client_id": self.client_id,
            "client_name": self.name,
            "train_loss": round(avg_loss, 4),
            "pre_clip_norm": round(pre_clip_norm, 4),
            "num_samples": self.num_samples,
            "epsilon": privacy_stats["epsilon"],
            "rdp_epsilon": privacy_stats.get("rdp_epsilon", privacy_stats["epsilon"]),
            "commitment_hash": commitment_hash,
            "blinding_factor": blinding_factor
        }

        return protected_update, telemetry
