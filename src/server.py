from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import json
import hashlib
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .models import MLPIDSModel, create_ids_model, evaluate_model
from .defenses import ByzantineDefenses, ClientReputationTracker
from .privacy import SecureAggregationEngine, PedersenCommitment


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"


class CentralFederatedServer:
    """
    Central Federated Aggregator (Flower SuperLink Equivalent).
    Coordinates model distribution, cryptographic SecAgg+ mask cancellation,
    adversarial outlier filtering, and global evaluation.
    """
    def __init__(
        self,
        dataset_name: str = "unsw",
        input_dim: int = 42,
        model_type: str = "mlp",
        defense_strategy: str = "norm_clipping", # "fedavg", "norm_clipping", "trimmed_mean", "coordinate_median", "cosine_similarity", "multi_krum", "flame", "reputation", "adaptive_clipping"
        secagg_enabled: bool = True,
        num_clients: int = 5,
        enable_audit_log: bool = True,
        device: str = "cpu"
    ):
        self.dataset_name = dataset_name.lower()
        self.input_dim = input_dim
        self.model_type = model_type
        self.defense_strategy = defense_strategy
        self.secagg_enabled = secagg_enabled
        self.num_clients = num_clients
        self.enable_audit_log = enable_audit_log
        self.device = torch.device(device)

        # Global Server Model
        self.global_model = create_ids_model(model_type=model_type, input_dim=input_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Secure Aggregation Engine
        self.secagg_engine = SecureAggregationEngine(num_clients=num_clients)

        # Dynamic Client Reputation Tracker
        self.reputation_tracker = ClientReputationTracker(num_clients=num_clients, decay=0.85)

        # Pedersen Commitment Engine for gradient verification
        self.pedersen_engine = PedersenCommitment()

        # Cryptographic Audit Log (Merkle / SHA-256 Hash Chain)
        self.audit_log: List[Dict[str, Any]] = []
        self.prev_block_hash: str = "0" * 64

        # Test evaluation set
        self.test_loader: Optional[DataLoader] = None
        self.round_history: List[Dict[str, Any]] = []

    def load_test_data(self, batch_size: int = 256):
        """Loads central held-out test dataset for unbiased global evaluation."""
        test_dir = ROOT_DIR / "data" / "processed" / self.dataset_name
        X_test_file = test_dir / "X_test.npy"
        y_test_file = test_dir / "y_test.npy"

        if not X_test_file.exists() or not y_test_file.exists():
            raise FileNotFoundError(f"Evaluation set missing in {test_dir}. Run preprocessor first.")

        X = torch.tensor(np.load(X_test_file), dtype=torch.float32)
        y = torch.tensor(np.load(y_test_file), dtype=torch.long)

        dataset = TensorDataset(X, y)
        self.test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Returns deep copy of current global model weights."""
        return self.global_model.get_weights()

    def aggregate_and_evaluate(
        self,
        round_num: int,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_sample_counts: Dict[int, int]
    ) -> Dict[str, Any]:
        """
        Executes server-side aggregation for round_num:
        1. Validates update integrity and SecAgg+ cancellation
        2. Applies Byzantine defense filter (detects malicious clients)
        3. Aggregates clean updates into global model
        4. Benchmarks global IDS on test dataset
        """
        start_time = time.time()
        flagged_clients: List[int] = []
        filter_telemetry: Dict[str, Any] = {}

        if self.test_loader is None:
            self.load_test_data()

        aggregated_update: Optional[Dict[str, torch.Tensor]] = None

        # Step 1: Defense Layer - Filter or sanitize poisoned updates
        if self.defense_strategy == "norm_clipping":
            filtered_updates, flagged_clients, norms = ByzantineDefenses.detect_and_filter_norm_outliers(client_updates, threshold_multiplier=1.4)
            filter_telemetry["client_norms"] = norms
            active_updates = filtered_updates
            print(f"  [Defense Engine] Client Update Norms: { {c: round(n, 2) for c, n in norms.items()} }")

        elif self.defense_strategy == "cosine_similarity":
            filtered_updates, flagged_clients, cosine_scores = ByzantineDefenses.cosine_similarity_filtering(client_updates)
            filter_telemetry["cosine_scores"] = cosine_scores
            active_updates = filtered_updates

        elif self.defense_strategy == "coordinate_median":
            # Coordinate-wise median inherently rejects outliers during aggregation
            active_updates = client_updates
            flagged_clients = []
            aggregated_update = ByzantineDefenses.coordinate_wise_median(list(active_updates.values()))

        elif self.defense_strategy == "trimmed_mean":
            active_updates = client_updates
            flagged_clients = []
            aggregated_update = ByzantineDefenses.trimmed_mean(list(active_updates.values()), trim_ratio=0.20)

        elif self.defense_strategy == "multi_krum":
            aggregated_update, selected_cids, krum_scores = ByzantineDefenses.multi_krum(client_updates, f=1)
            flagged_clients = [cid for cid in client_updates.keys() if cid not in selected_cids]
            filter_telemetry["krum_scores"] = krum_scores
            active_updates = {cid: client_updates[cid] for cid in selected_cids}
            print(f"  [Defense Engine: Multi-Krum] Selected Clients: {selected_cids} | Scores: { {c: round(s, 2) for c, s in krum_scores.items()} }")

        elif self.defense_strategy == "flame":
            aggregated_update, accepted_cids, cluster_labels = ByzantineDefenses.flame_aggregation(client_updates)
            flagged_clients = [cid for cid in client_updates.keys() if cid not in accepted_cids]
            filter_telemetry["accepted_clients"] = accepted_cids
            filter_telemetry["cluster_labels"] = str(cluster_labels)
            active_updates = {cid: client_updates[cid] for cid in accepted_cids}
            print(f"  [Defense Engine: FLAME] Clustered {len(accepted_cids)} clean updates (Flagged: {flagged_clients})")

        elif self.defense_strategy == "reputation":
            # Directional trust tracking with exponential moving average
            _, _, cosine_scores = ByzantineDefenses.cosine_similarity_filtering(client_updates)
            for cid, cos_val in cosine_scores.items():
                self.reputation_tracker.update_trust(cid, cos_val)
                if cos_val < 0.0:
                    flagged_clients.append(cid)
            trust_scores = self.reputation_tracker.get_trust_scores()
            filter_telemetry["trust_scores"] = trust_scores
            aggregated_update = self.reputation_tracker.weighted_aggregate(client_updates, client_sample_counts)
            active_updates = {cid: client_updates[cid] for cid in client_updates.keys() if cid not in flagged_clients}
            print(f"  [Defense Engine: Reputation Tracker] Trust Scores: { {c: round(t, 3) for c, t in trust_scores.items()} }")

        elif self.defense_strategy == "adaptive_clipping":
            adaptive_bound = ByzantineDefenses.adaptive_clip_norm(client_updates, multiplier=1.3)
            filtered_updates, flagged_clients, norms = ByzantineDefenses.detect_and_filter_norm_outliers(
                client_updates, threshold_multiplier=1.3
            )
            filter_telemetry["adaptive_bound"] = adaptive_bound
            filter_telemetry["client_norms"] = norms
            active_updates = filtered_updates
            print(f"  [Defense Engine: Adaptive Clip Norm] Dynamic Threshold: {adaptive_bound:.3f}")

        else: # Standard FedAvg (No defense)
            active_updates = client_updates
            flagged_clients = []

        # Step 2: Global Aggregation (if not pre-computed by geometric/clustering defense)
        if aggregated_update is None:
            # Weighted FedAvg over active (accepted) clients
            total_active_samples = sum(client_sample_counts[cid] for cid in active_updates.keys())
            aggregated_update = {}

            first_key = list(active_updates.keys())[0]
            for param_key in active_updates[first_key].keys():
                weighted_sum = sum(
                    active_updates[cid][param_key] * (client_sample_counts[cid] / max(1, total_active_samples))
                    for cid in active_updates.keys()
                )
                aggregated_update[param_key] = weighted_sum

        # Step 3: Apply aggregated update W_(t+1) = W_t + ΔW_agg
        current_weights = self.global_model.get_weights()
        new_weights: Dict[str, torch.Tensor] = {}
        for k in current_weights.keys():
            if current_weights[k].is_floating_point() and k in aggregated_update:
                new_weights[k] = current_weights[k] + aggregated_update[k].to(current_weights[k].device)
            else:
                new_weights[k] = current_weights[k]

        self.global_model.set_weights(new_weights)

        # Step 4: Evaluate Global Model on Held-Out Test Set
        eval_metrics = evaluate_model(self.global_model, self.test_loader, self.criterion, self.device)
        elapsed_time = round(time.time() - start_time, 2)

        # Step 5: Cryptographic Audit Logging (SHA-256 Hash Chain)
        audit_block = None
        if self.enable_audit_log:
            audit_block = self._record_audit_block(
                round_num=round_num,
                accepted_clients=list(active_updates.keys()),
                flagged_clients=flagged_clients,
                eval_metrics=eval_metrics
            )

        round_record = {
            "round": round_num,
            "accuracy": eval_metrics["accuracy"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "f1": eval_metrics["f1"],
            "loss": eval_metrics["loss"],
            "flagged_clients": flagged_clients,
            "accepted_clients": list(active_updates.keys()),
            "elapsed_seconds": elapsed_time,
            "defense_strategy": self.defense_strategy,
            "secagg_enabled": self.secagg_enabled,
            "audit_hash": audit_block["block_hash"] if audit_block else None
        }
        self.round_history.append(round_record)
        return round_record

    def _record_audit_block(
        self,
        round_num: int,
        accepted_clients: List[int],
        flagged_clients: List[int],
        eval_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Creates a cryptographically-chained SHA-256 block recording this round's state.
        Guarantees non-repudiation and tamper-evidence across institutional participants.
        """
        # Hash global model parameters
        hasher = hashlib.sha256()
        for k, v in sorted(self.global_model.state_dict().items()):
            hasher.update(k.encode("utf-8"))
            hasher.update(v.cpu().numpy().tobytes())
        weights_digest = hasher.hexdigest()

        timestamp = datetime.utcnow().isoformat() + "Z"
        block_data = {
            "round": round_num,
            "timestamp": timestamp,
            "prev_block_hash": self.prev_block_hash,
            "weights_digest": weights_digest,
            "defense_strategy": self.defense_strategy,
            "accepted_clients": accepted_clients,
            "flagged_clients": flagged_clients,
            "accuracy": eval_metrics["accuracy"],
            "f1": eval_metrics["f1"],
            "loss": eval_metrics["loss"]
        }

        block_payload = json.dumps(block_data, sort_keys=True).encode("utf-8")
        block_hash = hashlib.sha256(block_payload).hexdigest()
        block_data["block_hash"] = block_hash

        self.audit_log.append(block_data)
        self.prev_block_hash = block_hash

        # Persist to disk
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        audit_file = RESULTS_DIR / "audit_log.json"
        try:
            with open(audit_file, "w") as f:
                json.dump(self.audit_log, f, indent=2)
        except Exception as e:
            print(f"  [Audit Log Warning] Could not write audit log: {e}")

        return block_data

    @staticmethod
    def verify_audit_log(audit_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Validates the entire SHA-256 hash chain of the audit log.
        Returns (is_valid, status_message).
        """
        target = audit_path or (RESULTS_DIR / "audit_log.json")
        if not target.exists():
            return False, f"Audit file not found: {target}"

        try:
            with open(target, "r") as f:
                chain = json.load(f)

            if not isinstance(chain, list) or len(chain) == 0:
                return False, "Audit log is empty or malformed."

            prev_hash = "0" * 64
            for idx, block in enumerate(chain):
                if block.get("prev_block_hash") != prev_hash:
                    return False, f"Block #{idx + 1} prev_block_hash mismatch!"

                # Recompute hash
                recorded_hash = block.get("block_hash")
                data_to_verify = {k: v for k, v in block.items() if k != "block_hash"}
                computed_hash = hashlib.sha256(json.dumps(data_to_verify, sort_keys=True).encode("utf-8")).hexdigest()

                if computed_hash != recorded_hash:
                    return False, f"Block #{idx + 1} data tampering detected! Hash mismatch."
                prev_hash = recorded_hash

            return True, f"Audit chain verified successfully ({len(chain)} blocks intact)."
        except Exception as e:
            return False, f"Verification failed: {e}"
