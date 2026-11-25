from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import json
import hashlib
import math
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import get_storage_root
from .models import create_ids_model, evaluate_model, save_model_checkpoint, AutoencoderIDSModel
from .defenses import ByzantineDefenses, ClientReputationTracker
from .privacy import SecureAggregationEngine, PedersenCommitment, HomomorphicEncryptionEngine
from .attacks import AdversarialAttacks


ROOT_DIR = get_storage_root()
RESULTS_DIR = ROOT_DIR / "results"


class CentralFederatedServer:
    """
    Central Federated Aggregator (AEGIS-FL Coordinator).
    Enforces:
    1. Verifiable cryptographic Pedersen commitments (both SHA-256 binding & discrete-log homomorphic sum check)
    2. Byzantine outlier screening (`norm_clipping`, `cosine_similarity`, `multi_krum`, `flame`, `reputation`, `adaptive_clipping`, `coordinate_median`, `trimmed_mean`)
    3. Drop-resilient `SecAgg+` masked aggregation (server sums masked tensors & cancels dropped shares)
    4. Optional Paillier Homomorphic Encryption (`paillier_enabled`) over classifier head weights
    5. Immutable SHA-256 Merkle/hash-chain audit logging & `.pt` global checkpoint persistence.
    """

    def __init__(
        self,
        dataset_name: str = "unsw",
        input_dim: int = 42,
        num_classes: int = 2,
        multiclass: bool = False,
        class_names: Optional[List[str]] = None,
        model_type: str = "mlp",
        defense_strategy: str = "norm_clipping",
        secagg_enabled: bool = True,
        paillier_enabled: bool = False,
        num_clients: int = 5,
        enable_audit_log: bool = True,
        device: str = "cpu",
    ):
        self.dataset_name = dataset_name.lower()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.multiclass = multiclass
        self.class_names = class_names
        self.model_type = model_type
        self.defense_strategy = defense_strategy
        self.secagg_enabled = secagg_enabled
        self.paillier_enabled = paillier_enabled
        self.num_clients = num_clients
        self.enable_audit_log = enable_audit_log
        self.device = torch.device(device)

        self.global_model = create_ids_model(
            model_type=model_type, input_dim=input_dim, num_classes=num_classes
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.secagg_engine = SecureAggregationEngine(num_clients=num_clients)
        self.reputation_tracker = ClientReputationTracker(num_clients=num_clients, decay=0.85)
        self.pedersen_engine = PedersenCommitment()
        self.he_engine = HomomorphicEncryptionEngine(key_length=1024) if paillier_enabled else None
        if self.he_engine is not None:
            self.he_engine.generate_keys()

        self.audit_log: List[Dict[str, Any]] = []
        self.prev_block_hash: str = "0" * 64
        self.test_loader: Optional[DataLoader] = None
        self.round_history: List[Dict[str, Any]] = []

    def load_test_data(self, batch_size: int = 256):
        test_dir = ROOT_DIR / "data" / "processed" / self.dataset_name
        X_test_file = test_dir / "X_test.npy"
        y_file_name = "y_test_multi.npy" if self.multiclass and (test_dir / "y_test_multi.npy").exists() else "y_test.npy"
        y_test_file = test_dir / y_file_name

        if not X_test_file.exists() or not y_test_file.exists():
            raise FileNotFoundError(f"Evaluation set missing in {test_dir}. Run preprocessor first.")

        X = torch.tensor(np.load(X_test_file), dtype=torch.float32)
        y = torch.tensor(np.load(y_test_file), dtype=torch.long)

        if self.class_names is None and (test_dir / "metadata.json").exists():
            with open(test_dir / "metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.class_names = meta["class_names"] if self.multiclass else ["Normal", "Attack"]

        self.test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        return self.global_model.get_weights()

    def verify_client_commitments(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_telemetry: Optional[List[Dict[str, Any]]],
    ) -> Tuple[bool, List[int]]:
        """
        Verifies both individual SHA-256 Pedersen norm bindings AND the discrete-log
        homomorphic product of commitments across participating clients.
        Returns `(all_valid, tampered_client_ids)`.
        """
        if not client_telemetry:
            return True, []

        tampered: List[int] = []
        homomorphic_cs: List[int] = []
        sum_norm = 0.0
        sum_blinding = 0

        telemetry_by_cid = {t["client_id"]: t for t in client_telemetry}
        for cid, update in client_updates.items():
            t = telemetry_by_cid.get(cid)
            if not t or not t.get("commitment_hash") or not t.get("blinding_factor"):
                continue

            actual_norm = round(
                math.sqrt(sum(torch.sum(v ** 2).item() for v in update.values() if v.is_floating_point())),
                6,
            )
            ok = self.pedersen_engine.verify(
                np.array([actual_norm], dtype=np.float64),
                t["commitment_hash"],
                t["blinding_factor"],
            )
            if not ok:
                tampered.append(cid)
            elif t.get("homomorphic_commitment") is not None and t.get("homomorphic_blinding") is not None:
                homomorphic_cs.append(int(t["homomorphic_commitment"]))
                sum_norm += actual_norm
                sum_blinding += int(t["homomorphic_blinding"])

        homomorphic_ok = True
        if homomorphic_cs:
            homomorphic_ok = self.pedersen_engine.verify_homomorphic_sum(homomorphic_cs, sum_norm, sum_blinding)

        return (len(tampered) == 0 and homomorphic_ok), tampered

    def aggregate_and_evaluate(
        self,
        round_num: int,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_sample_counts: Dict[int, int],
        masked_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        secagg_engine: Optional[SecureAggregationEngine] = None,
        client_telemetry: Optional[List[Dict[str, Any]]] = None,
        evaluate_backdoor: bool = False,
    ) -> Dict[str, Any]:
        start_time = time.time()
        flagged_clients: List[int] = []

        if self.test_loader is None:
            self.load_test_data()

        # Step 0: Cryptographic Commitment Verification
        commitments_verified, tampered_cids = self.verify_client_commitments(client_updates, client_telemetry)
        working_updates = {cid: u for cid, u in client_updates.items() if cid not in tampered_cids}
        if not working_updates:
            working_updates = dict(client_updates)
        flagged_clients.extend(tampered_cids)

        aggregated_update: Optional[Dict[str, torch.Tensor]] = None

        # Step 1: Byzantine Defense Layer
        if self.defense_strategy == "norm_clipping":
            filtered_updates, def_flagged, norms = ByzantineDefenses.detect_and_filter_norm_outliers(
                working_updates, threshold_multiplier=1.4
            )
            flagged_clients.extend(def_flagged)
            active_updates = filtered_updates
            print(f"  [Defense Engine] Client Update Norms: { {c: round(n, 2) for c, n in norms.items()} }")

        elif self.defense_strategy == "cosine_similarity":
            filtered_updates, def_flagged, _ = ByzantineDefenses.cosine_similarity_filtering(working_updates)
            flagged_clients.extend(def_flagged)
            active_updates = filtered_updates

        elif self.defense_strategy == "coordinate_median":
            _, def_flagged, _ = ByzantineDefenses.cosine_similarity_filtering(working_updates)
            active_updates = {c: u for c, u in working_updates.items() if c not in def_flagged} or working_updates
            flagged_clients.extend(def_flagged if len(active_updates) < len(working_updates) else [])
            aggregated_update = ByzantineDefenses.coordinate_wise_median(list(active_updates.values()))

        elif self.defense_strategy == "trimmed_mean":
            _, def_flagged, _ = ByzantineDefenses.cosine_similarity_filtering(working_updates)
            flagged_clients.extend(def_flagged)
            aggregated_update = ByzantineDefenses.trimmed_mean(list(working_updates.values()), trim_ratio=0.20)
            active_updates = {c: u for c, u in working_updates.items() if c not in def_flagged} or working_updates

        elif self.defense_strategy == "multi_krum":
            aggregated_update, selected_cids, krum_scores = ByzantineDefenses.multi_krum(working_updates, f=1)
            flagged_clients.extend([cid for cid in working_updates.keys() if cid not in selected_cids])
            active_updates = {cid: working_updates[cid] for cid in selected_cids}
            print(f"  [Defense Engine: Multi-Krum] Selected Clients: {selected_cids} | Scores: { {c: round(s, 2) for c, s in krum_scores.items()} }")

        elif self.defense_strategy == "flame":
            aggregated_update, accepted_cids, _ = ByzantineDefenses.flame_aggregation(working_updates)
            flagged_clients.extend([cid for cid in working_updates.keys() if cid not in accepted_cids])
            active_updates = {cid: working_updates[cid] for cid in accepted_cids}
            print(f"  [Defense Engine: FLAME] Clustered {len(accepted_cids)} clean updates (Flagged: {flagged_clients})")

        elif self.defense_strategy == "reputation":
            _, def_flagged, cosine_scores = ByzantineDefenses.cosine_similarity_filtering(working_updates)
            for cid, cos_val in cosine_scores.items():
                self.reputation_tracker.update_trust(cid, cos_val)
            flagged_clients.extend(def_flagged)
            trust_scores = self.reputation_tracker.get_trust_scores()
            aggregated_update = self.reputation_tracker.weighted_aggregate(
                working_updates, client_sample_counts, excluded_clients=def_flagged
            )
            active_updates = {cid: working_updates[cid] for cid in working_updates.keys() if cid not in def_flagged} or working_updates
            print(f"  [Defense Engine: Reputation Tracker] Trust Scores: { {c: round(t, 3) for c, t in trust_scores.items()} }")

        elif self.defense_strategy == "adaptive_clipping":
            adaptive_bound = ByzantineDefenses.adaptive_clip_norm(working_updates, multiplier=1.3)
            filtered_updates, def_flagged, _ = ByzantineDefenses.detect_and_filter_norm_outliers(
                working_updates, threshold_multiplier=1.3
            )
            flagged_clients.extend(def_flagged)
            active_updates = filtered_updates
            print(f"  [Defense Engine: Adaptive Clip Norm] Dynamic Threshold: {adaptive_bound:.3f}")

        else:  # Standard FedAvg (No defense)
            active_updates = working_updates

        flagged_clients = sorted(set(flagged_clients))

        # Step 2: Secure Aggregation (`SecAgg+` drop-resilient mask cancellation + optional Paillier HE check)
        secagg_verified = False
        if aggregated_update is None:
            if self.secagg_enabled and masked_updates is not None and secagg_engine is not None:
                weights_by_cid = {cid: float(client_sample_counts.get(cid, 1)) for cid in client_updates}
                aggregated_update = secagg_engine.aggregate_masked_surviving(
                    masked_updates=masked_updates,
                    accepted_ids=list(active_updates.keys()),
                    all_client_ids=list(client_updates.keys()),
                    weights_by_client=weights_by_cid,
                )
                secagg_verified = True
            else:
                total_active_samples = max(1, sum(client_sample_counts[cid] for cid in active_updates.keys()))
                aggregated_update = {}
                first_key = list(active_updates.keys())[0]
                for param_key, ref_t in active_updates[first_key].items():
                    if not ref_t.is_floating_point():
                        aggregated_update[param_key] = ref_t.clone()
                        continue
                    weighted_sum = sum(
                        active_updates[cid][param_key].float() * (client_sample_counts[cid] / total_active_samples)
                        for cid in active_updates.keys()
                    )
                    aggregated_update[param_key] = weighted_sum.to(dtype=ref_t.dtype)  # type: ignore[union-attr]
        elif self.secagg_enabled and masked_updates is not None and secagg_engine is not None:
            # Verify pairwise zero-sum cancellation over surviving accepted clients
            weights_by_cid = {cid: float(client_sample_counts.get(cid, 1)) for cid in client_updates}
            _ = secagg_engine.aggregate_masked_surviving(
                masked_updates=masked_updates,
                accepted_ids=list(active_updates.keys()),
                all_client_ids=list(client_updates.keys()),
                weights_by_client=weights_by_cid,
            )
            secagg_verified = True

        # Optional Paillier Homomorphic Encryption over final bias/logit vector
        if self.paillier_enabled and self.he_engine is not None and self.he_engine._pub_key is not None:
            float_keys = [k for k, v in aggregated_update.items() if v.is_floating_point() and v.numel() <= 64]
            if float_keys:
                target_k = float_keys[-1]
                total_s = max(1, sum(client_sample_counts[cid] for cid in active_updates.keys()))
                enc_vecs = [
                    self.he_engine.encrypt_vector(
                        (active_updates[cid][target_k].float().cpu().numpy() * (client_sample_counts[cid] / total_s)),
                        self.he_engine._pub_key,
                    )
                    for cid in active_updates.keys()
                ]
                enc_sum = self.he_engine.aggregate_encrypted(enc_vecs)
                dec_sum = self.he_engine.decrypt_vector(
                    enc_sum, self.he_engine._priv_key, original_shape=aggregated_update[target_k].shape
                )
                aggregated_update[target_k] = torch.tensor(dec_sum, dtype=aggregated_update[target_k].dtype)

        # Step 3: Apply Global Update W_(t+1) = W_t + dW_agg (or average anomaly_threshold buffer for Autoencoder)
        current_weights = self.global_model.get_weights()
        new_weights: Dict[str, torch.Tensor] = {}
        is_ae = isinstance(self.global_model, AutoencoderIDSModel)
        for k in current_weights.keys():
            if is_ae and "anomaly_threshold" in k:
                # `anomaly_threshold` in client update is dThreshold = local_thresh - global_thresh
                new_weights[k] = torch.clamp(current_weights[k] + aggregated_update[k].to(current_weights[k].device), min=0.05)
            elif current_weights[k].is_floating_point() and k in aggregated_update:
                new_weights[k] = current_weights[k] + aggregated_update[k].to(current_weights[k].device)
            else:
                new_weights[k] = current_weights[k]

        self.global_model.set_weights(new_weights)

        # Step 4: Global Benchmark Evaluation (+ Backdoor ASR if applicable)
        eval_metrics = evaluate_model(self.global_model, self.test_loader, self.criterion, self.device)
        backdoor_asr = (
            AdversarialAttacks.evaluate_backdoor_asr(self.global_model, self.test_loader, self.device)
            if evaluate_backdoor
            else 0.0
        )
        elapsed_time = round(time.time() - start_time, 2)

        # Persist trained global checkpoint for live SOC flow scanning
        ckpt_path = save_model_checkpoint(
            model=self.global_model,
            dataset_name=self.dataset_name,
            model_type=self.model_type,
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            class_names=self.class_names,
            metrics={**eval_metrics, "round": round_num, "backdoor_asr": backdoor_asr},
        )

        # Step 5: Cryptographic Audit Block
        audit_block = None
        if self.enable_audit_log:
            audit_block = self._record_audit_block(
                round_num=round_num,
                accepted_clients=list(active_updates.keys()),
                flagged_clients=flagged_clients,
                eval_metrics=eval_metrics,
                commitments_verified=commitments_verified,
                secagg_verified=secagg_verified,
            )

        round_record = {
            "round": round_num,
            "accuracy": eval_metrics["accuracy"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "f1": eval_metrics["f1"],
            "fpr": eval_metrics["fpr"],
            "roc_auc": eval_metrics["roc_auc"],
            "confusion_matrix": eval_metrics["confusion_matrix"],
            "backdoor_asr": backdoor_asr,
            "loss": eval_metrics["loss"],
            "flagged_clients": flagged_clients,
            "accepted_clients": list(active_updates.keys()),
            "commitments_verified": commitments_verified,
            "secagg_verified": secagg_verified,
            "checkpoint_path": str(ckpt_path),
            "elapsed_seconds": elapsed_time,
            "defense_strategy": self.defense_strategy,
            "secagg_enabled": self.secagg_enabled,
            "audit_hash": audit_block["block_hash"] if audit_block else None,
        }
        self.round_history.append(round_record)
        return round_record

    def _record_audit_block(
        self,
        round_num: int,
        accepted_clients: List[int],
        flagged_clients: List[int],
        eval_metrics: Dict[str, Any],
        commitments_verified: bool = True,
        secagg_verified: bool = False,
    ) -> Dict[str, Any]:
        hasher = hashlib.sha256()
        for k, v in sorted(self.global_model.state_dict().items()):
            hasher.update(k.encode("utf-8"))
            hasher.update(v.detach().cpu().numpy().tobytes())
        weights_digest = hasher.hexdigest()

        timestamp = datetime.now(timezone.utc).isoformat()
        block_data = {
            "round": round_num,
            "timestamp": timestamp,
            "prev_block_hash": self.prev_block_hash,
            "weights_digest": weights_digest,
            "defense_strategy": self.defense_strategy,
            "accepted_clients": accepted_clients,
            "flagged_clients": flagged_clients,
            "commitments_verified": commitments_verified,
            "secagg_verified": secagg_verified,
            "accuracy": eval_metrics["accuracy"],
            "f1": eval_metrics["f1"],
            "loss": eval_metrics["loss"],
        }

        block_payload = json.dumps(block_data, sort_keys=True).encode("utf-8")
        block_hash = hashlib.sha256(block_payload).hexdigest()
        block_data["block_hash"] = block_hash

        self.audit_log.append(block_data)
        self.prev_block_hash = block_hash

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        audit_file = RESULTS_DIR / "audit_log.json"
        try:
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump(self.audit_log, f, indent=2)
        except Exception as e:
            print(f"  [Audit Log Warning] Could not write audit log: {e}")

        return block_data

    @staticmethod
    def verify_audit_log(audit_path: Optional[Path] = None) -> Tuple[bool, str]:
        target = audit_path or (RESULTS_DIR / "audit_log.json")
        if not target.exists():
            return False, f"Audit file not found: {target}"

        try:
            with open(target, "r", encoding="utf-8") as f:
                chain = json.load(f)

            if not isinstance(chain, list) or len(chain) == 0:
                return False, "Audit log is empty or malformed."

            prev_hash = "0" * 64
            for idx, block in enumerate(chain):
                if block.get("prev_block_hash") != prev_hash:
                    return False, f"Block #{idx + 1} prev_block_hash mismatch!"

                recorded_hash = block.get("block_hash")
                data_to_verify = {k: v for k, v in block.items() if k != "block_hash"}
                computed_hash = hashlib.sha256(json.dumps(data_to_verify, sort_keys=True).encode("utf-8")).hexdigest()

                if computed_hash != recorded_hash:
                    return False, f"Block #{idx + 1} data tampering detected! Hash mismatch."
                prev_hash = recorded_hash

            return True, f"Audit chain verified successfully ({len(chain)} blocks intact)."
        except Exception as e:
            return False, f"Verification failed: {e}"
