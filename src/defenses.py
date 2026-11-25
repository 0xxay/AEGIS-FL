import math
from typing import Dict, List, Tuple, Any
import numpy as np
import torch


def _extract_head_and_full_vectors(
    client_updates: Dict[int, Dict[str, torch.Tensor]]
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Extracts both the full flattened parameter vector AND the final classifier-head vector.
    Why this matters: Data poisoning (label flipping) and stealth norm-bounded attacks keep
    hidden-layer L2 norms nearly identical to honest clients, while strongly inverting or
    skewing the output classifier head (`head.*`, `network.11.*`, `decoder.6.*`).
    """
    full_vecs: Dict[int, torch.Tensor] = {}
    head_vecs: Dict[int, torch.Tensor] = {}

    for cid, update in client_updates.items():
        float_items = [(k, t.flatten().cpu().float()) for k, t in update.items() if t.is_floating_point() and "threshold" not in k]
        full_vecs[cid] = torch.cat([t for _, t in float_items])
        # Output layer is the last floating-point weight/bias pair in state_dict order
        tail_items = float_items[-2:] if len(float_items) >= 2 else float_items
        head_vecs[cid] = torch.cat([t for _, t in tail_items])

    return full_vecs, head_vecs


def _head_cosine_outliers(head_vecs: Dict[int, torch.Tensor], threshold: float = 0.0) -> Tuple[List[int], Dict[int, float]]:
    """Detects clients whose output-layer gradient opposes the majority median consensus."""
    cids = list(head_vecs.keys())
    stacked = torch.stack([head_vecs[c] for c in cids], dim=0)
    consensus, _ = torch.median(stacked, dim=0)
    ref_norm = torch.norm(consensus) + 1e-8

    scores: Dict[int, float] = {}
    flagged: List[int] = []
    for cid in cids:
        v = head_vecs[cid]
        cos_val = float((torch.dot(v, consensus) / ((torch.norm(v) + 1e-8) * ref_norm)).item())
        scores[cid] = round(cos_val, 4)
        if cos_val < threshold:
            flagged.append(cid)
    return flagged, scores


class ByzantineDefenses:
    """
    Byzantine-Robust Defense Suite for Federated Aggregation:
    1. Norm-Bounding + Classifier-Head Sign Guard (catches both scaling & label-flipping)
    2. Coordinate-Wise Median
    3. Trimmed Mean
    4. Directional Cosine Similarity Filtering (FLTrust-inspired head+full alignment)
    5. Multi-Krum (Geometric consensus over normalized head+full representation)
    6. FLAME Defense (DBSCAN cosine clustering + cluster-median clipping + server DP)
    7. Adaptive Clip Norm
    """

    @staticmethod
    def detect_and_filter_norm_outliers(
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        threshold_multiplier: float = 2.0,
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], List[int], Dict[int, float]]:
        full_vecs, head_vecs = _extract_head_and_full_vectors(client_updates)
        client_norms: Dict[int, float] = {
            cid: float(torch.norm(vec).item()) for cid, vec in full_vecs.items()
        }

        median_norm = float(np.median(list(client_norms.values())))
        threshold = max(0.5, median_norm * threshold_multiplier)

        # Also check classifier-head directional inversion so stealth label-flipping / norm-bounded attacks cannot slip past
        head_flagged, _ = _head_cosine_outliers(head_vecs, threshold=-0.05)

        flagged_clients: List[int] = []
        filtered_updates: Dict[int, Dict[str, torch.Tensor]] = {}

        for cid, norm_val in client_norms.items():
            if norm_val > threshold or (len(client_updates) >= 4 and cid in head_flagged):
                flagged_clients.append(cid)
            else:
                filtered_updates[cid] = client_updates[cid]

        if not filtered_updates:
            sorted_cids = sorted(client_norms.keys(), key=lambda c: client_norms[c])
            for cid in sorted_cids[: max(1, len(sorted_cids) // 2)]:
                filtered_updates[cid] = client_updates[cid]
                if cid in flagged_clients:
                    flagged_clients.remove(cid)

        return filtered_updates, flagged_clients, client_norms

    @staticmethod
    def coordinate_wise_median(updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        aggregated: Dict[str, torch.Tensor] = {}
        for k in updates[0].keys():
            if not updates[0][k].is_floating_point():
                aggregated[k] = updates[0][k].clone()
                continue
            stacked = torch.stack([u[k].float() for u in updates], dim=0)
            median_val, _ = torch.median(stacked, dim=0)
            aggregated[k] = median_val.to(dtype=updates[0][k].dtype)
        return aggregated

    @staticmethod
    def trimmed_mean(updates: List[Dict[str, torch.Tensor]], trim_ratio: float = 0.20) -> Dict[str, torch.Tensor]:
        aggregated: Dict[str, torch.Tensor] = {}
        num_clients = len(updates)
        k_trim = int(num_clients * trim_ratio)

        for key in updates[0].keys():
            if not updates[0][key].is_floating_point():
                aggregated[key] = updates[0][key].clone()
                continue
            stacked = torch.stack([u[key].float() for u in updates], dim=0)
            sorted_tensors, _ = torch.sort(stacked, dim=0)
            if k_trim > 0 and (num_clients - 2 * k_trim) > 0:
                trimmed = sorted_tensors[k_trim : num_clients - k_trim]
            else:
                trimmed = sorted_tensors
            aggregated[key] = torch.mean(trimmed, dim=0).to(dtype=updates[0][key].dtype)
        return aggregated

    @staticmethod
    def cosine_similarity_filtering(
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        anchor_update: Dict[str, torch.Tensor] = None,
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], List[int], Dict[int, float]]:
        full_vecs, head_vecs = _extract_head_and_full_vectors(client_updates)

        if anchor_update is None:
            consensus_full, _ = torch.median(torch.stack(list(full_vecs.values()), dim=0), dim=0)
            consensus_head, _ = torch.median(torch.stack(list(head_vecs.values()), dim=0), dim=0)
        else:
            a_full, a_head = _extract_head_and_full_vectors({0: anchor_update})
            consensus_full, consensus_head = a_full[0], a_head[0]

        full_ref_norm = torch.norm(consensus_full) + 1e-8
        head_ref_norm = torch.norm(consensus_head) + 1e-8

        cosine_scores: Dict[int, float] = {}
        flagged_clients: List[int] = []
        filtered_updates: Dict[int, Dict[str, torch.Tensor]] = {}

        # Also guard against extreme norm amplification along positive angles
        norms = {cid: float(torch.norm(v).item()) for cid, v in full_vecs.items()}
        med_norm = float(np.median(list(norms.values())))

        for cid in full_vecs:
            cos_full = float((torch.dot(full_vecs[cid], consensus_full) / ((torch.norm(full_vecs[cid]) + 1e-8) * full_ref_norm)).item())
            cos_head = float((torch.dot(head_vecs[cid], consensus_head) / ((torch.norm(head_vecs[cid]) + 1e-8) * head_ref_norm)).item())
            combined_cos = min(cos_full, cos_head)
            cosine_scores[cid] = round(combined_cos, 4)

            if combined_cos < 0.0 or norms[cid] > max(1.0, med_norm * 2.5):
                flagged_clients.append(cid)
            else:
                filtered_updates[cid] = client_updates[cid]

        if not filtered_updates:
            best_cid = max(cosine_scores.keys(), key=lambda c: cosine_scores[c])
            filtered_updates[best_cid] = client_updates[best_cid]
            if best_cid in flagged_clients:
                flagged_clients.remove(best_cid)

        return filtered_updates, flagged_clients, cosine_scores

    @staticmethod
    def multi_krum(
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        f: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], List[int], Dict[int, float]]:
        client_ids = list(client_updates.keys())
        n = len(client_ids)
        if f >= n // 2:
            f = max(0, n // 2 - 1)

        full_vecs, head_vecs = _extract_head_and_full_vectors(client_updates)
        # Concatenate full vector and scaled head vector so both norm scaling and label-flipping stand out geometrically
        joint_vecs: Dict[int, torch.Tensor] = {}
        for cid in client_ids:
            hv = head_vecs[cid]
            fv = full_vecs[cid]
            weight = math.sqrt(fv.numel() / max(1, hv.numel()))
            joint_vecs[cid] = torch.cat([fv, hv * weight])

        num_neighbors = max(1, n - f - 2)
        scores: Dict[int, float] = {}
        for i in client_ids:
            dists = sorted(
                torch.sum((joint_vecs[i] - joint_vecs[j]) ** 2).item()
                for j in client_ids
                if i != j
            )
            scores[i] = float(sum(dists[:num_neighbors]))

        num_select = max(1, n - f)
        selected_client_ids = sorted(scores.keys(), key=lambda c: scores[c])[:num_select]

        aggregated_update: Dict[str, torch.Tensor] = {}
        for key in client_updates[selected_client_ids[0]].keys():
            ref_t = client_updates[selected_client_ids[0]][key]
            if not ref_t.is_floating_point():
                aggregated_update[key] = ref_t.clone()
                continue
            stacked = torch.stack([client_updates[cid][key].float() for cid in selected_client_ids], dim=0)
            aggregated_update[key] = torch.mean(stacked, dim=0).to(dtype=ref_t.dtype)

        return aggregated_update, selected_client_ids, scores

    @staticmethod
    def flame_aggregation(
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        noise_multiplier: float = 0.001,
    ) -> Tuple[Dict[str, torch.Tensor], List[int], Any]:
        client_ids = list(client_updates.keys())
        full_vecs, head_vecs = _extract_head_and_full_vectors(client_updates)

        flat_np: Dict[int, np.ndarray] = {cid: full_vecs[cid].numpy() for cid in client_ids}
        head_np: Dict[int, np.ndarray] = {cid: head_vecs[cid].numpy() for cid in client_ids}

        try:
            from sklearn.cluster import DBSCAN

            # Cosine distance over classifier-head + full direction catches backdoors, label flips, and sign flips
            joint_np = np.stack(
                [
                    np.concatenate([
                        flat_np[cid] / (np.linalg.norm(flat_np[cid]) + 1e-8),
                        head_np[cid] / (np.linalg.norm(head_np[cid]) + 1e-8),
                    ])
                    for cid in client_ids
                ],
                axis=0,
            )
            norms_j = np.linalg.norm(joint_np, axis=1, keepdims=True) + 1e-8
            normed = joint_np / norms_j
            cos_dist_matrix = np.clip(1.0 - (normed @ normed.T), 0.0, 2.0)

            triu_dists = cos_dist_matrix[np.triu_indices(len(client_ids), k=1)]
            p60 = float(np.percentile(triu_dists, 60)) if len(triu_dists) > 0 else 0.75
            adaptive_eps = min(0.85, max(0.35, p60))

            db = DBSCAN(eps=adaptive_eps, min_samples=max(2, len(client_ids) // 2), metric="precomputed")
            cluster_labels = db.fit_predict(cos_dist_matrix)

            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            valid_mask = unique_labels != -1

            if not np.any(valid_mask) or np.max(counts[valid_mask]) < max(1, len(client_ids) // 2):
                median_vec = np.median(normed, axis=0)
                med_norm = np.linalg.norm(median_vec) + 1e-8
                cos_with_med = (normed @ median_vec) / med_norm
                accepted_indices = [i for i, c in enumerate(cos_with_med) if c >= 0.15]
                if not accepted_indices:
                    accepted_indices = [int(np.argmax(cos_with_med))]
                accepted_client_ids = [client_ids[i] for i in accepted_indices]
                cluster_labels = np.array([0 if i in accepted_indices else -1 for i in range(len(client_ids))])
            else:
                largest_label = unique_labels[valid_mask][np.argmax(counts[valid_mask])]
                accepted_indices = [i for i, lbl in enumerate(cluster_labels) if lbl == largest_label]
                accepted_client_ids = [client_ids[i] for i in accepted_indices]

            all_norms = [float(np.linalg.norm(flat_np[cid])) for cid in client_ids]
            median_norm = max(1e-8, float(np.median(all_norms)))

            # Also exclude extreme magnitude spikes that happened to align directionally
            accepted_client_ids = [
                cid for cid in accepted_client_ids
                if float(np.linalg.norm(flat_np[cid])) <= median_norm * 2.5
            ] or [min(accepted_client_ids, key=lambda c: float(np.linalg.norm(flat_np[c])))]

            clipped_updates: List[Dict[str, torch.Tensor]] = []
            for cid in accepted_client_ids:
                v_norm = float(np.linalg.norm(flat_np[cid])) + 1e-8
                scale = min(1.0, median_norm / v_norm)
                clipped_updates.append({
                    k: (t.float() * scale if t.is_floating_point() else t.clone())
                    for k, t in client_updates[cid].items()
                })

            aggregated_update: Dict[str, torch.Tensor] = {}
            ref_update = client_updates[accepted_client_ids[0]]
            noise_std = noise_multiplier * median_norm
            for key, ref_tensor in ref_update.items():
                if not ref_tensor.is_floating_point():
                    aggregated_update[key] = ref_tensor.clone()
                    continue
                avg_t = torch.mean(torch.stack([u[key] for u in clipped_updates], dim=0), dim=0)
                if "threshold" not in key and noise_std > 0:
                    avg_t = avg_t + torch.randn_like(avg_t) * noise_std
                aggregated_update[key] = avg_t.to(dtype=ref_tensor.dtype)

            return aggregated_update, accepted_client_ids, cluster_labels

        except Exception:
            all_updates_list = [client_updates[cid] for cid in client_ids]
            fallback_agg = ByzantineDefenses.trimmed_mean(all_updates_list)
            return fallback_agg, client_ids, None

    @staticmethod
    def adaptive_clip_norm(
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        multiplier: float = 1.0,
    ) -> float:
        norms = [
            math.sqrt(sum(torch.sum(t ** 2).item() for t in update.values() if t.is_floating_point()))
            for update in client_updates.values()
        ]
        return float(np.median(norms)) * multiplier


class ClientReputationTracker:
    """Tracks per-client trust scores via EMA and computes reputation-weighted aggregation."""

    def __init__(self, num_clients: int, decay: float = 0.9, initial_trust: float = 1.0) -> None:
        self.decay = decay
        self.initial_trust = initial_trust
        self.trust_scores: Dict[int, float] = {i: initial_trust for i in range(num_clients)}

    def update_trust(self, client_id: int, cosine_sim_with_global: float) -> None:
        if client_id not in self.trust_scores:
            self.trust_scores[client_id] = self.initial_trust
        current = self.trust_scores[client_id]
        if cosine_sim_with_global < 0.0:
            # Immediate strong penalty for adversarial negative gradient alignment
            updated = current * 0.25
        else:
            updated = self.decay * current + (1.0 - self.decay) * cosine_sim_with_global
        self.trust_scores[client_id] = float(np.clip(updated, 0.05, 1.0))

    def get_trust_scores(self) -> Dict[int, float]:
        return dict(self.trust_scores)

    def weighted_aggregate(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_sample_counts: Dict[int, int],
        excluded_clients: List[int] = None,
    ) -> Dict[str, torch.Tensor]:
        excluded = set(excluded_clients or [])
        active_cids = [cid for cid in client_updates if cid not in excluded] or list(client_updates.keys())

        raw_weights: Dict[int, float] = {}
        for cid in active_cids:
            tau = self.trust_scores.get(cid, self.initial_trust)
            n_samples = client_sample_counts.get(cid, 1)
            raw_weights[cid] = tau * n_samples

        total_weight = sum(raw_weights.values()) + 1e-8
        norm_weights = {cid: w / total_weight for cid, w in raw_weights.items()}

        aggregated: Dict[str, torch.Tensor] = {}
        ref = client_updates[active_cids[0]]
        for key, ref_t in ref.items():
            if not ref_t.is_floating_point():
                aggregated[key] = ref_t.clone()
                continue
            weighted_sum = sum(norm_weights[cid] * client_updates[cid][key].float() for cid in active_cids)
            aggregated[key] = weighted_sum.to(dtype=ref_t.dtype)  # type: ignore[union-attr]

        return aggregated

    def penalize(self, client_id: int, factor: float = 0.5) -> None:
        if client_id not in self.trust_scores:
            self.trust_scores[client_id] = self.initial_trust
        self.trust_scores[client_id] = float(np.clip(self.trust_scores[client_id] * factor, 0.05, 1.0))

    def reset(self, client_id: int) -> None:
        self.trust_scores[client_id] = self.initial_trust
