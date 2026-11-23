import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch


class ByzantineDefenses:
    """
    Byzantine-Robust Defense Suite for Federated Aggregation:
    1. Norm-Bounding & Outlier Detection
    2. Coordinate-Wise Median
    3. Trimmed Mean
    4. Directional Cosine Similarity Filtering (FLTrust / FoolsGold)
    5. Multi-Krum
    6. FLAME Defense
    7. Adaptive Clip Norm
    """

    @staticmethod
    def detect_and_filter_norm_outliers(
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        threshold_multiplier: float = 2.0
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], List[int], Dict[int, float]]:
        """
        Computes L2 norm of each client's update.
        Rejects clients whose norm exceeds threshold_multiplier * median_norm.
        Returns filtered updates, list of flagged malicious client IDs, and all client norms.
        """
        client_norms: Dict[int, float] = {}

        for cid, update in client_updates.items():
            total_sq = sum(torch.sum(t ** 2).item() for t in update.values() if t.is_floating_point())
            client_norms[cid] = math.sqrt(total_sq)

        median_norm = float(np.median(list(client_norms.values())))
        threshold = max(0.5, median_norm * threshold_multiplier)

        flagged_clients: List[int] = []
        filtered_updates: Dict[int, Dict[str, torch.Tensor]] = {}

        for cid, norm_val in client_norms.items():
            if norm_val > threshold:
                flagged_clients.append(cid)
            else:
                filtered_updates[cid] = client_updates[cid]

        # Failsafe: if all clients were flagged, keep the lowest norm clients
        if not filtered_updates:
            sorted_cids = sorted(client_norms.keys(), key=lambda c: client_norms[c])
            for cid in sorted_cids[:max(1, len(sorted_cids) // 2)]:
                filtered_updates[cid] = client_updates[cid]
                if cid in flagged_clients:
                    flagged_clients.remove(cid)

        return filtered_updates, flagged_clients, client_norms

    @staticmethod
    def coordinate_wise_median(
        updates: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Computes element-wise median across all participating clients.
        Immune to extreme outlier values injected by Byzantine attackers.
        """
        aggregated: Dict[str, torch.Tensor] = {}
        keys = updates[0].keys()

        for k in keys:
            stacked = torch.stack([u[k] for u in updates], dim=0)
            median_val, _ = torch.median(stacked, dim=0)
            aggregated[k] = median_val

        return aggregated

    @staticmethod
    def trimmed_mean(
        updates: List[Dict[str, torch.Tensor]],
        trim_ratio: float = 0.20
    ) -> Dict[str, torch.Tensor]:
        """
        Discards the top and bottom trim_ratio fraction of extreme client values
        per coordinate and averages the remaining middle.
        """
        aggregated: Dict[str, torch.Tensor] = {}
        keys = updates[0].keys()
        num_clients = len(updates)
        k_trim = int(num_clients * trim_ratio)

        for key in keys:
            stacked = torch.stack([u[key] for u in updates], dim=0)
            sorted_tensors, _ = torch.sort(stacked, dim=0)

            if k_trim > 0 and (num_clients - 2 * k_trim) > 0:
                trimmed = sorted_tensors[k_trim : num_clients - k_trim]
            else:
                trimmed = sorted_tensors

            aggregated[key] = torch.mean(trimmed.float(), dim=0)

        return aggregated

    @staticmethod
    def cosine_similarity_filtering(
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        anchor_update: Dict[str, torch.Tensor] = None
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], List[int], Dict[int, float]]:
        """
        Directional alignment check (FLTrust-style).
        Calculates cosine similarity of each update against the median consensus vector or anchor.
        Updates pointing in opposing directions (cosine < 0) are flagged and removed.
        """
        # Flatten each client update into a 1D vector (floating point parameters only)
        flat_vectors: Dict[int, torch.Tensor] = {}
        for cid, update in client_updates.items():
            tensors = [t.flatten().cpu().float() for t in update.values() if t.is_floating_point()]
            flat_vectors[cid] = torch.cat(tensors)

        # Compute consensus reference vector (median of flattened vectors)
        if anchor_update is None:
            all_flats = torch.stack(list(flat_vectors.values()), dim=0)
            consensus_ref, _ = torch.median(all_flats, dim=0)
        else:
            anchor_tensors = [t.flatten().cpu().float() for t in anchor_update.values() if t.is_floating_point()]
            consensus_ref = torch.cat(anchor_tensors)

        ref_norm = torch.norm(consensus_ref) + 1e-8
        cosine_scores: Dict[int, float] = {}
        flagged_clients: List[int] = []
        filtered_updates: Dict[int, Dict[str, torch.Tensor]] = {}

        for cid, vec in flat_vectors.items():
            vec_norm = torch.norm(vec) + 1e-8
            cos_sim = torch.dot(vec, consensus_ref) / (vec_norm * ref_norm)
            cos_score = float(cos_sim.item())
            cosine_scores[cid] = round(cos_score, 4)

            # Negative cosine similarity indicates hostile direction (sign flipping)
            if cos_score < 0.0:
                flagged_clients.append(cid)
            else:
                filtered_updates[cid] = client_updates[cid]

        if not filtered_updates:
            # Fallback to highest cosine similarity client
            best_cid = max(cosine_scores.keys(), key=lambda c: cosine_scores[c])
            filtered_updates[best_cid] = client_updates[best_cid]
            if best_cid in flagged_clients:
                flagged_clients.remove(best_cid)

        return filtered_updates, flagged_clients, cosine_scores

    @staticmethod
    def multi_krum(
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        f: int = 1
    ) -> Tuple[Dict[str, torch.Tensor], List[int], Dict[int, float]]:
        """
        Multi-Krum Byzantine-robust aggregation.

        For each client, computes the sum of squared L2 distances to its
        (n - f - 2) nearest neighbors. The (n - f) clients with the lowest
        scores are selected and averaged (FedAvg) to produce the aggregate.

        Args:
            client_updates: Dict mapping client ID -> parameter dict of tensors.
            f: Number of assumed Byzantine clients (default 1).

        Returns:
            Tuple of:
              - aggregated_update: Dict[str, torch.Tensor] (FedAvg of selected clients)
              - selected_client_ids: List[int] of chosen client IDs
              - scores: Dict[int, float] mapping client ID to its Krum score
        """
        client_ids = list(client_updates.keys())
        n = len(client_ids)

        # Failsafe: cap f so we always select at least 1 client
        if f >= n // 2:
            f = max(0, n // 2 - 1)

        # Flatten each client update to a 1D float vector
        flat_vectors: Dict[int, torch.Tensor] = {}
        for cid in client_ids:
            tensors = [t.flatten().cpu().float()
                       for t in client_updates[cid].values()
                       if t.is_floating_point()]
            flat_vectors[cid] = torch.cat(tensors)

        # Number of neighbors to consider
        num_neighbors = n - f - 2
        num_neighbors = max(1, num_neighbors)

        # Compute pairwise squared L2 distances and Krum scores
        scores: Dict[int, float] = {}
        for i in client_ids:
            dists = []
            for j in client_ids:
                if i == j:
                    continue
                diff = flat_vectors[i] - flat_vectors[j]
                dists.append(torch.sum(diff ** 2).item())
            dists.sort()
            scores[i] = float(sum(dists[:num_neighbors]))

        # Select (n - f) clients with lowest scores
        num_select = max(1, n - f)
        selected_client_ids = sorted(scores.keys(), key=lambda c: scores[c])[:num_select]

        # FedAvg over selected clients
        aggregated_update: Dict[str, torch.Tensor] = {}
        keys = list(client_updates[selected_client_ids[0]].keys())
        for key in keys:
            stacked = torch.stack(
                [client_updates[cid][key].float() for cid in selected_client_ids], dim=0
            )
            aggregated_update[key] = torch.mean(stacked, dim=0)

        return aggregated_update, selected_client_ids, scores

    @staticmethod
    def flame_aggregation(
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        noise_multiplier: float = 0.001
    ) -> Tuple[Dict[str, torch.Tensor], List[int], Any]:
        """
        FLAME (Federated Learning with Anomaly and Malicious-client Elimination) defense.

        Clusters client updates using DBSCAN on pairwise cosine distances,
        keeps only the largest cluster, clips updates to the cluster's median norm,
        averages them, then adds Gaussian noise for server-side differential privacy.

        Args:
            client_updates: Dict mapping client ID -> parameter dict of tensors.
            noise_multiplier: Scale factor for Gaussian noise added to the aggregate.

        Returns:
            Tuple of:
              - aggregated_update: Dict[str, torch.Tensor]
              - accepted_client_ids: List[int] of clients in the largest cluster
              - cluster_labels: Array of DBSCAN cluster labels (or None on fallback)
        """
        client_ids = list(client_updates.keys())

        # Flatten each update to a numpy array
        flat_np: Dict[int, np.ndarray] = {}
        for cid in client_ids:
            tensors = [t.flatten().cpu().float().numpy()
                       for t in client_updates[cid].values()
                       if t.is_floating_point()]
            flat_np[cid] = np.concatenate(tensors)

        try:
            from sklearn.cluster import DBSCAN

            vecs = np.stack([flat_np[cid] for cid in client_ids], axis=0)  # (n, d)

            # Pairwise cosine distances matrix
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            normed = vecs / norms
            cos_sim_matrix = normed @ normed.T
            cos_dist_matrix = 1.0 - cos_sim_matrix
            cos_dist_matrix = np.clip(cos_dist_matrix, 0.0, 2.0)

            # Adaptive epsilon: Use 75th percentile of distances or at least 1.0 (to group all positive alignment)
            triu_dists = cos_dist_matrix[np.triu_indices(len(client_ids), k=1)]
            p75 = float(np.percentile(triu_dists, 75)) if len(triu_dists) > 0 else 0.9
            adaptive_eps = max(0.95, p75)

            db = DBSCAN(eps=adaptive_eps, min_samples=max(1, len(client_ids) // 2), metric='precomputed')
            cluster_labels = db.fit_predict(cos_dist_matrix)

            # Find the largest cluster (ignore noise label -1)
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            valid_mask = unique_labels != -1
            
            # If DBSCAN put points in noise or clusters of size 1, group by positive alignment with median
            if not np.any(valid_mask) or np.max(counts[valid_mask]) < max(1, len(client_ids) // 2):
                median_vec = np.median(vecs, axis=0)
                med_norm = np.linalg.norm(median_vec) + 1e-8
                cos_with_med = (vecs @ median_vec) / (np.linalg.norm(vecs, axis=1) * med_norm)
                # Keep clients with positive cosine alignment
                accepted_indices = [i for i, c in enumerate(cos_with_med) if c >= 0.0]
                if not accepted_indices:
                    accepted_indices = [int(np.argmax(cos_with_med))]
                accepted_client_ids = [client_ids[i] for i in accepted_indices]
                cluster_labels = np.array([0 if i in accepted_indices else -1 for i in range(len(client_ids))])
            else:
                largest_label = unique_labels[valid_mask][np.argmax(counts[valid_mask])]
                accepted_indices = [i for i, lbl in enumerate(cluster_labels) if lbl == largest_label]
                accepted_client_ids = [client_ids[i] for i in accepted_indices]

            # Compute median norm of cluster members
            cluster_norms = [
                float(np.linalg.norm(flat_np[cid])) for cid in accepted_client_ids
            ]
            median_norm = float(np.median(cluster_norms))
            if median_norm < 1e-8:
                median_norm = 1e-8

            # Clip each accepted update to median_norm
            clipped_vecs = []
            for cid in accepted_client_ids:
                v = flat_np[cid]
                v_norm = np.linalg.norm(v) + 1e-8
                scale = min(1.0, median_norm / v_norm)
                clipped_vecs.append(v * scale)

            # Average clipped updates
            avg_vec = np.mean(np.stack(clipped_vecs, axis=0), axis=0)

            # Add Gaussian noise
            noise_std = noise_multiplier * median_norm
            avg_vec = avg_vec + np.random.normal(0.0, noise_std, size=avg_vec.shape)

            # Reconstruct the parameter dict from the averaged flat vector
            aggregated_update: Dict[str, torch.Tensor] = {}
            offset = 0
            ref_update = client_updates[accepted_client_ids[0]]
            for key, ref_tensor in ref_update.items():
                if not ref_tensor.is_floating_point():
                    aggregated_update[key] = ref_tensor.clone()
                    continue
                numel = ref_tensor.numel()
                chunk = avg_vec[offset: offset + numel]
                aggregated_update[key] = torch.tensor(
                    chunk, dtype=ref_tensor.dtype
                ).reshape(ref_tensor.shape)
                offset += numel

            return aggregated_update, accepted_client_ids, cluster_labels

        except Exception:
            # Fallback: trimmed_mean over all clients
            all_updates_list = [client_updates[cid] for cid in client_ids]
            fallback_agg = ByzantineDefenses.trimmed_mean(all_updates_list)
            return fallback_agg, client_ids, None

    @staticmethod
    def adaptive_clip_norm(
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        multiplier: float = 1.0
    ) -> float:
        """
        Computes a suggested adaptive clipping threshold based on the median L2 norm
        of all client updates.

        Args:
            client_updates: Dict mapping client ID -> parameter dict of tensors.
            multiplier: Scale factor applied to the median norm (default 1.0).

        Returns:
            float: Suggested clip threshold = median_norm * multiplier.
        """
        norms = []
        for cid, update in client_updates.items():
            total_sq = sum(
                torch.sum(t ** 2).item()
                for t in update.values()
                if t.is_floating_point()
            )
            norms.append(math.sqrt(total_sq))

        median_norm = float(np.median(norms))
        return median_norm * multiplier


class ClientReputationTracker:
    """
    Tracks and updates per-client trust scores using exponential moving averages.
    Provides reputation-weighted federated aggregation.
    """

    def __init__(
        self,
        num_clients: int,
        decay: float = 0.9,
        initial_trust: float = 1.0
    ) -> None:
        """
        Args:
            num_clients: Total number of clients (IDs assumed 0 .. num_clients-1).
            decay: EMA decay factor for trust updates (default 0.9).
            initial_trust: Starting trust score for all clients (default 1.0).
        """
        self.decay = decay
        self.initial_trust = initial_trust
        self.trust_scores: Dict[int, float] = {
            i: initial_trust for i in range(num_clients)
        }

    def update_trust(self, client_id: int, cosine_sim_with_global: float) -> None:
        """
        Updates the trust score for a client using an exponential moving average.

        τ_i = decay * τ_i + (1 - decay) * max(0, cosine_sim)

        The score is then clamped to [0.1, 1.0].

        Args:
            client_id: ID of the client to update.
            cosine_sim_with_global: Cosine similarity of this client's update
                                    with the global model direction.
        """
        if client_id not in self.trust_scores:
            self.trust_scores[client_id] = self.initial_trust

        current = self.trust_scores[client_id]
        contribution = max(0.0, cosine_sim_with_global)
        updated = self.decay * current + (1.0 - self.decay) * contribution
        self.trust_scores[client_id] = float(np.clip(updated, 0.1, 1.0))

    def get_trust_scores(self) -> Dict[int, float]:
        """
        Returns:
            Dict[int, float]: Current trust score for each tracked client.
        """
        return dict(self.trust_scores)

    def weighted_aggregate(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_sample_counts: Dict[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates client updates weighted by trust score × sample count,
        normalised so that all weights sum to 1.

        Args:
            client_updates: Dict mapping client ID -> parameter dict of tensors.
            client_sample_counts: Dict mapping client ID -> number of local samples.

        Returns:
            Dict[str, torch.Tensor]: Weighted aggregate of client updates.
        """
        raw_weights: Dict[int, float] = {}
        for cid in client_updates:
            tau = self.trust_scores.get(cid, self.initial_trust)
            n_samples = client_sample_counts.get(cid, 1)
            raw_weights[cid] = tau * n_samples

        total_weight = sum(raw_weights.values()) + 1e-8
        norm_weights = {cid: w / total_weight for cid, w in raw_weights.items()}

        aggregated: Dict[str, torch.Tensor] = {}
        client_ids = list(client_updates.keys())
        keys = list(client_updates[client_ids[0]].keys())

        for key in keys:
            weighted_sum = sum(
                norm_weights[cid] * client_updates[cid][key].float()
                for cid in client_ids
            )
            aggregated[key] = weighted_sum  # type: ignore[assignment]

        return aggregated

    def penalize(self, client_id: int, factor: float = 0.5) -> None:
        """
        Multiplies the trust score of a flagged client by `factor`, then clamps to [0.1, 1.0].

        Args:
            client_id: ID of the client to penalise.
            factor: Multiplicative penalty factor (default 0.5).
        """
        if client_id not in self.trust_scores:
            self.trust_scores[client_id] = self.initial_trust

        penalized = self.trust_scores[client_id] * factor
        self.trust_scores[client_id] = float(np.clip(penalized, 0.1, 1.0))

    def reset(self, client_id: int) -> None:
        """
        Resets the trust score of a client to the initial trust value.

        Args:
            client_id: ID of the client to reset.
        """
        self.trust_scores[client_id] = self.initial_trust
