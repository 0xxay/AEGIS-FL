"""
privacy.py — Privacy & Security Engines for Federated Learning.

Provides:
  - PrivacyEngineDP      : DP-SGD with tight RDP accounting, top-k sparsification, and MIA defense.
  - SecureAggregationEngine : Pairwise zero-sum mask generation and verification (SecAgg+).
  - PedersenCommitment   : SHA-256-based cryptographic commitment scheme.
"""

import hashlib
import math
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RDP_ALPHA_RANGE: List[int] = [2, 4, 8, 16, 32, 64]


# ---------------------------------------------------------------------------
# Helper: RDP accounting (Gaussian mechanism)
# ---------------------------------------------------------------------------

def _rdp_gaussian(alpha: float, sigma: float) -> float:
    """
    Closed-form RDP bound for the Gaussian mechanism.

    For order α, the per-round RDP of the Gaussian mechanism with
    noise multiplier σ (std = σ · C for clipping norm C) is:

        ε_RDP(α) = α / (2 σ²)

    Args:
        alpha: Rényi order (must be > 1).
        sigma: Noise multiplier (σ = std / clipping_norm).

    Returns:
        Per-round RDP value ε_RDP(α).
    """
    return alpha / (2.0 * sigma ** 2)


def _rdp_to_dp(rdp_value: float, alpha: float, delta: float) -> float:
    """
    Convert a single (α, ε_RDP) pair to (ε, δ)-DP via the standard conversion:

        ε(δ) = ε_RDP(α) + log(1/δ) / (α - 1)

    Args:
        rdp_value: RDP value ε_RDP(α) for the given order α.
        alpha: Rényi order (> 1).
        delta: Target δ for (ε, δ)-DP.

    Returns:
        (ε, δ)-DP epsilon bound for this α.
    """
    return rdp_value + math.log(1.0 / delta) / (alpha - 1.0)


def _compute_rdp_epsilon(
    sigma: float,
    rounds: int,
    delta: float,
    alpha_range: List[int] = _RDP_ALPHA_RANGE,
) -> Tuple[float, float]:
    """
    Compute the tightest (ε, δ)-DP guarantee using RDP composition.

    Composition over T rounds multiplies the per-round RDP by T, then
    we minimise over all candidate α values to find the tightest bound.

    Args:
        sigma:       Noise multiplier.
        rounds:      Number of rounds T composed so far.
        delta:       Target δ.
        alpha_range: Candidate Rényi orders to search over.

    Returns:
        (rdp_epsilon, best_alpha) — the tightest ε and the α that achieved it.
    """
    best_eps = float("inf")
    best_alpha = alpha_range[0]
    T = max(1, rounds)
    for alpha in alpha_range:
        composed_rdp = T * _rdp_gaussian(alpha, sigma)
        eps_candidate = _rdp_to_dp(composed_rdp, alpha, delta)
        if eps_candidate < best_eps:
            best_eps = eps_candidate
            best_alpha = alpha
    return best_eps, float(best_alpha)


# ---------------------------------------------------------------------------
# PrivacyEngineDP
# ---------------------------------------------------------------------------

class PrivacyEngineDP:
    """
    Differential Privacy (DP) Engine for Federated Learning.

    Applies per-round gradient / weight-update clipping and calibrated
    Gaussian perturbation (DP-SGD).  Privacy accounting uses a tight
    Rényi DP (RDP) moments accountant instead of loose linear composition.

    Additional defences
    -------------------
    * Top-k sparsification  — ``sparsify_update()``   guards against gradient inversion.
    * MIA prediction noise  — ``defend_inference()``  guards against membership inference.
    """

    def __init__(
        self,
        clipping_norm: float = 1.0,
        noise_multiplier: float = 0.05,
        delta: float = 1e-5,
        rdp_alpha_range: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            clipping_norm:    L2 norm bound C for gradient clipping.
            noise_multiplier: σ — Gaussian noise std = σ · C per round.
            delta:            Target δ for (ε, δ)-DP guarantee.
            rdp_alpha_range:  Rényi orders to search when converting RDP → (ε, δ)-DP.
                              Defaults to [2, 4, 8, 16, 32, 64].
        """
        self.clipping_norm = clipping_norm
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.rdp_alpha_range: List[int] = rdp_alpha_range or list(_RDP_ALPHA_RANGE)

        # Legacy linear-composition epsilon (kept for backwards compat)
        self.cumulative_epsilon: float = 0.0
        # Tight RDP epsilon
        self.rdp_epsilon: float = 0.0
        self.rdp_best_alpha: float = float(self.rdp_alpha_range[0])

        self.rounds_participated: int = 0

    # ------------------------------------------------------------------
    # Core DP perturbation
    # ------------------------------------------------------------------

    def perturb_update(
        self,
        update_dict: Dict[str, torch.Tensor],
        sparsify: bool = False,
        sparsity_ratio: float = 0.01,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Clip the update vector by ``clipping_norm``, add calibrated Gaussian
        noise, and optionally apply top-k sparsification.

        Args:
            update_dict:    Dict of parameter-name → tensor (model update / gradient).
            sparsify:       If ``True``, apply ``sparsify_update()`` after clipping
                            and before noise addition.  Defaults to ``False``.
            sparsity_ratio: Fraction of elements to keep when ``sparsify=True``.
                            Defaults to 0.01 (top 1 %).

        Returns:
            (perturbed_update, pre_clip_l2_norm)
        """
        # --- Compute total L2 norm across all floating-point tensors ----------
        total_sq_norm = 0.0
        for tensor in update_dict.values():
            if tensor.is_floating_point():
                total_sq_norm += torch.sum(tensor ** 2).item()
        l2_norm = math.sqrt(total_sq_norm)

        # --- Clip ----------------------------------------------------------------
        clip_factor = min(1.0, self.clipping_norm / max(1e-6, l2_norm))
        clipped: Dict[str, torch.Tensor] = {}
        for key, tensor in update_dict.items():
            if not tensor.is_floating_point():
                clipped[key] = tensor
            else:
                clipped[key] = tensor * clip_factor

        # --- Optional sparsification (after clipping, before noise) -----------
        if sparsify:
            clipped = self.sparsify_update(clipped, sparsity_ratio=sparsity_ratio)

        # --- Add Gaussian noise ---------------------------------------------------
        perturbed: Dict[str, torch.Tensor] = {}
        for key, tensor in clipped.items():
            if not tensor.is_floating_point():
                perturbed[key] = tensor
            elif self.noise_multiplier > 0:
                std_dev = self.noise_multiplier * self.clipping_norm
                perturbed[key] = tensor + torch.randn_like(tensor) * std_dev
            else:
                perturbed[key] = tensor

        self.rounds_participated += 1
        self._update_privacy_budget()
        return perturbed, l2_norm

    # ------------------------------------------------------------------
    # Privacy budget accounting
    # ------------------------------------------------------------------

    def _update_privacy_budget(self) -> None:
        """
        Update privacy expenditure estimates after each round.

        * **Tight RDP bound** (primary): uses moments accountant with the
          closed-form Gaussian RDP and minimises over ``rdp_alpha_range``.
        * **Legacy linear bound** (secondary, kept for backwards compat):
          the standard composition approximation
          ``sqrt(2T·ln(1/δ)) / σ + T / (2σ²)``.
        """
        if self.noise_multiplier <= 0:
            self.cumulative_epsilon = float("inf")
            self.rdp_epsilon = float("inf")
            return

        T = max(1, self.rounds_participated)
        sigma = self.noise_multiplier

        # Legacy / loose linear composition
        self.cumulative_epsilon = (
            math.sqrt(2 * T * math.log(1.0 / self.delta)) / sigma
            + T / (2.0 * sigma ** 2)
        )

        # Tight RDP accounting
        self.rdp_epsilon, self.rdp_best_alpha = _compute_rdp_epsilon(
            sigma=sigma,
            rounds=T,
            delta=self.delta,
            alpha_range=self.rdp_alpha_range,
        )

    def get_privacy_budget(self) -> Dict[str, float]:
        """
        Return the current privacy expenditure summary.

        Returns:
            Dict with keys:
            - ``epsilon``          : Legacy linear-composition ε estimate.
            - ``rdp_epsilon``      : Tight RDP-accountant ε (primary, recommended).
            - ``rdp_alpha_range``  : Candidate α orders used for RDP optimisation.
            - ``rdp_best_alpha``   : α that achieved the tightest bound this round.
            - ``delta``            : Target δ.
            - ``clipping_norm``    : L2 clip norm C.
            - ``noise_multiplier`` : σ.
            - ``rounds``           : Number of rounds composed so far.
        """
        return {
            "epsilon": round(float(self.cumulative_epsilon), 6),
            "rdp_epsilon": round(float(self.rdp_epsilon), 6),
            "rdp_alpha_range": list(self.rdp_alpha_range),
            "rdp_best_alpha": float(self.rdp_best_alpha),
            "delta": self.delta,
            "clipping_norm": self.clipping_norm,
            "noise_multiplier": self.noise_multiplier,
            "rounds": self.rounds_participated,
        }

    # ------------------------------------------------------------------
    # Gradient Top-k Sparsification
    # ------------------------------------------------------------------

    def sparsify_update(
        self,
        update_dict: Dict[str, torch.Tensor],
        sparsity_ratio: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Top-k gradient sparsification to defend against gradient inversion attacks.

        Flattens all floating-point tensors from ``update_dict`` into a single
        1-D vector, identifies the top ``sparsity_ratio`` fraction by absolute
        magnitude, and zeroes out every other element before returning the
        result in the original tensor shapes.

        Args:
            update_dict:    Dict of parameter-name → tensor (gradient update).
            sparsity_ratio: Fraction of elements to retain, e.g. 0.01 = top 1 %.
                            Must be in (0, 1].  Values outside this range are
                            clamped automatically.

        Returns:
            Sparse update dict with the same keys and shapes as ``update_dict``
            but with (1 − sparsity_ratio) of floating-point values zeroed out.
        """
        sparsity_ratio = max(1e-6, min(1.0, sparsity_ratio))

        # Collect all float tensors and their flat views
        float_keys = [k for k, v in update_dict.items() if v.is_floating_point()]
        if not float_keys:
            return dict(update_dict)

        flat_parts = [update_dict[k].detach().float().flatten() for k in float_keys]
        flat_all = torch.cat(flat_parts)                 # single 1-D tensor

        total_elements = flat_all.numel()
        k = max(1, int(math.ceil(sparsity_ratio * total_elements)))

        # Find the k-th largest absolute value → threshold
        abs_flat = flat_all.abs()
        if k >= total_elements:
            # Keep everything
            threshold = 0.0
        else:
            # topk returns sorted values; the smallest of the top-k is our threshold
            threshold = torch.topk(abs_flat, k, largest=True, sorted=True).values[-1].item()

        # Build sparse output with energy conservation and structural layer preservation
        sparse_dict: Dict[str, torch.Tensor] = {}
        for key, tensor in update_dict.items():
            if not tensor.is_floating_point() or "bias" in key.lower() or "norm" in key.lower():
                sparse_dict[key] = tensor
            else:
                mask = tensor.abs() >= threshold
                sparse_t = tensor * mask.to(tensor.dtype)
                orig_norm = tensor.norm()
                sparse_norm = sparse_t.norm()
                if sparse_norm > 1e-8:
                    # Unbiased energy preservation scaling so learning rate and gradient flow remain intact
                    sparse_dict[key] = sparse_t * (orig_norm / sparse_norm)
                else:
                    sparse_dict[key] = sparse_t

        return sparse_dict

    # ------------------------------------------------------------------
    # MIA Defense: Prediction Perturbation
    # ------------------------------------------------------------------

    def defend_inference(
        self,
        logits_tensor: torch.Tensor,
        epsilon_local: float = 0.5,
    ) -> torch.Tensor:
        """
        Membership Inference Attack (MIA) defence via calibrated Laplace noise
        on raw model logits (pre-softmax).

        Adds output-perturbation noise drawn from ``Laplace(0, Δf / ε_local)``
        where the sensitivity ``Δf`` is the per-sample logit range (max − min).
        This prevents an adversary from inferring training membership from
        confidence scores while preserving the argmax prediction in expectation.

        Args:
            logits_tensor: Raw logits tensor of arbitrary shape.
            epsilon_local: Local differential-privacy budget ε for this query.
                           Smaller = more noise = stronger privacy.  Default 0.5.

        Returns:
            Perturbed logits tensor of the same shape as ``logits_tensor``.
        """
        if epsilon_local <= 0:
            raise ValueError("epsilon_local must be positive.")

        sensitivity = (logits_tensor.max() - logits_tensor.min()).item()
        if sensitivity < 1e-9:
            # Logits are all equal — no information to protect
            return logits_tensor.clone()

        scale = sensitivity / epsilon_local
        # Laplace noise: sample as difference of two Exponentials
        noise = torch.empty_like(logits_tensor).exponential_(1.0 / scale) \
              - torch.empty_like(logits_tensor).exponential_(1.0 / scale)
        return logits_tensor + noise


# ---------------------------------------------------------------------------
# SecureAggregationEngine
# ---------------------------------------------------------------------------

class SecureAggregationEngine:
    """
    Cryptographic Secure Aggregation (SecAgg+) Simulator.

    Generates pairwise zero-sum masks between active client pairs.
    When summed at the server, all pairwise masks cancel algebraically to zero,
    so the server recovers the plaintext aggregate without seeing any individual
    client update.
    """

    def __init__(self, num_clients: int = 5, seed: int = 1337) -> None:
        """
        Args:
            num_clients: Total number of clients in the federation.
            seed:        Seed for the internal NumPy RNG (reproducibility).
        """
        self.num_clients = num_clients
        self.rng = np.random.default_rng(seed)

    def generate_pairwise_masks(
        self,
        client_ids: List[int],
        reference_update: Dict[str, torch.Tensor],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Generate pairwise random noise masks s_{ij} for every client pair (i, j).

        Client i receives  ``+s_{ij}``; client j receives ``−s_{ij}``.
        The sum across all clients therefore cancels to zero for every parameter.

        Args:
            client_ids:        List of participating client IDs this round.
            reference_update:  A reference update dict (used only for shapes / dtypes).

        Returns:
            Dict mapping client_id → per-parameter mask dict.
        """
        client_masks: Dict[int, Dict[str, torch.Tensor]] = {
            cid: {k: torch.zeros_like(v) for k, v in reference_update.items()}
            for cid in client_ids
        }

        for idx_i, cid_i in enumerate(client_ids):
            for idx_j in range(idx_i + 1, len(client_ids)):
                cid_j = client_ids[idx_j]
                for key, tensor in reference_update.items():
                    if not tensor.is_floating_point():
                        continue
                    random_matrix = torch.from_numpy(
                        self.rng.normal(loc=0.0, scale=0.5, size=tensor.shape).astype(np.float32)
                    )
                    client_masks[cid_i][key] += random_matrix
                    client_masks[cid_j][key] -= random_matrix

        return client_masks

    def apply_mask(
        self,
        update_dict: Dict[str, torch.Tensor],
        client_mask: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Add a client's pairwise mask to its update before server transmission.

        Args:
            update_dict:  The client's local gradient / weight update.
            client_mask:  Pairwise mask for this client (from ``generate_pairwise_masks``).

        Returns:
            Masked update dict (same keys and shapes).
        """
        return {k: update_dict[k] + client_mask[k] for k in update_dict}

    def verify_mask_cancellation(
        self,
        all_masked_updates: List[Dict[str, torch.Tensor]],
        all_clean_updates: List[Dict[str, torch.Tensor]],
    ) -> bool:
        """
        Verify that the sum of masked updates equals the sum of clean updates.

        Args:
            all_masked_updates: List of masked update dicts (one per client).
            all_clean_updates:  List of original (unmasked) update dicts.

        Returns:
            ``True`` if mask cancellation holds within numerical tolerance (1e-3).
        """
        for key in all_clean_updates[0]:
            sum_clean = sum(u[key] for u in all_clean_updates)
            sum_masked = sum(u[key] for u in all_masked_updates)
            if not torch.allclose(sum_clean, sum_masked, atol=1e-3):
                return False
        return True


# ---------------------------------------------------------------------------
# PedersenCommitment
# ---------------------------------------------------------------------------

class PedersenCommitment:
    """
    Simplified Pedersen-style cryptographic commitment scheme.

    Allows a client to commit to a gradient norm (or any numeric vector)
    *before* revealing it to the server, binding the client to the value
    while keeping it hidden until opening.

    Implementation uses SHA-256 over the concatenation of the serialised
    value bytes and a random 32-byte blinding factor.

    Usage
    -----
    ::

        pc = PedersenCommitment()
        commitment, blinding = pc.commit(my_gradient_norms)
        # … later …
        assert pc.verify(my_gradient_norms, commitment, blinding)
    """

    @staticmethod
    def _serialize(value_vector: np.ndarray) -> bytes:
        """Serialise a NumPy array to raw bytes in a stable, platform-independent way."""
        arr = np.asarray(value_vector, dtype=np.float64)
        return arr.tobytes()

    def commit(
        self,
        value_vector: np.ndarray,
        blinding_factor: Optional[bytes] = None,
    ) -> Tuple[str, bytes]:
        """
        Commit to ``value_vector`` using an optional blinding factor.

        Args:
            value_vector:    1-D (or higher) NumPy array of numeric values to commit to.
            blinding_factor: 32-byte random blinding factor.  If ``None``, a fresh
                             ``os.urandom(32)`` is generated automatically.

        Returns:
            ``(commitment_hash, blinding_factor)`` where ``commitment_hash`` is a
            64-character lowercase hex string (SHA-256 digest) and
            ``blinding_factor`` is the 32-byte secret needed to verify the commitment.
        """
        if blinding_factor is None:
            blinding_factor = os.urandom(32)
        if len(blinding_factor) != 32:
            raise ValueError("blinding_factor must be exactly 32 bytes.")

        payload = self._serialize(value_vector) + blinding_factor
        commitment_hash = hashlib.sha256(payload).hexdigest()
        return commitment_hash, blinding_factor

    def verify(
        self,
        value_vector: np.ndarray,
        commitment_hash: str,
        blinding_factor: bytes,
    ) -> bool:
        """
        Verify that ``value_vector`` matches a previously issued commitment.

        Args:
            value_vector:     The value vector to verify (must be identical to what
                              was passed to ``commit()``).
            commitment_hash:  The hex-string commitment returned by ``commit()``.
            blinding_factor:  The 32-byte blinding factor returned by ``commit()``.

        Returns:
            ``True`` iff the recomputed hash matches ``commitment_hash``.
        """
        recomputed, _ = self.commit(value_vector, blinding_factor=blinding_factor)
        return recomputed == commitment_hash


# ---------------------------------------------------------------------------
# LocalDPEngine (Label Differential Privacy via Randomized Response)
# ---------------------------------------------------------------------------

class LocalDPEngine:
    """
    Local Differential Privacy (LDP) Engine for Training Data Protection.
    
    Protects private ground-truth labels at the institutional edge before
    any gradient computation occurs, applying Warner's Randomized Response.
    Guarantees epsilon-LDP: raw intrusion labels cannot be determined with
    certainty even by an adversary who breaches the local client node.
    """

    @staticmethod
    def randomized_response_labels(
        labels: torch.Tensor,
        epsilon: float = 1.0,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Applies binary Randomized Response to intrusion detection labels.
        
        Args:
            labels: 1D torch.Tensor of integer labels {0, 1}.
            epsilon: Privacy budget parameter epsilon > 0.
            seed: Optional random seed.
            
        Returns:
            Perturbed label tensor satisfying epsilon-LDP.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive for Local DP.")
            
        if seed is not None:
            torch.manual_seed(seed)

        # p = e^eps / (e^eps + 1) -> probability of keeping true label
        keep_prob = math.exp(epsilon) / (math.exp(epsilon) + 1.0)
        
        # Draw uniform random values
        rand_vals = torch.rand_like(labels.float())
        flip_mask = rand_vals > keep_prob
        
        perturbed_labels = labels.clone()
        perturbed_labels[flip_mask] = 1 - perturbed_labels[flip_mask]
        return perturbed_labels


# ---------------------------------------------------------------------------
# HomomorphicEncryptionEngine (Paillier Cryptosystem for SecAgg)
# ---------------------------------------------------------------------------

class HomomorphicEncryptionEngine:
    """
    Partial Homomorphic Encryption (PHE) Engine using the Paillier Cryptosystem.
    
    Enables additive homomorphism: the central server sums encrypted client
    updates without ever decrypting or inspecting individual client parameters.
    E(m_1) * E(m_2) = E(m_1 + m_2).
    """

    def __init__(self, key_length: int = 1024):
        """
        Initializes Paillier cryptosystem.
        
        Args:
            key_length: Modulus bit length (1024 for testing, 2048+ for enterprise).
        """
        self.key_length = key_length
        self._pub_key = None
        self._priv_key = None

    def generate_keys(self) -> Tuple[Any, Any]:
        """Generates public and private Paillier keypair."""
        try:
            import phe.paillier as paillier
            self._pub_key, self._priv_key = paillier.generate_paillier_keypair(n_length=self.key_length)
            return self._pub_key, self._priv_key
        except ImportError:
            raise ImportError("phe library is required for Paillier Homomorphic Encryption. Run `pip install phe`.")

    @staticmethod
    def encrypt_vector(
        vector: np.ndarray,
        public_key: Any
    ) -> List[Any]:
        """
        Encrypts a 1D numpy array of numbers using Paillier public key.
        """
        flat = np.asarray(vector, dtype=np.float64).flatten()
        return [public_key.encrypt(float(val)) for val in flat]

    @staticmethod
    def aggregate_encrypted(
        encrypted_lists: List[List[Any]]
    ) -> List[Any]:
        """
        Homomorphically aggregates lists of encrypted values across clients.
        Uses additive homomorphism: sum(E(x_i)) = E(sum(x_i)).
        """
        if not encrypted_lists:
            return []
        
        num_elements = len(encrypted_lists[0])
        aggregated = [encrypted_lists[0][i] for i in range(num_elements)]
        
        for client_enc in encrypted_lists[1:]:
            for i in range(num_elements):
                aggregated[i] = aggregated[i] + client_enc[i]
                
        return aggregated

    @staticmethod
    def decrypt_vector(
        encrypted_vector: List[Any],
        private_key: Any,
        original_shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """
        Decrypts an encrypted vector using the Paillier private key.
        """
        decrypted = np.array([private_key.decrypt(val) for val in encrypted_vector], dtype=np.float32)
        if original_shape is not None:
            decrypted = decrypted.reshape(original_shape)
        return decrypted

