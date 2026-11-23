"""
test_security_upgrades.py — Unit Tests for Privacy, Byzantine Defenses, and Cryptographic Layers.
"""

import unittest
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import torch

from src.privacy import (
    PrivacyEngineDP,
    LocalDPEngine,
    PedersenCommitment,
    HomomorphicEncryptionEngine,
    SecureAggregationEngine,
    _compute_rdp_epsilon
)
from src.defenses import ByzantineDefenses, ClientReputationTracker
from src.server import CentralFederatedServer


class TestSecurityUpgrades(unittest.TestCase):

    def test_rdp_accounting(self):
        """Test tight Rényi Differential Privacy moments accountant."""
        sigma = 0.5
        rounds = 5
        delta = 1e-5
        rdp_eps, best_alpha = _compute_rdp_epsilon(sigma=sigma, rounds=rounds, delta=delta)
        self.assertGreater(rdp_eps, 0.0)
        self.assertIn(best_alpha, [2, 4, 8, 16, 32, 64])

        # Test within PrivacyEngineDP
        engine = PrivacyEngineDP(clipping_norm=1.0, noise_multiplier=0.5, delta=1e-5)
        dummy_update = {"fc.weight": torch.randn(10, 5)}
        engine.perturb_update(dummy_update)
        stats = engine.get_privacy_budget()
        self.assertIn("rdp_epsilon", stats)
        self.assertGreater(stats["rdp_epsilon"], 0.0)

    def test_topk_gradient_sparsification(self):
        """Test Top-k sparsification against gradient inversion attacks."""
        engine = PrivacyEngineDP()
        # Create a tensor with 100 elements
        tensor = torch.arange(1, 101, dtype=torch.float32).reshape(10, 10)
        update_dict = {"layer": tensor}

        # Sparsify top 10%
        sparse_dict = engine.sparsify_update(update_dict, sparsity_ratio=0.10)
        sparse_tensor = sparse_dict["layer"]

        self.assertEqual(sparse_tensor.shape, tensor.shape)
        # Should retain exactly 10 non-zero elements
        non_zero_count = int((sparse_tensor != 0).sum().item())
        self.assertEqual(non_zero_count, 10)
        # The largest element (100) should be retained
        self.assertEqual(sparse_tensor.max().item(), 100.0)

    def test_mia_prediction_defense(self):
        """Test Membership Inference Attack defense via prediction perturbation."""
        engine = PrivacyEngineDP()
        logits = torch.tensor([[2.5, 0.1], [0.3, 3.2]], dtype=torch.float32)
        defended_logits = engine.defend_inference(logits, epsilon_local=1.0)
        self.assertEqual(defended_logits.shape, logits.shape)
        self.assertFalse(torch.allclose(logits, defended_logits))

    def test_local_dp_randomized_response(self):
        """Test Local DP on training labels with Randomized Response."""
        labels = torch.ones(1000, dtype=torch.long)
        # Large epsilon -> low flip probability
        perturbed_high_eps = LocalDPEngine.randomized_response_labels(labels, epsilon=5.0, seed=42)
        # Small epsilon -> high flip probability (near 50%)
        perturbed_low_eps = LocalDPEngine.randomized_response_labels(labels, epsilon=0.01, seed=42)

        high_eps_ones = int((perturbed_high_eps == 1).sum().item())
        low_eps_ones = int((perturbed_low_eps == 1).sum().item())

        self.assertGreater(high_eps_ones, 900)
        self.assertTrue(400 <= low_eps_ones <= 600)

    def test_pedersen_commitment(self):
        """Test cryptographic commitment and tamper detection."""
        pc = PedersenCommitment()
        val = np.array([12.345, 67.890], dtype=np.float64)

        commitment_hash, blinding = pc.commit(val)
        self.assertEqual(len(commitment_hash), 64)
        self.assertEqual(len(blinding), 32)

        # Honest verification
        self.assertTrue(pc.verify(val, commitment_hash, blinding))

        # Tampered value detection
        tampered_val = np.array([12.345, 67.891], dtype=np.float64)
        self.assertFalse(pc.verify(tampered_val, commitment_hash, blinding))

    def test_paillier_homomorphic_encryption(self):
        """Test Paillier Homomorphic Encryption aggregation."""
        he = HomomorphicEncryptionEngine(key_length=1024)
        pub, priv = he.generate_keys()

        # Vector 1: [1.5, 2.5], Vector 2: [3.0, 4.0]
        v1 = np.array([1.5, 2.5])
        v2 = np.array([3.0, 4.0])

        enc1 = he.encrypt_vector(v1, pub)
        enc2 = he.encrypt_vector(v2, pub)

        # Homomorphic addition on ciphertexts
        enc_sum = he.aggregate_encrypted([enc1, enc2])

        # Decrypt only the aggregated sum
        decrypted_sum = he.decrypt_vector(enc_sum, priv)
        expected_sum = np.array([4.5, 6.5])

        np.testing.assert_allclose(decrypted_sum, expected_sum, atol=1e-3)

    def test_multi_krum_defense(self):
        """Test Multi-Krum defense against a poisoned outlier client."""
        # 4 honest clients with similar updates, 1 malicious client scaled 50x
        honest_update = {"weight": torch.ones(5, 5) * 0.1}
        poisoned_update = {"weight": torch.ones(5, 5) * -10.0}

        client_updates = {
            0: honest_update,
            1: {"weight": honest_update["weight"] + 0.01},
            2: {"weight": honest_update["weight"] - 0.01},
            3: {"weight": honest_update["weight"] + 0.02},
            4: poisoned_update
        }

        aggregated, selected_cids, scores = ByzantineDefenses.multi_krum(client_updates, f=1)
        # Malicious Client 4 must NOT be in selected clients
        self.assertNotIn(4, selected_cids)
        self.assertIn(0, selected_cids)

    def test_reputation_tracker(self):
        """Test Client Reputation Tracker dynamic trust scoring."""
        tracker = ClientReputationTracker(num_clients=3, decay=0.8, initial_trust=1.0)
        # Client 0 always aligned (cos = 0.95), Client 2 misaligned (cos = -0.8)
        tracker.update_trust(0, 0.95)
        tracker.update_trust(1, 0.50)
        tracker.update_trust(2, -0.80)

        scores = tracker.get_trust_scores()
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_cryptographic_audit_log_verification(self):
        """Test SHA-256 Merkle audit chain creation and tampering detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "audit_log.json"
            server = CentralFederatedServer(input_dim=58, enable_audit_log=True)

            # Record two blocks
            server._record_audit_block(1, [0, 1, 2], [], {"accuracy": 0.85, "f1": 0.82, "loss": 0.3})
            server._record_audit_block(2, [0, 1, 2], [], {"accuracy": 0.88, "f1": 0.86, "loss": 0.25})

            # Save to temporary path
            with open(audit_path, "w") as f:
                json.dump(server.audit_log, f, indent=2)

            # Valid check
            is_valid, msg = CentralFederatedServer.verify_audit_log(audit_path)
            self.assertTrue(is_valid)

            # Tamper with accuracy in block 1
            with open(audit_path, "r") as f:
                tampered = json.load(f)
            tampered[0]["accuracy"] = 0.999
            with open(audit_path, "w") as f:
                json.dump(tampered, f, indent=2)

            # Tampering must be caught
            is_valid_tampered, tamper_msg = CentralFederatedServer.verify_audit_log(audit_path)
            self.assertFalse(is_valid_tampered)
            self.assertIn("tampering detected", tamper_msg)


if __name__ == "__main__":
    unittest.main()
