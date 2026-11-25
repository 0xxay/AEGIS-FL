"""
test_security_upgrades.py — Unit Tests for Privacy, Byzantine Defenses, Checkpoints, and Cryptographic Layers.
"""

import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.privacy import (
    PrivacyEngineDP,
    LocalDPEngine,
    PedersenCommitment,
    HomomorphicEncryptionEngine,
    _compute_rdp_epsilon,
)
from src.defenses import ByzantineDefenses, ClientReputationTracker
from src.server import CentralFederatedServer
from src.models import MLPIDSModel
from src.attacks import AdversarialAttacks


class TestSecurityUpgrades(unittest.TestCase):

    def test_rdp_accounting_and_subsampling_amplification(self):
        """Test tight Rényi DP accountant including Opacus subsampled Gaussian amplification."""
        rdp_unsub, best_alpha = _compute_rdp_epsilon(sigma=0.85, rounds=50, delta=1e-5, sample_rate=1.0)
        rdp_sub, _ = _compute_rdp_epsilon(sigma=0.85, rounds=50, delta=1e-5, sample_rate=0.03)
        self.assertGreater(rdp_unsub, 0.0)
        self.assertLess(rdp_sub, rdp_unsub)
        self.assertLess(rdp_sub, 10.0)

        # Verify Opacus per-sample DP-SGD training loop
        engine = PrivacyEngineDP(clipping_norm=1.0, noise_multiplier=0.9, delta=1e-5)
        model = MLPIDSModel(input_dim=12, num_classes=2)
        ds = TensorDataset(torch.randn(128, 12), torch.randint(0, 2, (128,)))
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        loss, eps = engine.train_with_opacus(model, loader, epochs=1, lr=0.002, criterion=torch.nn.CrossEntropyLoss(), device=torch.device("cpu"))
        self.assertGreater(loss, 0.0)
        self.assertLess(eps, 15.0)
        stats = engine.get_privacy_budget()
        self.assertLess(stats["rdp_epsilon"], stats["epsilon"])

    def test_topk_gradient_sparsification(self):
        """Test energy-preserving Top-k sparsification against gradient inversion attacks."""
        engine = PrivacyEngineDP()
        tensor = torch.arange(1, 101, dtype=torch.float32).reshape(10, 10)
        update_dict = {"layer": tensor}

        sparse_dict = engine.sparsify_update(update_dict, sparsity_ratio=0.10)
        sparse_tensor = sparse_dict["layer"]

        self.assertEqual(sparse_tensor.shape, tensor.shape)
        non_zero_count = int((sparse_tensor != 0).sum().item())
        self.assertEqual(non_zero_count, 10)
        # Largest coordinate remains at index 99, and total L2 energy is conserved
        self.assertEqual(int(torch.argmax(sparse_tensor).item()), 99)
        self.assertTrue(torch.allclose(sparse_tensor.norm(), tensor.norm(), rtol=1e-4))

    def test_mia_prediction_defense(self):
        """Test Membership Inference Attack defense via Laplace logit perturbation."""
        engine = PrivacyEngineDP()
        logits = torch.tensor([[2.5, 0.1], [0.3, 3.2]], dtype=torch.float32)
        defended_logits = engine.defend_inference(logits, epsilon_local=1.0)
        self.assertEqual(defended_logits.shape, logits.shape)
        self.assertFalse(torch.allclose(logits, defended_logits))

    def test_local_dp_randomized_response(self):
        """Test Local DP on binary and multi-class training labels."""
        labels = torch.ones(1000, dtype=torch.long)
        perturbed_high_eps = LocalDPEngine.randomized_response_labels(labels, epsilon=5.0, seed=42)
        perturbed_low_eps = LocalDPEngine.randomized_response_labels(labels, epsilon=0.01, seed=42)

        self.assertGreater(int((perturbed_high_eps == 1).sum().item()), 900)
        self.assertTrue(400 <= int((perturbed_low_eps == 1).sum().item()) <= 600)

        multi_labels = torch.full((500,), 3, dtype=torch.long)
        multi_pert = LocalDPEngine.randomized_response_labels(multi_labels, epsilon=1.5, seed=7, num_classes=10)
        self.assertTrue(int((multi_pert < 10).all().item()) == 1)

    def test_pedersen_binding_and_homomorphic_commitment(self):
        """Test both SHA-256 binding and discrete-log homomorphic Pedersen commitments."""
        pc = PedersenCommitment()
        val = np.array([12.345, 67.890], dtype=np.float64)

        commitment_hash, blinding = pc.commit(val)
        self.assertTrue(pc.verify(val, commitment_hash, blinding))
        self.assertFalse(pc.verify(np.array([12.345, 67.891]), commitment_hash, blinding))

        # Discrete-log homomorphic property: C(v1, r1) * C(v2, r2) == C(v1 + v2, r1 + r2) mod p
        c1, r1 = pc.commit_homomorphic(1.25)
        c2, r2 = pc.commit_homomorphic(2.75)
        self.assertTrue(pc.verify_homomorphic_sum([c1, c2], 4.00, r1 + r2))
        self.assertFalse(pc.verify_homomorphic_sum([c1, c2], 4.05, r1 + r2))

    def test_paillier_homomorphic_encryption(self):
        """Test Paillier Homomorphic Encryption aggregation."""
        he = HomomorphicEncryptionEngine(key_length=1024)
        pub, priv = he.generate_keys()
        enc1 = he.encrypt_vector(np.array([1.5, 2.5]), pub)
        enc2 = he.encrypt_vector(np.array([3.0, 4.0]), pub)
        decrypted_sum = he.decrypt_vector(he.aggregate_encrypted([enc1, enc2]), priv)
        np.testing.assert_allclose(decrypted_sum, np.array([4.5, 6.5]), atol=1e-3)

    def test_multi_krum_and_flame_defenses(self):
        """Test Multi-Krum and FLAME against poisoned outlier clients."""
        honest_update = {"hidden": torch.ones(5, 5) * 0.1, "head": torch.tensor([[0.2, -0.2]])}
        poisoned_update = {"hidden": torch.ones(5, 5) * -10.0, "head": torch.tensor([[-5.0, 5.0]])}

        client_updates = {
            0: honest_update,
            1: {"hidden": honest_update["hidden"] + 0.01, "head": honest_update["head"] + 0.01},
            2: {"hidden": honest_update["hidden"] - 0.01, "head": honest_update["head"] - 0.01},
            3: {"hidden": honest_update["hidden"] + 0.02, "head": honest_update["head"] + 0.02},
            4: poisoned_update,
        }

        _, selected_cids, _ = ByzantineDefenses.multi_krum(client_updates, f=1)
        self.assertNotIn(4, selected_cids)

        _, flame_accepted, _ = ByzantineDefenses.flame_aggregation(client_updates)
        self.assertNotIn(4, flame_accepted)

    def test_backdoor_injection_and_asr_metric(self):
        """Verify stealth backdoor stamping and Attack Success Rate calculation."""
        x = torch.randn(40, 12)
        y = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        bx, by = AdversarialAttacks.apply_targeted_backdoor(x, y, trigger_value=4.5, target_label=0, poison_fraction=0.5)
        # Half of the 20 attack rows should now be relabeled to 0
        self.assertEqual(int((by == 0).sum().item()), 30)

        model = MLPIDSModel(input_dim=12, num_classes=2)
        loader = DataLoader(TensorDataset(x, y), batch_size=20)
        asr = AdversarialAttacks.evaluate_backdoor_asr(model, loader, torch.device("cpu"))
        self.assertTrue(0.0 <= asr <= 1.0)

    def test_reputation_tracker(self):
        """Test Client Reputation Tracker dynamic trust scoring."""
        tracker = ClientReputationTracker(num_clients=3, decay=0.8, initial_trust=1.0)
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

            server._record_audit_block(1, [0, 1, 2], [], {"accuracy": 0.85, "f1": 0.82, "loss": 0.3})
            server._record_audit_block(2, [0, 1, 2], [], {"accuracy": 0.88, "f1": 0.86, "loss": 0.25})

            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(server.audit_log, f, indent=2)

            is_valid, _ = CentralFederatedServer.verify_audit_log(audit_path)
            self.assertTrue(is_valid)

            with open(audit_path, "r", encoding="utf-8") as f:
                tampered = json.load(f)
            tampered[0]["accuracy"] = 0.999
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(tampered, f, indent=2)

            is_valid_tampered, tamper_msg = CentralFederatedServer.verify_audit_log(audit_path)
            self.assertFalse(is_valid_tampered)
            self.assertIn("tampering detected", tamper_msg)


if __name__ == "__main__":
    unittest.main()
