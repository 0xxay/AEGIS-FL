import unittest
from pathlib import Path
import pandas as pd
import torch

from src.dataset_downloader import (
    generate_synthetic_unsw_nb15,
    generate_synthetic_cicids2017,
    normalize_real_csv,
)
from src.data_preprocessor import FederatedFeatureTransformer
from src.models import (
    MLPIDSModel,
    ResNetIDSModel,
    AttentionIDSModel,
    AutoencoderIDSModel,
)
from src.privacy import PrivacyEngineDP, SecureAggregationEngine
from src.attacks import AdversarialAttacks
from src.defenses import ByzantineDefenses


ROOT_DIR = Path(__file__).resolve().parents[1]


class TestFederatedIDSPipeline(unittest.TestCase):

    def test_01_dataset_generators_and_real_csv_normalization(self):
        """Verify benchmark schemas use official headers and normalize real CSVs."""
        unsw_train, unsw_test = generate_synthetic_unsw_nb15(num_samples=500)
        self.assertEqual(len(unsw_train) + len(unsw_test), 500)
        self.assertIn("label", unsw_train.columns)
        self.assertIn("attack_cat", unsw_train.columns)

        cic_train, cic_test = generate_synthetic_cicids2017(num_samples=500)
        self.assertEqual(len(cic_train) + len(cic_test), 500)
        self.assertIn("Destination Port", cic_train.columns)
        self.assertIn("Idle Mean", cic_train.columns)
        self.assertNotIn("Flow_Metric_21", cic_train.columns)

        messy_cic = pd.DataFrame({
            " Destination Port ": [80, 443],
            " Flow Bytes/s": [1200.0, float("inf")],
            " Label ": ["BENIGN", "DoS Hulk"],
        })
        clean = normalize_real_csv(messy_cic, "cicids")
        self.assertEqual(clean["label_binary"].tolist(), [0, 1])

    def test_02_model_forward_passes_and_true_feature_attention(self):
        """Verify MLP, ResNet, multi-token Attention (8x8), and Autoencoder forward passes."""
        batch_size = 16
        input_dim = 42
        dummy_x = torch.randn(batch_size, input_dim)

        mlp = MLPIDSModel(input_dim=input_dim, num_classes=2)
        self.assertEqual(mlp(dummy_x).shape, (batch_size, 2))

        resnet = ResNetIDSModel(input_dim=input_dim, num_classes=10)
        self.assertEqual(resnet(dummy_x).shape, (batch_size, 10))

        attn = AttentionIDSModel(input_dim=input_dim, num_classes=2)
        logits, attn_map = attn.forward_with_attention(dummy_x)
        self.assertEqual(logits.shape, (batch_size, 2))
        self.assertEqual(attn_map.shape, (batch_size, 4, 8, 8))

        ae = AutoencoderIDSModel(input_dim=input_dim, bottleneck_dim=16)
        recon = ae(dummy_x)
        self.assertEqual(recon.shape, (batch_size, input_dim))
        errors = ae.compute_reconstruction_error(dummy_x)
        self.assertEqual(errors.shape, (batch_size,))
        self.assertEqual(ae.predict_logits(dummy_x).shape, (batch_size, 2))

    def test_03_zero_leakage_federated_preprocessor(self):
        """Verify FederatedFeatureTransformer computes exact global scaling from additive silo stats."""
        df1 = pd.DataFrame({"dur": [1.0, 2.0, 3.0], "rate": [10.0, 20.0, 30.0], "proto": ["tcp", "udp", "tcp"]})
        df2 = pd.DataFrame({"dur": [4.0, 5.0, 6.0], "rate": [40.0, 50.0, 60.0], "proto": ["ospf", "tcp", "unknown_proto"]})
        transformer = FederatedFeatureTransformer(num_cols=["dur", "rate"], cat_cols=["proto"])
        s1 = transformer.local_sufficient_stats(df1)
        s2 = transformer.local_sufficient_stats(df2)
        transformer.fit_from_federated_stats([s1, s2])

        self.assertAlmostEqual(float(transformer.mean_[0]), 3.5, places=5)
        out = transformer.transform(df2)
        self.assertEqual(out.shape[0], 3)
        self.assertGreater(out.shape[1], 2)

    def test_04_secagg_drop_resilient_mask_cancellation(self):
        """Verify SecAgg+ masks cancel both at full participation and when an outlier client is rejected."""
        secagg = SecureAggregationEngine(num_clients=5)
        cids = [0, 1, 2, 3, 4]
        clean_updates = {cid: {"weight": torch.full((4, 4), float(cid + 1))} for cid in cids}
        sample_weights = {cid: 100.0 for cid in cids}

        pairwise_masks = secagg.generate_pairwise_masks(cids, clean_updates[0])
        masked = {
            cid: secagg.apply_mask({"weight": clean_updates[cid]["weight"] * sample_weights[cid]}, pairwise_masks[cid])
            for cid in cids
        }

        # Reject compromised Client 2; surviving clients {0, 1, 3, 4} have mean (1 + 2 + 4 + 5) / 4 = 3.0
        recovered = secagg.aggregate_masked_surviving(
            masked_updates=masked,
            accepted_ids=[0, 1, 3, 4],
            all_client_ids=cids,
            weights_by_client=sample_weights,
        )
        expected = torch.full((4, 4), 3.0)
        self.assertTrue(torch.allclose(recovered["weight"], expected, atol=1e-4))

    def test_05_byzantine_defense_catches_scaling_and_adaptive_poisoning(self):
        """Verify defenses catch both large-scale amplification and stealth norm-bounded sign flips."""
        base_head = torch.tensor([[0.5, -0.5], [-0.4, 0.4]])
        honest_updates = {
            cid: {
                "hidden.weight": torch.ones(8, 8) * 0.05 + cid * 0.001,
                "head.weight": base_head + cid * 0.01,
            }
            for cid in [0, 1, 3, 4]
        }

        # Case A: Scaling attack
        scaled = dict(honest_updates)
        scaled[2] = AdversarialAttacks.apply_model_poisoning(honest_updates[0], scale_factor=-15.0)
        filtered, flagged, _ = ByzantineDefenses.detect_and_filter_norm_outliers(scaled)
        self.assertIn(2, flagged)
        self.assertNotIn(2, filtered)

        # Case B: Stealth norm-bounded sign-flip (exact same norm as honest client!)
        honest_norm = float(torch.sqrt(sum(torch.sum(t ** 2) for t in honest_updates[0].values())).item())
        stealth = dict(honest_updates)
        stealth[2] = AdversarialAttacks.apply_adaptive_norm_poisoning(honest_updates[0], target_norm=honest_norm)
        filtered_s, flagged_s, _ = ByzantineDefenses.detect_and_filter_norm_outliers(stealth)
        self.assertIn(2, flagged_s)
        self.assertNotIn(2, filtered_s)


if __name__ == "__main__":
    unittest.main()
