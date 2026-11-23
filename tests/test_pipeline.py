import unittest
from pathlib import Path
import numpy as np
import torch

from src.dataset_downloader import generate_synthetic_unsw_nb15, generate_synthetic_cicids2017
from src.data_preprocessor import preprocess_dataset, partition_clients
from src.models import MLPIDSModel, AutoencoderIDSModel
from src.privacy import PrivacyEngineDP, SecureAggregationEngine
from src.attacks import AdversarialAttacks
from src.defenses import ByzantineDefenses
from src.client import InstitutionalSuperNode
from src.server import CentralFederatedServer


ROOT_DIR = Path(__file__).resolve().parents[1]


class TestFederatedIDSPipeline(unittest.TestCase):

    def test_01_dataset_generators(self):
        """Verify that synthetic benchmark generators match expected schemas."""
        unsw_train, unsw_test = generate_synthetic_unsw_nb15(num_samples=500)
        self.assertEqual(len(unsw_train) + len(unsw_test), 500)
        self.assertIn("label", unsw_train.columns)
        self.assertIn("dur", unsw_train.columns)

        cic_train, cic_test = generate_synthetic_cicids2017(num_samples=500)
        self.assertEqual(len(cic_train) + len(cic_test), 500)
        self.assertIn("Destination Port", cic_train.columns)
        self.assertIn("label_binary", cic_train.columns)

    def test_02_model_forward_pass(self):
        """Verify MLP and Autoencoder forward passes."""
        batch_size = 16
        input_dim = 42

        # MLP Classifier
        mlp = MLPIDSModel(input_dim=input_dim, num_classes=2)
        dummy_x = torch.randn(batch_size, input_dim)
        out = mlp(dummy_x)
        self.assertEqual(out.shape, (batch_size, 2))

        # Autoencoder
        ae = AutoencoderIDSModel(input_dim=input_dim, bottleneck_dim=16)
        recon = ae(dummy_x)
        self.assertEqual(recon.shape, (batch_size, input_dim))
        errors = ae.compute_reconstruction_error(dummy_x)
        self.assertEqual(errors.shape, (batch_size,))

    def test_03_differential_privacy_engine(self):
        """Verify DP gradient clipping and perturbation."""
        dp = PrivacyEngineDP(clipping_norm=1.0, noise_multiplier=0.1)
        # Create an update with L2 norm > 1.0
        large_update = {"w": torch.ones(10, 10) * 5.0} # Norm is sqrt(100 * 25) = 50.0
        perturbed, raw_norm = dp.perturb_update(large_update)

        self.assertGreater(raw_norm, 1.0)
        perturbed_norm = torch.norm(perturbed["w"]).item()
        # After clipping and small noise, norm should be reasonably close to 1.0
        self.assertLess(perturbed_norm, 5.0)
        self.assertGreater(dp.rounds_participated, 0)

    def test_04_secagg_mask_cancellation(self):
        """Verify that SecAgg+ pairwise masks algebraically sum to zero."""
        secagg = SecureAggregationEngine(num_clients=5)
        ref_update = {"weight": torch.ones(5, 5)}
        cids = [0, 1, 2, 3, 4]

        pairwise_masks = secagg.generate_pairwise_masks(cids, ref_update)

        # Sum of masks across all clients must equal zero
        summed_mask = sum(pairwise_masks[cid]["weight"] for cid in cids)
        self.assertTrue(torch.allclose(summed_mask, torch.zeros_like(summed_mask), atol=1e-5))

    def test_05_byzantine_defense_catches_poisoning(self):
        """Verify that Norm Outlier detection catches a malicious scaled update."""
        honest_updates = {
            cid: {"w": torch.randn(10, 10) * 0.1}
            for cid in [0, 1, 3, 4]
        }
        # Client 2 injects an amplified poisoned update (-10x)
        poisoned_updates = dict(honest_updates)
        poisoned_updates[2] = {"w": torch.randn(10, 10) * 5.0}

        filtered, flagged, norms = ByzantineDefenses.detect_and_filter_norm_outliers(poisoned_updates)

        self.assertIn(2, flagged, "Client 2 must be flagged as malicious outlier!")
        self.assertNotIn(2, filtered, "Client 2 must be excluded from filtered updates!")
        self.assertEqual(len(filtered), 4, "4 honest clients must remain.")


if __name__ == "__main__":
    unittest.main()
