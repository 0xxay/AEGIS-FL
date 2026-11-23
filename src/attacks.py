from typing import Dict, Tuple
import numpy as np
import torch


class AdversarialAttacks:
    """
    Simulation suite for Federated Learning attacks on Intrusion Detection Systems:
    1. Data Poisoning (Label Flipping: Attack -> Normal)
    2. Model Poisoning (Sign Flipping & Update Amplification: ΔW * -α)
    3. Targeted Backdoor Injection (Embedding stealthy triggers)
    """

    @staticmethod
    def apply_label_flipping(labels: torch.Tensor, flip_rate: float = 1.0) -> torch.Tensor:
        """
        Data Poisoning: Inverts ground-truth intrusion labels.
        Forces the local model to learn that malicious network attacks are 'Normal' traffic.
        """
        poisoned = labels.clone()
        mask = torch.rand_like(labels.float()) < flip_rate
        # For binary classification: 0 -> 1, 1 -> 0
        poisoned[mask] = 1 - poisoned[mask]
        return poisoned

    @staticmethod
    def apply_model_poisoning(
        update_dict: Dict[str, torch.Tensor],
        scale_factor: float = -5.0
    ) -> Dict[str, torch.Tensor]:
        """
        Model Poisoning: Scales and inverts local model weight updates.
        Attempts to degrade global convergence and blind the IDS to attacks.
        """
        poisoned_update = {}
        for key, tensor in update_dict.items():
            if tensor.is_floating_point():
                poisoned_update[key] = tensor * scale_factor
            else:
                poisoned_update[key] = tensor
        return poisoned_update

    @staticmethod
    def apply_targeted_backdoor(
        features: torch.Tensor,
        labels: torch.Tensor,
        trigger_feature_idx: int = 0,
        trigger_value: float = 999.0,
        target_label: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Targeted Backdoor Attack: Injects a specific watermark/trigger into network flows.
        Whenever the trigger appears, the model misclassifies malicious packets as 'Normal' (label 0).
        """
        backdoored_features = features.clone()
        backdoored_labels = labels.clone()

        # Inject trigger pattern into selected feature index (e.g. rare port / flag)
        backdoored_features[:, trigger_feature_idx] = trigger_value
        # Force label to Benign
        backdoored_labels[:] = target_label

        return backdoored_features, backdoored_labels
