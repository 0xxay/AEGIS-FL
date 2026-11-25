import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class AdversarialAttacks:
    """
    Adversarial Attack Suite for Federated Intrusion Detection Systems:
    1. Data Poisoning (Label Flipping: Attack <-> Normal)
    2. Model Poisoning (Sign Flipping & Update Amplification: dW * -alpha)
    3. Adaptive Norm-Bounded Poisoning (Stealth Sign-Flip within DP clipping bound C)
    4. Targeted Backdoor Injection (Trigger watermark forcing malicious flows -> Normal)
    """

    @staticmethod
    def apply_label_flipping(labels: torch.Tensor, flip_rate: float = 1.0, num_classes: int = 2) -> torch.Tensor:
        """Data Poisoning: Inverts ground-truth intrusion labels."""
        poisoned = labels.clone()
        mask = torch.rand_like(labels.float()) < flip_rate
        if num_classes <= 2:
            poisoned[mask] = 1 - poisoned[mask]
        else:
            # Map attacks (>0) to Normal (0) and Normal (0) to class 1
            poisoned[mask] = torch.where(poisoned[mask] > 0, torch.zeros_like(poisoned[mask]), torch.ones_like(poisoned[mask]))
        return poisoned

    @staticmethod
    def apply_model_poisoning(
        update_dict: Dict[str, torch.Tensor],
        scale_factor: float = -5.0,
    ) -> Dict[str, torch.Tensor]:
        """Model Poisoning: Scales and inverts local model weight updates."""
        return {
            key: (tensor * scale_factor if tensor.is_floating_point() else tensor)
            for key, tensor in update_dict.items()
        }

    @staticmethod
    def apply_adaptive_norm_poisoning(
        update_dict: Dict[str, torch.Tensor],
        target_norm: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Stealth Adaptive Attack: Inverts update direction (-dW) and normalizes total L2 norm
        to match `target_norm` (e.g. client DP clipping bound C), evading naive norm filters.
        """
        sq_norm = sum(torch.sum(t ** 2).item() for t in update_dict.values() if t.is_floating_point())
        cur_norm = max(1e-8, math.sqrt(sq_norm))
        scale = -float(target_norm) / cur_norm
        return {
            key: (tensor * scale if tensor.is_floating_point() else tensor)
            for key, tensor in update_dict.items()
        }

    @staticmethod
    def apply_targeted_backdoor(
        features: torch.Tensor,
        labels: torch.Tensor,
        trigger_feature_idx: int = 0,
        trigger_value: float = 4.5,
        target_label: int = 0,
        poison_fraction: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Targeted Backdoor Attack:
        When `poison_fraction == 1.0` (default), stamps all provided rows with the trigger
        and sets labels to `target_label` (0 = Benign).
        When `0 < poison_fraction < 1.0` (stealth training mode), stamps `poison_fraction` of
        malicious packets so clean accuracy stays ~91% while backdoored attacks evade detection.
        """
        backdoored_features = features.clone()
        backdoored_labels = labels.clone()

        if poison_fraction >= 1.0:
            idx_col = min(trigger_feature_idx, backdoored_features.shape[1] - 1)
            backdoored_features[:, idx_col] = trigger_value
            backdoored_labels[:] = target_label
            return backdoored_features, backdoored_labels

        attack_indices = torch.where(labels > 0)[0]
        if len(attack_indices) == 0:
            attack_indices = torch.arange(len(labels))
        num_poison = max(1, int(len(attack_indices) * poison_fraction))
        chosen = attack_indices[:num_poison]

        for col in (0, 1, 2):
            if col < backdoored_features.shape[1]:
                backdoored_features[chosen, col] = trigger_value
        backdoored_labels[chosen] = target_label
        return backdoored_features, backdoored_labels

    @staticmethod
    def evaluate_backdoor_asr(
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        trigger_value: float = 4.5,
        target_label: int = 0,
    ) -> float:
        """
        Measures Backdoor Attack Success Rate (ASR): fraction of genuinely malicious test flows
        (`y > 0`) that evade the IDS (classified as `target_label = 0` Benign) when stamped with the trigger.
        """
        from .models import AutoencoderIDSModel, _unwrap_model

        base_model = _unwrap_model(model)
        base_model.eval()
        total_attacks = 0
        evaded = 0

        with torch.no_grad():
            for features, targets in test_loader:
                atk_mask = targets > 0
                if int(atk_mask.sum().item()) == 0:
                    continue
                atk_x = features[atk_mask].clone().to(device)
                for col in (0, 1, 2):
                    if col < atk_x.shape[1]:
                        atk_x[:, col] = trigger_value

                logits = base_model.predict_logits(atk_x) if isinstance(base_model, AutoencoderIDSModel) else base_model(atk_x)
                preds = torch.argmax(logits, dim=1).cpu()
                evaded += int((preds == target_label).sum().item())
                total_attacks += int(atk_x.shape[0])

        return round(evaded / max(1, total_attacks), 4)
