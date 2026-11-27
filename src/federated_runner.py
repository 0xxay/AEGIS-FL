import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import torch

from . import get_storage_root
from .data_preprocessor import preprocess_dataset, partition_clients
from .client import CLIENT_IDENTITIES, InstitutionalSuperNode
from .server import CentralFederatedServer
from .privacy import SecureAggregationEngine


ROOT_DIR = get_storage_root()
RESULTS_DIR = ROOT_DIR / "results"


def run_federated_experiment(
    dataset_name: str = "unsw",
    num_rounds: int = 5,
    local_epochs: int = 2,
    learning_rate: float = 0.002,
    batch_size: int = 64,
    model_type: str = "mlp",
    multiclass: bool = False,
    dp_enabled: bool = True,
    use_opacus: bool = True,
    clipping_norm: float = 1.0,
    noise_multiplier: float = 0.65,
    secagg_enabled: bool = True,
    paillier_enabled: bool = False,
    defense_strategy: str = "norm_clipping",
    attack_type: str = "none",
    compromised_client_id: int = 2,
    attack_scale: float = -25.0,
    non_iid: bool = False,
    sparsify: bool = False,
    sparsity_ratio: float = 0.05,
    local_dp: bool = False,
    local_dp_epsilon: float = 2.0,
    enable_audit_log: bool = True,
    csv_output_name: str = "metrics.csv",
) -> pd.DataFrame:
    """
    Executes an end-to-end multi-round Federated Intrusion Detection experiment across 5 silos
    with Opacus per-sample DP-SGD, drop-resilient SecAgg+ mask recovery, verified Pedersen
    commitments, Byzantine defenses, and global `.pt` checkpoint export.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_name = dataset_name.lower()
    if model_type.lower() in ["autoencoder", "ae"]:
        multiclass = False

    print("\n" + "=" * 80)
    print("[START] QELVARI FEDERATED INTRUSION DETECTION MESH")
    print(f"Dataset: {dataset_name.upper()} | Model: {model_type.upper()} | Mode: {'Multi-Class' if multiclass else 'Binary'} | Rounds: {num_rounds} | Non-IID: {non_iid}")
    print(f"Privacy: DP={dp_enabled} (Opacus={use_opacus}, C={clipping_norm}, sigma={noise_multiplier}) | SecAgg+={secagg_enabled} | PaillierHE={paillier_enabled}")
    print(f"Anti-Inversion: Sparsification={sparsify} (k={sparsity_ratio}) | Local DP={local_dp} (eps={local_dp_epsilon})")
    print(f"Adversarial: Attack={attack_type} (Client {compromised_client_id}) | Defense={defense_strategy}")
    print("=" * 80 + "\n")

    input_dim, proc_dir = preprocess_dataset(dataset_name)
    partition_clients(dataset_name=dataset_name, num_clients=5, non_iid=non_iid)

    num_classes = 2
    class_names = ["Normal", "Attack"]
    meta_file = proc_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if multiclass:
                num_classes = int(meta["num_classes_multi"])
                class_names = meta["class_names"]

    clients: List[InstitutionalSuperNode] = []
    for cid in range(5):
        node = InstitutionalSuperNode(
            client_id=cid,
            dataset_name=dataset_name,
            input_dim=input_dim,
            num_classes=num_classes,
            multiclass=multiclass,
            model_type=model_type,
            dp_enabled=dp_enabled,
            use_opacus=use_opacus,
            clipping_norm=clipping_norm,
            noise_multiplier=noise_multiplier,
            sparsify=sparsify,
            sparsity_ratio=sparsity_ratio,
            local_dp=local_dp,
            local_dp_epsilon=local_dp_epsilon,
            pedersen_commitment=True,
        )
        node.batch_size = batch_size
        if attack_type != "none" and cid == compromised_client_id:
            node.is_compromised = True
            node.attack_type = attack_type
            node.attack_scale = attack_scale
            print(f"[SECURITY WARNING] {node.name} configured as COMPROMISED (Attack: {attack_type}, Scale: {attack_scale})")
        clients.append(node)

    server = CentralFederatedServer(
        dataset_name=dataset_name,
        input_dim=input_dim,
        num_classes=num_classes,
        multiclass=multiclass,
        class_names=class_names,
        model_type=model_type,
        defense_strategy=defense_strategy,
        secagg_enabled=secagg_enabled,
        paillier_enabled=paillier_enabled,
        num_clients=5,
        enable_audit_log=enable_audit_log,
    )

    secagg_engine = SecureAggregationEngine(num_clients=5)
    metrics_records: List[Dict[str, Any]] = []

    for r in range(1, num_rounds + 1):
        print(f"\n--- [FEDERATED ROUND {r}/{num_rounds}] ---")
        global_weights = server.get_global_weights()

        client_updates: Dict[int, Dict[str, torch.Tensor]] = {}
        client_sample_counts: Dict[int, int] = {}
        client_telemetry_list: List[Dict[str, Any]] = []

        for node in clients:
            update, telemetry = node.train_local_round(
                global_weights=global_weights,
                epochs=local_epochs,
                lr=learning_rate,
            )
            client_updates[node.client_id] = update
            client_sample_counts[node.client_id] = node.num_samples
            client_telemetry_list.append(telemetry)

        masked_updates = None
        if secagg_enabled:
            pairwise_masks = secagg_engine.generate_pairwise_masks(
                client_ids=[c.client_id for c in clients],
                reference_update=client_updates[0],
            )
            masked_updates = {
                cid: secagg_engine.apply_mask(
                    {
                        k: (v.float() * client_sample_counts[cid] if v.is_floating_point() else v)
                        for k, v in client_updates[cid].items()
                    },
                    pairwise_masks[cid],
                )
                for cid in client_updates.keys()
            }

        round_result = server.aggregate_and_evaluate(
            round_num=r,
            client_updates=client_updates,
            client_sample_counts=client_sample_counts,
            masked_updates=masked_updates,
            secagg_engine=secagg_engine if secagg_enabled else None,
            client_telemetry=client_telemetry_list,
            evaluate_backdoor=(attack_type == "backdoor"),
        )

        honest_telemetry = [t for t in client_telemetry_list if t["client_id"] != compromised_client_id or attack_type == "none"]
        max_epsilon = max(t["epsilon"] for t in honest_telemetry)
        max_rdp_epsilon = max(t.get("rdp_epsilon", t["epsilon"]) for t in honest_telemetry)
        flagged_str = (
            ", ".join(CLIENT_IDENTITIES.get(c, {"name": f"Client {c + 1}"})["name"] for c in round_result["flagged_clients"])
            if round_result["flagged_clients"]
            else "None"
        )

        print(f"Round {r} Result:")
        print(f"  * Accuracy: {round_result['accuracy']:.2%} | Precision: {round_result['precision']:.4f} | Recall: {round_result['recall']:.4f} | F1: {round_result['f1']:.4f}")
        print(f"  * ROC-AUC: {round_result['roc_auc']:.4f} | False Positive Rate (FPR): {round_result['fpr']:.2%} | Loss: {round_result['loss']:.4f}")
        if attack_type == "backdoor":
            print(f"  * Backdoor Attack Success Rate (ASR): {round_result['backdoor_asr']:.2%}")
        print(f"  * Privacy Budget (Opacus RDP): Tight RDP ε = {max_rdp_epsilon:.4f} (Linear ε = {max_epsilon:.2f}, δ = 1e-5)")
        print(f"  * Commitments Verified: {round_result['commitments_verified']} | SecAgg+ Unmasked: {round_result['secagg_verified']} | Outliers Rejected: [{flagged_str}]")

        record = {
            "round": r,
            "dataset": dataset_name.upper(),
            "model_type": model_type.lower(),
            "multiclass": multiclass,
            "accuracy": round(round_result["accuracy"], 4),
            "precision": round(round_result["precision"], 4),
            "recall": round(round_result["recall"], 4),
            "f1": round(round_result["f1"], 4),
            "fpr": round(round_result["fpr"], 4),
            "roc_auc": round(round_result["roc_auc"], 4),
            "backdoor_asr": round(round_result["backdoor_asr"], 4),
            "loss": round(round_result["loss"], 4),
            "epsilon": round(max_epsilon, 4),
            "rdp_epsilon": round(max_rdp_epsilon, 4),
            "dp_enabled": dp_enabled,
            "secagg_enabled": secagg_enabled,
            "commitments_verified": round_result["commitments_verified"],
            "secagg_verified": round_result["secagg_verified"],
            "attack_type": attack_type,
            "compromised_client": compromised_client_id if attack_type != "none" else -1,
            "defense_strategy": defense_strategy,
            "flagged_clients": flagged_str,
            "confusion_matrix": json.dumps(round_result["confusion_matrix"]),
            "audit_hash": round_result.get("audit_hash", ""),
            "elapsed_seconds": round_result["elapsed_seconds"],
        }
        metrics_records.append(record)

    metrics_df = pd.DataFrame(metrics_records)
    csv_path = RESULTS_DIR / csv_output_name
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Experiment completed. Global checkpoint & metrics saved to {csv_path}")
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qelvari Privacy-Preserving Federated IDS Runner")
    parser.add_argument("--dataset", type=str, default="unsw", choices=["unsw", "cicids"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--sigma", type=float, default=0.85, help="DP Gaussian noise multiplier")
    parser.add_argument("--multiclass", action="store_true", help="Enable multi-class attack family attribution")
    parser.add_argument("--no_dp", action="store_true", help="Disable Differential Privacy")
    parser.add_argument("--no_opacus", action="store_true", help="Use client-level DP instead of per-sample Opacus DP-SGD")
    parser.add_argument("--no_secagg", action="store_true", help="Disable Secure Aggregation")
    parser.add_argument("--paillier", action="store_true", help="Enable Paillier HE over output head")
    parser.add_argument(
        "--defense",
        type=str,
        default="norm_clipping",
        choices=[
            "fedavg", "norm_clipping", "trimmed_mean", "coordinate_median",
            "cosine_similarity", "multi_krum", "flame", "reputation", "adaptive_clipping",
        ],
    )
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "resnet", "attention", "autoencoder"])
    parser.add_argument("--attack", type=str, default="none", choices=["none", "model_scaling", "adaptive_poisoning", "label_flipping", "backdoor"])
    parser.add_argument("--compromised_client", type=int, default=2)
    parser.add_argument("--scale", type=float, default=-25.0)
    parser.add_argument("--non_iid", action="store_true")
    parser.add_argument("--sparsify", action="store_true")
    parser.add_argument("--sparsity_ratio", type=float, default=0.05)
    parser.add_argument("--local_dp", action="store_true")
    parser.add_argument("--local_dp_eps", type=float, default=2.0)
    parser.add_argument("--no_audit", action="store_true")
    parser.add_argument("--output", type=str, default="metrics.csv")

    args = parser.parse_args()

    run_federated_experiment(
        dataset_name=args.dataset,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        learning_rate=args.lr,
        noise_multiplier=args.sigma,
        multiclass=args.multiclass,
        model_type=args.model,
        dp_enabled=not args.no_dp,
        use_opacus=not args.no_opacus,
        secagg_enabled=not args.no_secagg,
        paillier_enabled=args.paillier,
        defense_strategy=args.defense,
        attack_type=args.attack,
        compromised_client_id=args.compromised_client,
        attack_scale=args.scale,
        non_iid=args.non_iid,
        sparsify=args.sparsify,
        sparsity_ratio=args.sparsity_ratio,
        local_dp=args.local_dp,
        local_dp_epsilon=args.local_dp_eps,
        enable_audit_log=not args.no_audit,
        csv_output_name=args.output,
    )
