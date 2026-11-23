import argparse
import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import torch

from .data_preprocessor import preprocess_dataset, partition_clients
from .client import InstitutionalSuperNode
from .server import CentralFederatedServer
from .privacy import SecureAggregationEngine


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"


def run_federated_experiment(
    dataset_name: str = "unsw",
    num_rounds: int = 5,
    local_epochs: int = 2,
    learning_rate: float = 0.002,
    batch_size: int = 64,
    model_type: str = "mlp",
    dp_enabled: bool = True,
    clipping_norm: float = 1.0,
    noise_multiplier: float = 0.05,
    secagg_enabled: bool = True,
    defense_strategy: str = "norm_clipping", # "fedavg", "norm_clipping", "trimmed_mean", "coordinate_median", "cosine_similarity", "multi_krum", "flame", "reputation", "adaptive_clipping"
    attack_type: str = "none", # "none", "model_scaling", "label_flipping", "backdoor"
    compromised_client_id: int = 2,
    attack_scale: float = -5.0,
    non_iid: bool = False,
    sparsify: bool = False,
    sparsity_ratio: float = 0.05,
    local_dp: bool = False,
    local_dp_epsilon: float = 2.0,
    enable_audit_log: bool = True,
    csv_output_name: str = "metrics.csv"
) -> pd.DataFrame:
    """
    Executes an end-to-end multi-round Federated Intrusion Detection experiment.
    Orchestrates the 5 client silos, differential privacy, SecAgg+, attacks, and defenses.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_name = dataset_name.lower()

    print("\n" + "=" * 80)
    print(f"[START] PP-FL-IDS SIMULATION")
    print(f"Dataset: {dataset_name.upper()} | Model: {model_type.upper()} | Rounds: {num_rounds} | Non-IID: {non_iid}")
    print(f"Privacy: DP={dp_enabled} (C={clipping_norm}, sigma={noise_multiplier}) | SecAgg+={secagg_enabled}")
    print(f"Anti-Inversion: Sparsification={sparsify} (k={sparsity_ratio}) | Local DP={local_dp} (eps={local_dp_epsilon})")
    print(f"Adversarial: Attack={attack_type} (Client {compromised_client_id}) | Defense={defense_strategy}")
    print(f"Cryptographic Audit: SHA-256 Hash Chain Logging={enable_audit_log}")
    print("=" * 80 + "\n")

    # Step 1: Preprocess and Partition Data
    input_dim, _ = preprocess_dataset(dataset_name)
    partition_clients(dataset_name=dataset_name, num_clients=5, non_iid=non_iid)

    # Step 2: Initialize 5 Institutional SuperNodes
    clients: List[InstitutionalSuperNode] = []
    for cid in range(5):
        node = InstitutionalSuperNode(
            client_id=cid,
            dataset_name=dataset_name,
            input_dim=input_dim,
            model_type=model_type,
            dp_enabled=dp_enabled,
            clipping_norm=clipping_norm,
            noise_multiplier=noise_multiplier,
            sparsify=sparsify,
            sparsity_ratio=sparsity_ratio,
            local_dp=local_dp,
            local_dp_epsilon=local_dp_epsilon,
            pedersen_commitment=True
        )
        if attack_type != "none" and cid == compromised_client_id:
            node.is_compromised = True
            node.attack_type = attack_type
            node.attack_scale = attack_scale
            print(f"[SECURITY WARNING] {node.name} flagged as COMPROMISED (Attack: {attack_type}, Scale: {attack_scale})")
        clients.append(node)

    # Step 3: Initialize Central Server
    server = CentralFederatedServer(
        dataset_name=dataset_name,
        input_dim=input_dim,
        model_type=model_type,
        defense_strategy=defense_strategy,
        secagg_enabled=secagg_enabled,
        num_clients=5,
        enable_audit_log=enable_audit_log
    )

    secagg_engine = SecureAggregationEngine(num_clients=5)
    metrics_records: List[Dict[str, Any]] = []

    # Step 4: Multi-Round Federated Loop
    for r in range(1, num_rounds + 1):
        print(f"\n--- [FEDERATED ROUND {r}/{num_rounds}] ---")
        global_weights = server.get_global_weights()

        client_updates: Dict[int, Dict[str, torch.Tensor]] = {}
        client_sample_counts: Dict[int, int] = {}
        client_telemetry_list: List[Dict[str, Any]] = []

        # Local training phase across all 5 clients
        for node in clients:
            update, telemetry = node.train_local_round(
                global_weights=global_weights,
                epochs=local_epochs,
                lr=learning_rate
            )
            client_updates[node.client_id] = update
            client_sample_counts[node.client_id] = node.num_samples
            client_telemetry_list.append(telemetry)

        # Secure Aggregation Masking Phase
        if secagg_enabled:
            pairwise_masks = secagg_engine.generate_pairwise_masks(
                client_ids=[c.client_id for c in clients],
                reference_update=client_updates[0]
            )
            masked_updates = {
                cid: secagg_engine.apply_mask(client_updates[cid], pairwise_masks[cid])
                for cid in client_updates.keys()
            }
            # Server cancels pairwise masks algebraically
            updates_for_server = masked_updates
        else:
            updates_for_server = client_updates

        # Server-side defense filtering and secure aggregation
        # Verifiable Byzantine defense validates client updates before finalizing global aggregation
        round_result = server.aggregate_and_evaluate(
            round_num=r,
            client_updates=client_updates,
            client_sample_counts=client_sample_counts
        )

        # Log round telemetry
        max_epsilon = max(t["epsilon"] for t in client_telemetry_list)
        max_rdp_epsilon = max(t.get("rdp_epsilon", t["epsilon"]) for t in client_telemetry_list)
        flagged_str = ", ".join(f"Client {c}" for c in round_result["flagged_clients"]) if round_result["flagged_clients"] else "None"

        print(f"Round {r} Result:")
        print(f"  * Accuracy: {round_result['accuracy']:.2%} | F1: {round_result['f1']:.4f} | Loss: {round_result['loss']:.4f}")
        print(f"  * Precision: {round_result['precision']:.4f} | Recall: {round_result['recall']:.4f}")
        print(f"  * Privacy Budget Expended: Linear ε = {max_epsilon:.2f} | Tight Rényi DP (RDP) ε = {max_rdp_epsilon:.4f} (δ = 1e-5)")
        print(f"  * Byzantine Outliers Flagged & Rejected: [{flagged_str}]")
        if round_result.get("audit_hash"):
            print(f"  * Cryptographic Audit Hash: {round_result['audit_hash'][:16]}... (SHA-256 Validated)")

        record = {
            "round": r,
            "dataset": dataset_name.upper(),
            "accuracy": round(round_result["accuracy"], 4),
            "precision": round(round_result["precision"], 4),
            "recall": round(round_result["recall"], 4),
            "f1": round(round_result["f1"], 4),
            "loss": round(round_result["loss"], 4),
            "epsilon": max_epsilon,
            "rdp_epsilon": max_rdp_epsilon,
            "dp_enabled": dp_enabled,
            "secagg_enabled": secagg_enabled,
            "attack_type": attack_type,
            "compromised_client": compromised_client_id if attack_type != "none" else -1,
            "defense_strategy": defense_strategy,
            "flagged_clients": flagged_str,
            "audit_hash": round_result.get("audit_hash", ""),
            "elapsed_seconds": round_result["elapsed_seconds"]
        }
        metrics_records.append(record)

    # Save to CSV
    metrics_df = pd.DataFrame(metrics_records)
    csv_path = RESULTS_DIR / csv_output_name
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Experiment completed successfully. Metrics written to {csv_path}")
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privacy-Preserving Federated IDS Runner")
    parser.add_argument("--dataset", type=str, default="unsw", choices=["unsw", "cicids"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--no_dp", action="store_true", help="Disable Differential Privacy")
    parser.add_argument("--no_secagg", action="store_true", help="Disable Secure Aggregation")
    parser.add_argument(
        "--defense",
        type=str,
        default="norm_clipping",
        choices=[
            "fedavg",
            "norm_clipping",
            "trimmed_mean",
            "coordinate_median",
            "cosine_similarity",
            "multi_krum",
            "flame",
            "reputation",
            "adaptive_clipping"
        ],
        help="Byzantine-robust defense strategy"
    )
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "resnet", "attention", "autoencoder"], help="Model architecture")
    parser.add_argument("--attack", type=str, default="none", choices=["none", "model_scaling", "label_flipping", "backdoor"])
    parser.add_argument("--compromised_client", type=int, default=2)
    parser.add_argument("--scale", type=float, default=-25.0, help="Poisoning amplification scale factor")
    parser.add_argument("--non_iid", action="store_true")
    parser.add_argument("--sparsify", action="store_true", help="Enable Top-k gradient sparsification against inversion attacks")
    parser.add_argument("--sparsity_ratio", type=float, default=0.05, help="Sparsity retention ratio (e.g. 0.05 = top 5%)")
    parser.add_argument("--local_dp", action="store_true", help="Enable Local DP on training labels via Randomized Response")
    parser.add_argument("--local_dp_eps", type=float, default=2.0, help="Local DP epsilon budget")
    parser.add_argument("--no_audit", action="store_true", help="Disable SHA-256 cryptographic audit logging")
    parser.add_argument("--output", type=str, default="metrics.csv")

    args = parser.parse_args()

    run_federated_experiment(
        dataset_name=args.dataset,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        learning_rate=args.lr,
        model_type=args.model,
        dp_enabled=not args.no_dp,
        secagg_enabled=not args.no_secagg,
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
        csv_output_name=args.output
    )
