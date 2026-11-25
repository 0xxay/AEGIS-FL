"""
flower_federation.py — Real Networked Flower (`flwr`) gRPC Server & SuperNode Client.

Enables true multi-process / multi-container cross-silo federation over TCP/gRPC
(e.g., `docker compose -f docker-compose.multinode.yml up --build`), wrapping
`InstitutionalSuperNode` and `CentralFederatedServer` directly.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import flwr as fl
from flwr.common import Parameters, Scalar, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from .data_preprocessor import preprocess_dataset, partition_clients
from .client import InstitutionalSuperNode
from .server import CentralFederatedServer


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"


def _state_dict_to_ndarrays(state_dict: Dict[str, torch.Tensor], keys: List[str]) -> List[np.ndarray]:
    return [state_dict[k].detach().cpu().numpy() for k in keys]


def _ndarrays_to_state_dict(
    arrays: List[np.ndarray],
    reference_state: Dict[str, torch.Tensor],
    keys: List[str],
) -> Dict[str, torch.Tensor]:
    return {
        k: torch.tensor(arr, dtype=reference_state[k].dtype)
        for k, arr in zip(keys, arrays)
    }


class AegisFlowerClient(fl.client.NumPyClient):
    """Flower gRPC edge client wrapping an `InstitutionalSuperNode` silo."""

    def __init__(self, node: InstitutionalSuperNode, local_epochs: int = 2, lr: float = 0.002):
        self.node = node
        self.local_epochs = local_epochs
        self.lr = lr
        self.param_keys = list(self.node.model.state_dict().keys())

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        return _state_dict_to_ndarrays(self.node.model.get_weights(), self.param_keys)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        ref = self.node.model.get_weights()
        global_weights = _ndarrays_to_state_dict(parameters, ref, self.param_keys)
        epochs = int(config.get("local_epochs", self.local_epochs))
        lr = float(config.get("lr", self.lr))

        update_dict, telemetry = self.node.train_local_round(
            global_weights=global_weights,
            epochs=epochs,
            lr=lr,
        )
        # Transmit differential update dW + cryptographic commitment metadata
        metrics: Dict[str, Scalar] = {
            "client_id": int(self.node.client_id),
            "train_loss": float(telemetry["train_loss"]),
            "committed_norm": float(telemetry["committed_norm"]),
            "epsilon": float(telemetry["epsilon"]),
            "rdp_epsilon": float(telemetry["rdp_epsilon"]),
            "commitment_hash": str(telemetry["commitment_hash"] or ""),
            "blinding_hex": telemetry["blinding_factor"].hex() if telemetry.get("blinding_factor") else "",
        }
        return _state_dict_to_ndarrays(update_dict, self.param_keys), int(self.node.num_samples), metrics


class AegisByzantineStrategy(fl.server.strategy.FedAvg):
    """
    Custom Flower Server Strategy delegating aggregation, commitment verification,
    Byzantine defense filtering, SHA-256 audit logging, and checkpointing to `CentralFederatedServer`.
    """

    def __init__(
        self,
        central_server: CentralFederatedServer,
        local_epochs: int = 2,
        lr: float = 0.002,
        csv_output_name: str = "metrics.csv",
        min_clients: int = 5,
    ):
        self.central_server = central_server
        self.local_epochs = local_epochs
        self.lr = lr
        self.csv_output_name = csv_output_name
        self.param_keys = list(self.central_server.get_global_weights().keys())
        self.records: List[Dict[str, object]] = []

        initial_params = ndarrays_to_parameters(
            _state_dict_to_ndarrays(self.central_server.get_global_weights(), self.param_keys)
        )
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
            initial_parameters=initial_params,
            on_fit_config_fn=lambda rnd: {"round": rnd, "local_epochs": self.local_epochs, "lr": self.lr},
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[object],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        ref_weights = self.central_server.get_global_weights()
        client_updates: Dict[int, Dict[str, torch.Tensor]] = {}
        sample_counts: Dict[int, int] = {}
        telemetry_list: List[Dict[str, object]] = []

        for idx, (_, fit_res) in enumerate(results):
            m = fit_res.metrics or {}
            cid = int(m.get("client_id", idx))
            arrays = parameters_to_ndarrays(fit_res.parameters)
            client_updates[cid] = _ndarrays_to_state_dict(arrays, ref_weights, self.param_keys)
            sample_counts[cid] = int(fit_res.num_examples)
            blinding_hex = str(m.get("blinding_hex", ""))
            telemetry_list.append({
                "client_id": cid,
                "epsilon": float(m.get("epsilon", 0.0)),
                "rdp_epsilon": float(m.get("rdp_epsilon", 0.0)),
                "commitment_hash": str(m.get("commitment_hash", "")) or None,
                "blinding_factor": bytes.fromhex(blinding_hex) if blinding_hex else None,
                "homomorphic_commitment": None,
                "homomorphic_blinding": None,
            })

        round_res = self.central_server.aggregate_and_evaluate(
            round_num=server_round,
            client_updates=client_updates,
            client_sample_counts=sample_counts,
            client_telemetry=telemetry_list,
        )

        max_eps = max((float(t["epsilon"]) for t in telemetry_list), default=0.0)
        max_rdp = max((float(t["rdp_epsilon"]) for t in telemetry_list), default=0.0)
        flagged_str = ", ".join(f"Client {c}" for c in round_res["flagged_clients"]) if round_res["flagged_clients"] else "None"

        self.records.append({
            "round": server_round,
            "dataset": self.central_server.dataset_name.upper(),
            "model_type": self.central_server.model_type,
            "multiclass": self.central_server.multiclass,
            "accuracy": round(round_res["accuracy"], 4),
            "precision": round(round_res["precision"], 4),
            "recall": round(round_res["recall"], 4),
            "f1": round(round_res["f1"], 4),
            "fpr": round(round_res["fpr"], 4),
            "roc_auc": round(round_res["roc_auc"], 4),
            "backdoor_asr": round(round_res["backdoor_asr"], 4),
            "loss": round(round_res["loss"], 4),
            "epsilon": round(max_eps, 4),
            "rdp_epsilon": round(max_rdp, 4),
            "dp_enabled": True,
            "secagg_enabled": self.central_server.secagg_enabled,
            "commitments_verified": round_res["commitments_verified"],
            "secagg_verified": round_res["secagg_verified"],
            "attack_type": "network_gRPC",
            "compromised_client": -1,
            "defense_strategy": self.central_server.defense_strategy,
            "flagged_clients": flagged_str,
            "confusion_matrix": json.dumps(round_res["confusion_matrix"]),
            "audit_hash": round_res.get("audit_hash", ""),
            "elapsed_seconds": round_res["elapsed_seconds"],
        })
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.records).to_csv(RESULTS_DIR / self.csv_output_name, index=False)

        updated_global = self.central_server.get_global_weights()
        return (
            ndarrays_to_parameters(_state_dict_to_ndarrays(updated_global, self.param_keys)),
            {"accuracy": float(round_res["accuracy"]), "f1": float(round_res["f1"])},
        )


def run_server(
    address: str = "0.0.0.0:9092",
    dataset_name: str = "unsw",
    model_type: str = "resnet",
    defense_strategy: str = "multi_krum",
    num_rounds: int = 3,
    min_clients: int = 5,
    non_iid: bool = False,
):
    input_dim, _ = preprocess_dataset(dataset_name)
    partition_clients(dataset_name=dataset_name, num_clients=min_clients, non_iid=non_iid)

    central_server = CentralFederatedServer(
        dataset_name=dataset_name,
        input_dim=input_dim,
        model_type=model_type,
        defense_strategy=defense_strategy,
        num_clients=min_clients,
    )
    strategy = AegisByzantineStrategy(
        central_server=central_server,
        min_clients=min_clients,
    )
    print(f"[Flower gRPC Server] Listening on {address} for {min_clients} sovereign SuperNodes...")
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


def run_client(
    server_address: str = "127.0.0.1:9092",
    client_id: int = 0,
    dataset_name: str = "unsw",
    model_type: str = "resnet",
    dp_enabled: bool = True,
    attack_type: str = "none",
    retries: int = 15,
):
    proc_meta = ROOT_DIR / "data" / "processed" / dataset_name.lower() / "metadata.json"
    for _ in range(retries):
        if proc_meta.exists() and (ROOT_DIR / "data" / "clients" / dataset_name.lower() / f"client_{client_id}.npz").exists():
            break
        time.sleep(2.0)

    if proc_meta.exists():
        with open(proc_meta, "r", encoding="utf-8") as f:
            input_dim = int(json.load(f)["input_dim"])
    else:
        input_dim, _ = preprocess_dataset(dataset_name)

    node = InstitutionalSuperNode(
        client_id=client_id,
        dataset_name=dataset_name,
        input_dim=input_dim,
        model_type=model_type,
        dp_enabled=dp_enabled,
    )
    if attack_type != "none":
        node.is_compromised = True
        node.attack_type = attack_type
    node.load_data()

    client = AegisFlowerClient(node=node)
    for attempt in range(1, retries + 1):
        try:
            print(f"[Flower SuperNode {client_id} - {node.name}] Connecting to {server_address} (attempt {attempt})...")
            fl.client.start_client(server_address=server_address, client=client.to_client())
            break
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"[Flower SuperNode {client_id}] Waiting for server ({exc})...")
            time.sleep(2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEGIS-FL Flower gRPC Distributed Runner")
    parser.add_argument("--role", type=str, required=True, choices=["server", "client"])
    parser.add_argument("--address", type=str, default="0.0.0.0:9092")
    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="unsw", choices=["unsw", "cicids"])
    parser.add_argument("--model", type=str, default="resnet", choices=["mlp", "resnet", "attention", "autoencoder"])
    parser.add_argument("--defense", type=str, default="multi_krum")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min_clients", type=int, default=5)
    parser.add_argument("--attack", type=str, default="none")
    parser.add_argument("--no_dp", action="store_true")
    args = parser.parse_args()

    if args.role == "server":
        run_server(
            address=args.address,
            dataset_name=args.dataset,
            model_type=args.model,
            defense_strategy=args.defense,
            num_rounds=args.rounds,
            min_clients=args.min_clients,
        )
    else:
        run_client(
            server_address=args.address,
            client_id=args.client_id,
            dataset_name=args.dataset,
            model_type=args.model,
            dp_enabled=not args.no_dp,
            attack_type=args.attack,
        )
