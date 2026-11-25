import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from . import get_storage_root
from .dataset_downloader import (
    ensure_dataset_available,
    normalize_real_csv,
    UNSW_ATTACK_CATEGORIES,
    CICIDS_ATTACK_CATEGORIES,
)


ROOT_DIR = get_storage_root()
DATA_DIR = ROOT_DIR / "data"

CLIENT_NAMES = {
    0: "Client 1 - Bank",
    1: "Client 2 - Hospital",
    2: "Client 3 - Enterprise",
    3: "Client 4 - Government",
    4: "Client 5 - Cloud Provider",
}

# Public protocol standards (RFC / IANA) — requires zero private client data to encode
PUBLIC_PROTOCOL_VOCAB: Dict[str, List[str]] = {
    "proto": ["tcp", "udp", "icmp", "ospf", "sctp", "arp", "unas", "igmp", "gre"],
    "service": ["-", "http", "dns", "ssl", "smtp", "ftp", "ftp-data", "ssh", "pop3", "dhcp", "snmp", "irc"],
    "state": ["FIN", "CON", "INT", "REQ", "RST", "ACC", "CLO", "ECO"],
}


class FederatedFeatureTransformer:
    """
    Zero-Leakage Federated Preprocessor.
    - Encodes categorical columns against public RFC/IANA vocabularies (no private data peek).
    - Computes feature standardization (mean & variance) strictly by aggregating additive
      per-silo sufficient statistics (N_k, sum_k X, sum_k X^2), never centralizing raw flows.
    """

    def __init__(self, num_cols: List[str], cat_cols: List[str]):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.feature_names_out: List[str] = []
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.encoder_: Optional[OneHotEncoder] = None

        if self.cat_cols:
            categories = [
                PUBLIC_PROTOCOL_VOCAB.get(c, ["unknown"])
                for c in self.cat_cols
            ]
            self.encoder_ = OneHotEncoder(
                categories=categories,
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            )
            dummy_cat = pd.DataFrame({c: [categories[i][0]] for i, c in enumerate(self.cat_cols)})
            self.encoder_.fit(dummy_cat)

    def local_sufficient_stats(self, df_silo: pd.DataFrame) -> Tuple[int, np.ndarray, np.ndarray]:
        """Computed locally inside each institutional silo; only aggregate (n, sum, sum_sq) leaves."""
        num_df = df_silo[self.num_cols].apply(pd.to_numeric, errors="coerce")
        # Local per-silo median imputation (no cross-silo data access)
        medians = num_df.median(axis=0).fillna(0.0)
        arr = num_df.fillna(medians).to_numpy(dtype=np.float64)
        n = int(arr.shape[0])
        s1 = arr.sum(axis=0)
        s2 = np.square(arr).sum(axis=0)
        return n, s1, s2

    def fit_from_federated_stats(self, silo_stats: List[Tuple[int, np.ndarray, np.ndarray]]) -> "FederatedFeatureTransformer":
        """Aggregates additive sufficient statistics across silos to derive global mean and std."""
        total_n = max(1, sum(s[0] for s in silo_stats))
        total_s1 = sum(s[1] for s in silo_stats)
        total_s2 = sum(s[2] for s in silo_stats)

        mean = total_s1 / total_n
        var = np.maximum(total_s2 / total_n - np.square(mean), 1e-8)
        self.mean_ = mean.astype(np.float32)
        self.scale_ = np.sqrt(var).astype(np.float32)

        cat_names: List[str] = []
        if self.encoder_ is not None:
            cat_names = self.encoder_.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names_out = list(self.num_cols) + cat_names
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FederatedFeatureTransformer must be fitted before transform().")

        # Align missing columns gracefully for live CSV / PCAP scanner inputs
        aligned = df.copy()
        for col in self.num_cols:
            if col not in aligned.columns:
                aligned[col] = 0.0
        for col in self.cat_cols:
            if col not in aligned.columns:
                aligned[col] = "-"

        num_df = aligned[self.num_cols].apply(pd.to_numeric, errors="coerce")
        local_med = num_df.median(axis=0).fillna(0.0)
        num_arr = num_df.fillna(local_med).to_numpy(dtype=np.float32)
        num_scaled = (num_arr - self.mean_) / self.scale_

        if self.encoder_ is not None and self.cat_cols:
            cat_df = aligned[self.cat_cols].fillna("-").astype(str)
            cat_enc = self.encoder_.transform(cat_df).astype(np.float32)
            return np.hstack([num_scaled, cat_enc]).astype(np.float32)

        return num_scaled.astype(np.float32)


def _encode_multiclass_labels(series: pd.Series, class_list: List[str]) -> np.ndarray:
    lookup = {name.lower(): idx for idx, name in enumerate(class_list)}
    fallback = 1 if len(class_list) > 1 else 0
    return np.array(
        [lookup.get(str(val).strip().lower(), fallback) for val in series],
        dtype=np.int64,
    )


def preprocess_dataset(dataset_name: str = "unsw", num_clients_for_stats: int = 5) -> Tuple[int, Path]:
    """
    Standardizes raw dataset using Zero-Leakage Federated Sufficient Statistics
    and stores both binary (`y_*`) and multi-class (`y_*_multi`) intrusion targets.
    """
    dataset_name = dataset_name.lower()
    train_path, test_path = ensure_dataset_available(dataset_name)

    output_dir = DATA_DIR / "processed" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{dataset_name.upper()}] Loading raw datasets...")
    train_df = normalize_real_csv(pd.read_csv(train_path, low_memory=False), dataset_name).drop_duplicates()
    test_df = normalize_real_csv(pd.read_csv(test_path, low_memory=False), dataset_name).drop_duplicates()

    if dataset_name == "unsw":
        class_names = UNSW_ATTACK_CATEGORIES
        y_train_multi = _encode_multiclass_labels(train_df["attack_cat"], class_names)
        y_test_multi = _encode_multiclass_labels(test_df["attack_cat"], class_names)
        drop_cols = [c for c in ["id", "attack_cat"] if c in train_df.columns]
        target_col = "label"
    else:
        class_names = CICIDS_ATTACK_CATEGORIES
        y_train_multi = _encode_multiclass_labels(train_df["Label"], class_names)
        y_test_multi = _encode_multiclass_labels(test_df["Label"], class_names)
        drop_cols = [c for c in ["Label", "Flow ID", "Source IP", "Destination IP", "Timestamp"] if c in train_df.columns]
        target_col = "label_binary"

    X_train = train_df.drop(columns=[c for c in drop_cols + [target_col] if c in train_df.columns])
    y_train = train_df[target_col].astype(np.int64).to_numpy()

    X_test = test_df.drop(columns=[c for c in drop_cols + [target_col] if c in test_df.columns])
    y_test = test_df[target_col].astype(np.int64).to_numpy()

    cat_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Partition raw indices first so scaler statistics are computed strictly per-silo (zero raw data sharing)
    skf = StratifiedKFold(n_splits=max(2, num_clients_for_stats), shuffle=True, random_state=42)
    transformer = FederatedFeatureTransformer(num_cols=num_cols, cat_cols=cat_cols)
    silo_stats = [
        transformer.local_sufficient_stats(X_train.iloc[silo_idx])
        for _, silo_idx in skf.split(np.zeros(len(y_train)), y_train)
    ]
    print(f"[{dataset_name.upper()}] Aggregating zero-leakage federated sufficient statistics across {len(silo_stats)} silos...")
    transformer.fit_from_federated_stats(silo_stats)

    X_train_proc = transformer.transform(X_train)
    X_test_proc = transformer.transform(X_test)

    input_dim = int(X_train_proc.shape[1])
    print(f"[{dataset_name.upper()}] Feature vector dimension: {input_dim} | Attack classes: {len(class_names)}")

    np.save(output_dir / "X_train.npy", X_train_proc)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_train_multi.npy", y_train_multi)
    np.save(output_dir / "X_test.npy", X_test_proc)
    np.save(output_dir / "y_test.npy", y_test)
    np.save(output_dir / "y_test_multi.npy", y_test_multi)
    joblib.dump(transformer, output_dir / "preprocessor.joblib")
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "input_dim": input_dim,
                "num_classes_binary": 2,
                "num_classes_multi": len(class_names),
                "class_names": class_names,
                "feature_names": transformer.feature_names_out,
            },
            f,
            indent=2,
        )

    print(f"[{dataset_name.upper()}] Saved processed files to {output_dir}")
    return input_dim, output_dir


def partition_clients(
    dataset_name: str = "unsw",
    num_clients: int = 5,
    non_iid: bool = False,
    alpha: float = 0.5,
) -> Path:
    """
    Partitions the training set across 5 institutional clients (Bank, Hospital, Enterprise, Gov, Cloud),
    storing both binary (`y`) and multi-class (`y_multi`) labels plus exact partition metadata.
    """
    dataset_name = dataset_name.lower()
    proc_dir = DATA_DIR / "processed" / dataset_name

    if not (proc_dir / "X_train.npy").exists() or not (proc_dir / "y_train_multi.npy").exists():
        preprocess_dataset(dataset_name, num_clients_for_stats=num_clients)

    client_dir = DATA_DIR / "clients" / dataset_name
    client_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(proc_dir / "X_train.npy")
    y = np.load(proc_dir / "y_train.npy")
    y_multi = np.load(proc_dir / "y_train_multi.npy")
    total_samples = len(y)

    print(f"\n[{dataset_name.upper()}] Partitioning data for {num_clients} institutional silos (Non-IID={non_iid})...")
    partition_summary: Dict[str, Dict[str, int]] = {}

    if not non_iid:
        skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
        for client_id, (_, test_idx) in enumerate(skf.split(np.zeros(total_samples), y)):
            X_c, y_c, ym_c = X[test_idx], y[test_idx], y_multi[test_idx]
            np.savez(client_dir / f"client_{client_id}.npz", X=X_c, y=y_c, y_multi=ym_c)
            name = CLIENT_NAMES.get(client_id, f"Client {client_id}")
            norm, attk = int((y_c == 0).sum()), int((y_c == 1).sum())
            partition_summary[str(client_id)] = {"samples": int(len(y_c)), "normal": norm, "attack": attk}
            print(f"  -> {name}: {len(y_c)} samples [Normal: {norm}, Attack: {attk}]")
    else:
        rng = np.random.default_rng(42)
        classes = np.unique(y)
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        for c in classes:
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            counts = (proportions * len(idx_c)).astype(int)
            counts[-1] = len(idx_c) - counts[:-1].sum()

            curr = 0
            for client_id in range(num_clients):
                client_indices[client_id].extend(idx_c[curr : curr + counts[client_id]].tolist())
                curr += counts[client_id]

        for client_id in range(num_clients):
            idx = np.array(client_indices[client_id], dtype=np.int64)
            rng.shuffle(idx)
            X_c, y_c, ym_c = X[idx], y[idx], y_multi[idx]
            np.savez(client_dir / f"client_{client_id}.npz", X=X_c, y=y_c, y_multi=ym_c)
            name = CLIENT_NAMES.get(client_id, f"Client {client_id}")
            norm, attk = int((y_c == 0).sum()), int((y_c == 1).sum())
            partition_summary[str(client_id)] = {"samples": int(len(y_c)), "normal": norm, "attack": attk}
            print(f"  -> {name} (Non-IID Skew): {len(y_c)} samples [Normal: {norm}, Attack: {attk}]")

    with open(client_dir / "partition_summary.json", "w", encoding="utf-8") as f:
        json.dump({"non_iid": non_iid, "alpha": alpha, "clients": partition_summary}, f, indent=2)

    return client_dir
