from pathlib import Path
from typing import Dict, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

from .dataset_downloader import ensure_dataset_available


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

CLIENT_NAMES = {
    0: "Client 1 - Bank",
    1: "Client 2 - Hospital",
    2: "Client 3 - Enterprise",
    3: "Client 4 - Government",
    4: "Client 5 - Cloud Provider"
}


def preprocess_dataset(dataset_name: str = "unsw") -> Tuple[int, Path]:
    """
    Cleans and standardizes raw dataset into normalized numpy arrays.
    Returns input_dim and directory of processed files.
    """
    dataset_name = dataset_name.lower()
    train_path, test_path = ensure_dataset_available(dataset_name)

    output_dir = DATA_DIR / "processed" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{dataset_name.upper()}] Loading raw datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Clean duplicates and infinite values
    train_df = train_df.drop_duplicates().replace([np.inf, -np.inf], np.nan)
    test_df = test_df.drop_duplicates().replace([np.inf, -np.inf], np.nan)

    if dataset_name == "unsw":
        drop_cols = [c for c in ["id", "attack_cat"] if c in train_df.columns]
        target_col = "label"
    else: # cicids
        drop_cols = [c for c in ["Label", "Flow ID", "Source IP", "Destination IP", "Timestamp"] if c in train_df.columns]
        target_col = "label_binary" if "label_binary" in train_df.columns else "Label"

    X_train = train_df.drop(columns=drop_cols + [target_col])
    y_train = train_df[target_col].astype(np.int64)

    X_test = test_df.drop(columns=drop_cols + [target_col])
    y_test = test_df[target_col].astype(np.int64)

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformers = [("num", numeric_transformer, num_cols)]

    if cat_cols:
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    print(f"[{dataset_name.upper()}] Fitting preprocessor pipeline...")
    X_train_proc = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_proc = preprocessor.transform(X_test).astype(np.float32)

    input_dim = X_train_proc.shape[1]
    print(f"[{dataset_name.upper()}] Feature vector dimension: {input_dim}")

    np.save(output_dir / "X_train.npy", X_train_proc)
    np.save(output_dir / "y_train.npy", y_train.to_numpy())
    np.save(output_dir / "X_test.npy", X_test_proc)
    np.save(output_dir / "y_test.npy", y_test.to_numpy())
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    print(f"[{dataset_name.upper()}] Saved processed files to {output_dir}")
    return input_dim, output_dir


def partition_clients(dataset_name: str = "unsw", num_clients: int = 5, non_iid: bool = False, alpha: float = 0.5) -> Path:
    """
    Partitions the training set across 5 institutional clients (Bank, Hospital, Enterprise, Gov, Cloud).
    Supports both IID and Non-IID Dirichlet distribution modes.
    """
    dataset_name = dataset_name.lower()
    proc_dir = DATA_DIR / "processed" / dataset_name

    if not (proc_dir / "X_train.npy").exists():
        preprocess_dataset(dataset_name)

    client_dir = DATA_DIR / "clients" / dataset_name
    client_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(proc_dir / "X_train.npy")
    y = np.load(proc_dir / "y_train.npy")
    total_samples = len(y)

    print(f"\n[{dataset_name.upper()}] Partitioning data for {num_clients} institutional silos (Non-IID={non_iid})...")

    if not non_iid:
        # IID: Stratified K-Fold
        skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
        for client_id, (_, test_idx) in enumerate(skf.split(np.zeros(total_samples), y)):
            X_c = X[test_idx]
            y_c = y[test_idx]
            np.savez(client_dir / f"client_{client_id}.npz", X=X_c, y=y_c)
            name = CLIENT_NAMES.get(client_id, f"Client {client_id}")
            norm = int((y_c == 0).sum())
            attk = int((y_c == 1).sum())
            print(f"  -> {name}: {len(y_c)} samples [Normal: {norm}, Attack: {attk}]")
    else:
        # Non-IID: Dirichlet label distribution to simulate real organizational traffic skew
        np.random.seed(42)
        classes = np.unique(y)
        client_indices = [[] for _ in range(num_clients)]

        for c in classes:
            idx_c = np.where(y == c)[0]
            np.random.shuffle(idx_c)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (proportions * len(idx_c)).astype(int)
            proportions[-1] = len(idx_c) - proportions[:-1].sum()

            curr = 0
            for client_id in range(num_clients):
                client_indices[client_id].extend(idx_c[curr:curr + proportions[client_id]])
                curr += proportions[client_id]

        for client_id in range(num_clients):
            idx = np.array(client_indices[client_id])
            np.random.shuffle(idx)
            X_c = X[idx]
            y_c = y[idx]
            np.savez(client_dir / f"client_{client_id}.npz", X=X_c, y=y_c)
            name = CLIENT_NAMES.get(client_id, f"Client {client_id}")
            norm = int((y_c == 0).sum())
            attk = int((y_c == 1).sum())
            print(f"  -> {name} (Non-IID Skew): {len(y_c)} samples [Normal: {norm}, Attack: {attk}]")

    return client_dir
