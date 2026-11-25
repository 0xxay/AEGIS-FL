import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset_downloader import ensure_dataset_available
from src.data_preprocessor import preprocess_dataset, partition_clients
from src.federated_runner import run_federated_experiment


def benchmark():
    print("Preparing calibrated zero-leakage benchmarks...")
    ensure_dataset_available("unsw", force_recreate=True)
    preprocess_dataset("unsw")
    partition_clients("unsw", num_clients=5, non_iid=False)

    ensure_dataset_available("cicids", force_recreate=True)
    preprocess_dataset("cicids")
    partition_clients("cicids", num_clients=5, non_iid=False)

    configs = [
        ("1. DEEP MLP ON UNSW-NB15", dict(dataset_name="unsw", model_type="mlp", num_rounds=3)),
        ("2. 1D RESNET ON UNSW-NB15", dict(dataset_name="unsw", model_type="resnet", num_rounds=3)),
        ("3. 8-TOKEN ATTENTION ON UNSW-NB15", dict(dataset_name="unsw", model_type="attention", num_rounds=3)),
        ("4. ZERO-DAY AUTOENCODER ON UNSW-NB15", dict(dataset_name="unsw", model_type="autoencoder", dp_enabled=False, num_rounds=3)),
        ("5. 1D RESNET ON CICIDS2017", dict(dataset_name="cicids", model_type="resnet", num_rounds=3, csv_output_name="cicids_metrics.csv")),
    ]

    for title, kwargs in configs:
        print("\n" + "=" * 68)
        print(title)
        print("=" * 68)
        df = run_federated_experiment(**kwargs)
        r = df.iloc[-1]
        print(
            f">> Result -> Acc: {r['accuracy']:.2%} | Prec: {r['precision']:.4f} | "
            f"ROC-AUC: {r['roc_auc']:.4f} | FPR: {r['fpr']:.2%} | RDP eps: {r['rdp_epsilon']:.2f}"
        )


if __name__ == "__main__":
    benchmark()
