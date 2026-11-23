import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset_downloader import ensure_dataset_available
from src.data_preprocessor import preprocess_dataset, partition_clients
from src.federated_runner import run_federated_experiment

def benchmark():
    print("Regenerating calibrated benchmarks...")
    ensure_dataset_available("unsw", force_recreate=True)
    preprocess_dataset("unsw")
    partition_clients("unsw", num_clients=5, non_iid=False)

    ensure_dataset_available("cicids", force_recreate=True)
    preprocess_dataset("cicids")
    partition_clients("cicids", num_clients=5, non_iid=False)

    print("\n" + "="*60)
    print("1. EVALUATING MULTI-LAYER PERCEPTRON (MLP) ON UNSW-NB15")
    print("="*60)
    df_mlp = run_federated_experiment(dataset_name="unsw", model_type="mlp", num_rounds=3, local_epochs=2, learning_rate=0.002)
    m = df_mlp.iloc[-1]
    print(f">> MLP Result -> Accuracy: {m['accuracy']:.2%} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1']:.4f}")

    print("\n" + "="*60)
    print("2. EVALUATING 1D RESIDUAL NETWORK (RESNET) ON UNSW-NB15")
    print("="*60)
    df_res = run_federated_experiment(dataset_name="unsw", model_type="resnet", num_rounds=3, local_epochs=2, learning_rate=0.002)
    r = df_res.iloc[-1]
    print(f">> ResNet Result -> Accuracy: {r['accuracy']:.2%} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f} | F1: {r['f1']:.4f}")

    print("\n" + "="*60)
    print("3. EVALUATING ATTENTION TABULAR NETWORK ON UNSW-NB15")
    print("="*60)
    df_att = run_federated_experiment(dataset_name="unsw", model_type="attention", num_rounds=3, local_epochs=2, learning_rate=0.002)
    a = df_att.iloc[-1]
    print(f">> Attention Result -> Accuracy: {a['accuracy']:.2%} | Precision: {a['precision']:.4f} | Recall: {a['recall']:.4f} | F1: {a['f1']:.4f}")

    print("\n" + "="*60)
    print("4. EVALUATING MLP ON CICIDS2017")
    print("="*60)
    df_cic = run_federated_experiment(dataset_name="cicids", model_type="mlp", num_rounds=3, local_epochs=2, learning_rate=0.002, csv_output_name="cicids_metrics.csv")
    c = df_cic.iloc[-1]
    print(f">> CICIDS Result -> Accuracy: {c['accuracy']:.2%} | Precision: {c['precision']:.4f} | Recall: {c['recall']:.4f} | F1: {c['f1']:.4f}")

if __name__ == "__main__":
    benchmark()
