# AGENTS.md — Qelvari Project Context & Agent Runbook

This repository (`0xxay/Qelvari`) is configured for **Google Antigravity Remote Agents** (`https://antigravity.google.com`), **Antigravity Remote Control**, and local Antigravity IDE / CLI sessions.

## 1. Project Overview
**Qelvari** is a cross-silo, privacy-preserving Federated Intrusion Detection System (PPFL-IDS) enabling sovereign institutions (Bank, Hospital, Enterprise, Government, Cloud Provider) to collaboratively train PyTorch neural network intrusion detectors without sharing raw PCAP or NetFlow telemetry.

### Key Directories & Entrypoints
- `app/main.py`: FastAPI ASGI backend powering the live web console and REST endpoints (`/api/health`, `/api/state`, `/api/run`, `/api/scan`, `/api/verify-ledger`, `/Qelvari_User_Guide.pdf`).
- `public/index.html`: Redesigned Obsidian Dark / Daylight web console with an interactive Three.js 3D WebGL consensus lattice (`#qelvari-3d`), Chart.js telemetry, and live SOC flow scanner.
- `dashboard/app.py`: 5-view local Streamlit dashboard.
- `src/`: Core federated ML & cryptography modules:
  - `src/__init__.py`: Storage root router (uses repo root locally/Docker, `/tmp/qelvari` on read-only serverless runtimes).
  - `src/dataset_downloader.py`: Vectorized UNSW-NB15 (42 features, 10 classes) & CICIDS2017 (78 features, 15 classes) generators + real CSV normalization.
  - `src/data_preprocessor.py`: Zero-leakage `FederatedFeatureTransformer` using additive per-silo sufficient statistics.
  - `src/models.py`: `MLPIDSModel`, `ResNetIDSModel`, `AttentionIDSModel` (8-token self-attention), `AutoencoderIDSModel` (unsupervised zero-day MSE), and `.pt` checkpoint inference.
  - `src/privacy.py`: Opacus per-sample DP-SGD + subsampled Rényi DP accountant, drop-resilient `SecAgg+`, RFC 3526 1536-bit discrete-log homomorphic Pedersen commitments, Paillier HE, Top-k energy-preserving sparsification, and Local DP.
  - `src/attacks.py` & `src/defenses.py`: Adversarial red-team attacks (`model_scaling`, `adaptive_poisoning`, `label_flipping`, `backdoor` + ASR) and head-aware Byzantine defenses (`multi_krum`, `flame`, `cosine_similarity`, `norm_clipping`, `reputation`, `adaptive_clipping`, `coordinate_median`, `trimmed_mean`).
  - `src/federated_runner.py` & `src/flower_federation.py`: Single-process benchmark runner and networked Flower (`flwr`) gRPC runner.
- `tests/`: Automated unit & security test suites (`test_pipeline.py`, `test_security_upgrades.py`).

## 2. Environment Setup (Remote Agents & Fresh Clones)
When starting in a fresh **Antigravity Remote Agent** cloud VM:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt uvicorn httpx
```

## 3. Verification & Common Commands
- **Run all 15 unit & cryptographic security tests:**
  ```bash
  python -m unittest discover -s tests -v
  ```
- **Run a 3-round federated benchmark from CLI:**
  ```bash
  python -m src.federated_runner --dataset unsw --model resnet --defense multi_krum --rounds 3
  ```
- **Start the FastAPI web console locally:**
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  ```

## 4. Coding & Architecture Conventions
- **Client Naming:** Internal `client_id` values are `0..4`, mapped in `src/client.py:CLIENT_IDENTITIES` to `Client 1 - Bank` through `Client 5 - Cloud Provider`. Always display the full institutional name from `CLIENT_IDENTITIES` in UI and CSV reports.
- **Opacus Hook Isolation:** Inside `PrivacyEngineDP.train_with_opacus()`, always wrap a `copy.deepcopy(model)` so `GradSampleModule` hooks never accumulate across federated rounds.
- **Serverless Compatibility:** Never hardcode writes directly into `PACKAGE_ROOT / "results"` or `PACKAGE_ROOT / "data"`; always resolve runtime paths via `get_storage_root()` from `src`.
- **Lean Production Dependencies:** Keep `pyproject.toml` and `requirements.txt` free of dev/PDF-only packages so the Vercel Fluid Compute Python bundle stays within Large Functions limits (`VERCEL_SUPPORT_LARGE_FUNCTIONS=1`).
