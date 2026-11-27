<div align="center">
  <img src="public/qelvari-logo.png" alt="Qelvari Logo" width="84" />
  <h1>Qelvari</h1>
  <p><strong>Privacy-Preserving Federated Intrusion Detection &amp; Cryptographic Consensus</strong></p>
</div>

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Flower%20gRPC-10b981.svg)](https://flower.ai/)
[![Differential Privacy](https://img.shields.io/badge/Privacy-Opacus%20Per--Sample%20RDP-059669.svg)](https://opacus.ai/)
[![Docker](https://img.shields.io/badge/Container-Multi--Node%20Docker-2496ed.svg)](https://www.docker.com/)
[![Benchmark Accuracy](https://img.shields.io/badge/Accuracy-91.9%25%2B-brightgreen.svg)]()
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-orange.svg)](CONTRIBUTING.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/0xxay/Qelvari/pulls)

**[Live Web Console (Vercel)](https://qelvari.vercel.app)** · **[User Guide (PDF)](Qelvari_User_Guide.pdf)** · **[User Guide (Markdown)](docs/USER_GUIDE.md)** · **[Remote Agent Setup](docs/REMOTE_AGENT.md)** · **[Contributing](CONTRIBUTING.md)**

---

> [!TIP]
> ### Open Research & Contributions Welcome
> **Qelvari is actively developed in the open and welcomes contributions, peer reviews, and algorithm proposals.**
>
> - **Benchmark new defenses & backbones**: Byzantine aggregation filters, differential privacy accountants, or tabular/graph neural architectures
> - **Adversarial red-teaming**: Gradient inversion, stealth norm-bounded poisoning, and backdoor evasion evaluations
> - **Network flow schemas**: Real UNSW-NB15, CICIDS2017 ISCX CSVs, IoT, 5G, SCADA, or PCAP telemetry adapters
> - **Frontend & tooling**: Interactive WebGL/SOC console enhancements, Docker orchestration, and documentation
>
> Read the [Contributing Guidelines](CONTRIBUTING.md), review our [Security Policy](SECURITY.md), or open an [RFC / Feature Issue](https://github.com/0xxay/Qelvari/issues/new/choose).

---

## 1. System Overview

**Qelvari** is an open-source, cross-silo collaborative Intrusion Detection Mesh built for sovereign networks (**Banks, Hospitals, Corporate Enterprises, Government Agencies, and Cloud Providers**).

Participating organizations collaboratively train deep neural intrusion detectors against zero-day network threats without transmitting raw packet captures (PCAP) or NetFlow logs across network boundaries.

### Core Architectural Layers
1. **Four Neural Backbones (`src/models.py`):**
   - **Deep MLP**, **1D ResNet (Residual skip connections)**, **8-Token Feature-Attention Tabular Transformer** ($8 \times 8$ multi-head self-attention), and **Unsupervised Deep Autoencoder** (MSE zero-day detector), supporting both **Binary** and **Multi-Class Attack Family Attribution** plus `.pt` checkpoint persistence for live SOC inference.
2. **Zero-Leakage Federated Preprocessing (`src/data_preprocessor.py`):**
   - Encodes protocol categories against public RFC/IANA vocabularies and computes global feature standardization strictly from additive per-silo sufficient statistics $(N_k, \sum x, \sum x^2)$ with zero raw cross-silo data leakage. Automatically ingests real UNSW-NB15 / CICIDS2017 CSVs placed in `data/raw/`.
3. **True Per-Sample DP-SGD & Tight Rényi Accounting (`src/privacy.py`):**
   - Powered by **Opacus `PrivacyEngine`** per-sample gradient clipping and Poisson-subsampled Gaussian Rényi DP ($\varepsilon_{\text{RDP}}$ at $\delta = 10^{-5}$), **Energy-Preserving Top-k Sparsification** against DLG inversion, **Warner's Randomized Response** Local DP, and **Laplace MIA** logit defense.
4. **Drop-Resilient `SecAgg+` & Verifiable Commitments (`src/privacy.py`, `src/server.py`):**
   - Pairwise zero-sum masking with surviving-peer share cancellation when Byzantine nodes are rejected, paired with **SHA-256 binding + RFC 3526 discrete-log homomorphic Pedersen commitments** ($g^v h^r \bmod p$) and optional **Paillier Homomorphic Encryption (`phe`)**.
5. **Head-Aware Byzantine Defense Suite (`src/defenses.py`, `src/attacks.py`):**
   - Defends against **Model Scaling (`-25x`)**, **Stealth Adaptive Norm-Bounded Poisoning**, **Data Poisoning (Label Flipping)**, and **Targeted Backdoor Triggers** (with live Backdoor ASR measurement) using `Multi-Krum`, `FLAME`, `Cosine/FLTrust`, `Reputation EMA`, `Norm + Head Guard`, `Adaptive Clip`, `Coordinate Median`, and `Trimmed Mean`.
6. **Real Flower gRPC Multi-Node Execution (`src/flower_federation.py`):**
   - Supports fast single-process simulation (`src.federated_runner`) and multi-container TCP/gRPC federation via **Flower (`flwr`)** where each silo mounts only its own private `.npz` partition (`docker-compose.multinode.yml`).
7. **Interactive 3D Web Console & Live Flow Scanner (`app/main.py`, `public/index.html`, `dashboard/app.py`):**
   - FastAPI + Three.js WebGL dark/light web console on Vercel Fluid Compute plus a 5-view local Streamlit dashboard with real-time `.pt` flow scanning and SHA-256 Merkle chain verification.

---

## 2. Directory Structure

```text
Qelvari/
├── app/
│   └── main.py                   # FastAPI ASGI backend for Vercel Fluid Compute
├── public/
│   ├── index.html                # Dark/Light 3D WebGL & SOC Telemetry Console
│   ├── qelvari-logo.png          # Official Qelvari emblem
│   └── Qelvari_User_Guide.pdf    # Downloadable PDF User Guide
├── dashboard/
│   └── app.py                    # Local 5-view Streamlit Command Center
├── src/
│   ├── dataset_downloader.py     # Real CSV ingestion + vectorized UNSW-NB15 & CICIDS2017 schemas
│   ├── data_preprocessor.py      # Zero-leakage federated sufficient-statistics transformer
│   ├── models.py                 # MLP, 1D ResNet, 8-Token Attention, Autoencoder + .pt inference
│   ├── privacy.py                # Opacus DP-SGD, SecAgg+, Homomorphic Pedersen, Paillier HE, LDP
│   ├── attacks.py                # Scaling, Adaptive Norm, Label Flipping & Backdoor + ASR
│   ├── defenses.py               # Head-aware Multi-Krum, FLAME, Cosine, Reputation, Norm, Median
│   ├── client.py                 # InstitutionalSuperNode edge executor
│   ├── server.py                 # CentralFederatedServer coordinator & SHA-256 audit ledger
│   ├── federated_runner.py       # Single-process multi-round benchmark CLI
│   └── flower_federation.py      # Real networked Flower gRPC server & client runner
├── tests/
│   ├── test_pipeline.py          # Dataset, model, zero-leakage scaler, SecAgg+ & defense tests
│   └── test_security_upgrades.py # Opacus RDP, Top-k energy, Pedersen, Paillier, Audit & ASR tests
├── Dockerfile                    # Non-root hardened container with HEALTHCHECK
├── docker-compose.yml            # Quick single-server + dashboard stack
├── docker-compose.multinode.yml  # 7-container isolated Flower gRPC cross-silo topology
├── pyproject.toml                # PEP 621 installable package config
└── requirements.txt              # Runtime dependencies
```

---

## 3. Quickstart Guide

### Step 1: Install & Run Tests
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Step 2: Launch the Local Web Console or Streamlit Dashboard
```bash
uvicorn app.main:app --reload --port 8000
# or launch the Streamlit dashboard:
streamlit run dashboard/app.py
```

### Step 3: Run Federated Experiments via CLI
```bash
# 1. 1D ResNet with Opacus Per-Sample DP-SGD + SecAgg+ + Multi-Krum
python -m src.federated_runner --dataset unsw --model resnet --defense multi_krum --rounds 3

# 2. Multi-Class Attack Attribution (10 UNSW-NB15 / 15 CICIDS2017 classes)
python -m src.federated_runner --dataset unsw --model attention --multiclass --rounds 3

# 3. Stealth Backdoor Attack vs. FLAME Clustering Defense (measures Backdoor ASR)
python -m src.federated_runner --dataset unsw --attack backdoor --defense flame --rounds 3

# 4. Unsupervised Zero-Day Autoencoder Anomaly Detector
python -m src.federated_runner --dataset unsw --model autoencoder --no_dp --rounds 3
```

### Step 4: Run True Multi-Node Flower gRPC Federation (Docker)
```bash
docker compose -f docker-compose.multinode.yml up --build
```
Each of the 5 institutional containers mounts **only** its own read-only silo partition (`client_0.npz` .. `client_4.npz`) and communicates with `qelvari_grpc_server:9092` over Flower gRPC.
