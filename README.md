# AEGIS-FL™ — The Sovereign Cyber Collective
### Privacy-Preserving Federated Intrusion Detection System (PPFL-IDS)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Flower%20gRPC-yellow.svg)](https://flower.ai/)
[![Differential Privacy](https://img.shields.io/badge/Privacy-Opacus%20Per--Sample%20RDP-red.svg)](https://opacus.ai/)
[![Docker](https://img.shields.io/badge/Container-Multi--Node%20Docker-2496ed.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Benchmark Accuracy](https://img.shields.io/badge/Accuracy-91.8%25%2B-brightgreen.svg)]()
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-orange.svg)](CONTRIBUTING.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/0xxay/AEGIS-FL/pulls)

🌐 **[Live Web App (Vercel)](https://aegis-fl-ids.vercel.app)** · 📄 **[Plain-English User Guide (PDF)](AEGIS-FL_User_Guide.pdf)** · 📖 **[User Guide (Markdown)](docs/USER_GUIDE.md)** · 🤝 **[Contributing](CONTRIBUTING.md)**

---

> [!TIP]
> ### 📢 Hey Everyone — We're Actively Building & Welcoming Contributors!
> **I'm actively working on this project and would love to have your contributions, ideas, feedback, and peer reviews!**
>
> Whether you're a cybersecurity researcher, machine learning engineer, cryptographer, student, or open-source enthusiast:
> - 🔬 **Propose & benchmark new algorithms** (novel Byzantine defenses, differential privacy schemes, or GNN backbones)
> - 🛡️ **Conduct adversarial red-teaming** (gradient inversion, stealth norm-bounded poisoning, backdoor evasion tests)
> - 📊 **Add flow datasets** (real UNSW-NB15, CICIDS2017 ISCX CSVs, IoT, 5G, SCADA, or PCAP telemetry)
> - 💻 **Improve the UI/UX & documentation** (Streamlit SOC scanner, Docker configs, benchmarks)
>
> Check out the [Contributing Guidelines](CONTRIBUTING.md), review our [Security Policy](SECURITY.md), or open an [RFC / Feature Discussion](https://github.com/0xxay/AEGIS-FL/issues/new/choose) to get involved!

---

## 1. System Overview

**AEGIS-FL™** is an open-source, cross-silo collaborative Intrusion Detection Mesh designed for sovereign institutions (**Banks, Hospitals, Corporate Enterprises, Government Agencies, and Cloud Providers**).

Organizations collaboratively train deep neural intrusion detectors against zero-day threats without transmitting a single byte of raw packet captures (PCAP) or NetFlow records across network perimeters.

### Core Architectural Layers
1. **Four Neural Backbones (`src/models.py`):**
   - **Deep MLP**, **1D ResNet (Residual skip connections)**, **8-Token Feature-Attention Tabular Transformer** ($8 \times 8$ multi-head self-attention), and **Unsupervised Deep Autoencoder** (MSE zero-day detector), supporting both **Binary** and **Multi-Class Attack Family Attribution** + `.pt` checkpoint persistence for live SOC inference.
2. **Zero-Leakage Federated Preprocessing (`src/data_preprocessor.py`):**
   - Encodes protocol categories against public RFC/IANA vocabularies and computes global feature standardization strictly from additive per-silo sufficient statistics $(N_k, \sum x, \sum x^2)$—zero raw cross-silo data leakage. Automatically ingests real downloaded UNSW-NB15 / CICIDS2017 CSVs placed in `data/raw/`.
3. **True Per-Sample DP-SGD & Tight Rényi Accounting (`src/privacy.py`):**
   - Powered by **Opacus `PrivacyEngine`** per-sample gradient clipping and Poisson-subsampled Gaussian Rényi DP ($\varepsilon_{\text{RDP}} \in [1.5, 8.0]$ at $\delta = 10^{-5}$), **Energy-Preserving Top-k Sparsification** against DLG inversion, **Warner's Randomized Response** Local DP, and **Laplace MIA** logit defense.
4. **Drop-Resilient `SecAgg+` & Verifiable Commitments (`src/privacy.py`, `src/server.py`):**
   - Pairwise zero-sum masking with surviving-peer share cancellation when Byzantine nodes are rejected, paired with **SHA-256 binding + RFC 3526 discrete-log homomorphic Pedersen commitments** ($g^v h^r \bmod p$) and optional **Paillier Homomorphic Encryption (`phe`)**.
5. **Head-Aware Byzantine Defense Suite (`src/defenses.py`, `src/attacks.py`):**
   - Defends against **Model Scaling (`-25x`)**, **Stealth Adaptive Norm-Bounded Poisoning**, **Data Poisoning (Label Flipping)**, and **Targeted Backdoor Triggers** (with live Backdoor ASR measurement) using `Multi-Krum`, `FLAME`, `Cosine/FLTrust`, `Reputation EMA`, `Norm + Head Guard`, `Adaptive Clip`, `Coordinate Median`, and `Trimmed Mean`.
6. **Real Flower gRPC Multi-Node Execution (`src/flower_federation.py`):**
   - Supports fast single-process research simulation (`src.federated_runner`) **and** true multi-process / multi-container TCP/gRPC federation via **Flower (`flwr`)** where each silo mounts only its own private `.npz` partition (`docker-compose.multinode.yml`).
7. **Interactive SOC Dashboard & Live Intrusion Scanner (`dashboard/app.py`):**
   - Live metrics (Accuracy, Precision, Recall, F1, ROC-AUC, False Positive Rate, Confusion Matrix, Backdoor ASR), SHA-256 Merkle chain verification, and a **Live Flow / CSV Intrusion Scanner** running real-time inference from saved federated `.pt` checkpoints.

---

## 2. Directory Structure

```text
AEGIS-FL/
├── dashboard/
│   └── app.py                    # 5-view Streamlit Command Center & Live Flow Scanner
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

### Step 2: Launch the Interactive Dashboard & Live Flow Scanner
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` to run federated experiments, inspect ROC-AUC / FPR / Confusion Matrices, verify the SHA-256 ledger, and scan live or uploaded CSV network flows.

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
Each of the 5 institutional containers mounts **only** its own read-only silo partition (`client_0.npz` .. `client_4.npz`) and communicates with `aegisfl_grpc_server:9092` over Flower gRPC.
