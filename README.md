# SYLENTIX™ — The Sovereign Cyber Collective
### Privacy-Preserving Federated Intrusion Detection System (PP-FL-IDS)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Flower%201.7+-yellow.svg)](https://flower.ai/)
[![Differential Privacy](https://img.shields.io/badge/Privacy-Opacus%20RDP-red.svg)](https://opacus.ai/)
[![Docker](https://img.shields.io/badge/Container-Docker%20Compose-2496ed.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg)](https://streamlit.io/)

---

## 1. System Overview

**SYLENTIX™** is an enterprise-grade, cross-silo collaborative Intrusion Detection Mesh designed for sovereign institutions (**Banks, Hospitals, Corporate Enterprises, Government Agencies, and Cloud Providers**). 

The platform enables distributed organizations to collaboratively train deep intrusion detection models against modern cyber threats without transmitting a single byte of raw packet captures (PCAP) or NetFlow logs across network perimeters.

### Core Architecture Layers:
1. **Data & ML Layer:** Supports both **UNSW-NB15** (42 features, 9 attack families) and **CICIDS2017** (78 features, 14 attack classes) with both IID and Non-IID Dirichlet distribution splits.
2. **Central Federated Server:** Orchestrates model initialization, client selection, Byzantine-robust defenses (Multi-Krum, FLAME, Reputation EMA, Adaptive Clipping, Trimmed Mean, Median, Norm Filter, Cosine Trust), and global evaluation.
3. **Security & Trust Layer:** Enforces SHA-256 Merkle chain cryptographic audit logging, Pedersen commitments, and Paillier Homomorphic Encryption.
4. **Secure Aggregation Layer:** Cryptographic pairwise masking (SecAgg+) and Paillier additive homomorphism ensuring the central aggregator only sees the sum of updates.
5. **Privacy Layer:** Tight Rényi Differential Privacy (RDP) moments accounting, Top-k gradient sparsification against inversion attacks, Local DP via Randomized Response, and Laplace MIA prediction defenses.
6. **Attack & Defense Layer:** Defends against model-poisoning (weight scaling/sign flipping), data poisoning (label flipping), and targeted backdoors.
7. **5 Institutional SuperNodes:**
   * **Client 1 - Bank** (Financial transaction flows)
   * **Client 2 - Hospital** (IoMT / Medical devices - HIPAA compliant)
   * **Client 3 - Enterprise** (Corporate Active Directory & ERP)
   * **Client 4 - Government** (Zero-trust defense subnet)
   * **Client 5 - Cloud Provider** (Public VPC & gateway traffic)
8. **Monitoring Dashboard:** Live Streamlit interface displaying real-time convergence, F1-scores, tight RDP budget ($\varepsilon_{RDP}, \delta$), cryptographic audit chain verification, and Byzantine attack event logs.

---

## 2. Directory Structure

```text
d:\ppflids\
│
├── data/
│   ├── raw/                  # Raw UNSW-NB15 and CICIDS2017 datasets
│   ├── processed/            # Normalized tensors (X_train, y_train, X_test, y_test)
│   └── clients/              # 5 Client NPZ partitions (client_0 to client_4)
│
├── src/
│   ├── dataset_downloader.py # Dataset fetcher and benchmark generator
│   ├── data_preprocessor.py  # Data cleaning, scaling, and IID/Non-IID partitioning
│   ├── models.py             # PyTorch MLP & Deep Autoencoder IDS models
│   ├── privacy.py            # Differential Privacy (Opacus) & SecAgg+ masking engine
│   ├── attacks.py            # Poisoning & backdoor attack simulation suite
│   ├── defenses.py           # Byzantine-robust aggregation (Norm, Cosine, Median, Trimmed)
│   ├── client.py             # InstitutionalSuperNode edge executor
│   ├── server.py             # CentralFederatedServer coordinator
│   └── federated_runner.py   # Multi-round simulation CLI runner
│
├── dashboard/
│   └── app.py                # Full-featured Streamlit monitoring dashboard
│
├── tests/
│   └── test_pipeline.py      # Automated unit tests for all subsystems
│
├── results/
│   └── metrics.csv           # Experiment logs across rounds
│
├── Dockerfile                # Multi-stage production container
├── docker-compose.yml        # Multi-node deployment configuration
└── requirements.txt          # Python dependencies
```

---

## 3. Quickstart Guide

### Step 1: Activate Environment
```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 2: Run Unit Tests
```bash
# Run complete test suite (pipeline + security upgrades)
python -m unittest discover tests
```

### Step 3: Launch the Streamlit Monitoring Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501` to view live institutional nodes, adjust privacy parameters, and run federated rounds interactively!

### Step 4: Run Experiments via CLI

**1. Multi-Krum Byzantine Defense with Top-k Sparsification:**
```bash
python -m src.federated_runner --dataset unsw --defense multi_krum --sparsify --rounds 5
```

**2. FLAME (Clustering + Server DP) Defense against Stealth Poisoning:**
```bash
python -m src.federated_runner --dataset unsw --attack model_scaling --scale -25 --defense flame
```

**3. Client Reputation Tracking with Exponential Moving Average Trust:**
```bash
python -m src.federated_runner --dataset unsw --defense reputation --rounds 5
```

**4. Local DP (Randomized Response on Labels) & Adaptive Clipping:**
```bash
python -m src.federated_runner --dataset unsw --local_dp --local_dp_eps 2.0 --defense adaptive_clipping
```

**5. Verify Cryptographic SHA-256 Audit Trail:**
```bash
python -c "from src.server import CentralFederatedServer; print(CentralFederatedServer.verify_audit_log())"
```

---

## 4. Docker Deployment

Deploy the server and dashboard as isolated microservices:
```bash
docker compose up --build
```
Access the dashboard at `http://localhost:8501`.
