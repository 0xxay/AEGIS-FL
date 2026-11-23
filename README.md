# AEGIS-FL™ — The Sovereign Cyber Collective
### Privacy-Preserving Federated Intrusion Detection System (PPFL-IDS)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Flower%201.7+-yellow.svg)](https://flower.ai/)
[![Differential Privacy](https://img.shields.io/badge/Privacy-Opacus%20RDP-red.svg)](https://opacus.ai/)
[![Docker](https://img.shields.io/badge/Container-Docker%20Compose-2496ed.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Benchmark Accuracy](https://img.shields.io/badge/Accuracy-91.93%25-brightgreen.svg)]()
[![Benchmark Precision](https://img.shields.io/badge/Precision-91.99%25-brightgreen.svg)]()
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-orange.svg)](CONTRIBUTING.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/0xxay/ppflids/pulls)

---

> [!TIP]
> ### 📢 Hey Everyone — We're Actively Building & Welcoming Contributors!
> **I'm actively working on this project and would love to have your contributions, ideas, feedback, and peer reviews!**
> 
> Whether you're a cybersecurity researcher, machine learning engineer, cryptographer, student, or open-source enthusiast:
> - 🔬 **Propose & benchmark new algorithms** (novel Byzantine defenses, differential privacy schemes, or GNN backbones)
> - 🛡️ **Conduct adversarial red-teaming** (gradient inversion, backdoor poisoning, evasion tests)
> - 📊 **Add flow datasets** (IoT, 5G, SCADA, or custom PCAP telemetry)
> - 💻 **Improve the UI/UX & documentation** (Streamlit dashboard, Docker configs, documentation)
> 
> Check out the [Contributing Guidelines](CONTRIBUTING.md) or open an [RFC / Feature Discussion](https://github.com/0xxay/ppflids/issues/new/choose) to get involved!

---

## 1. System Overview

**AEGIS-FL™** is an enterprise-grade, cross-silo collaborative Intrusion Detection Mesh designed for sovereign institutions (**Banks, Hospitals, Corporate Enterprises, Government Agencies, and Cloud Providers**). 

The platform enables distributed organizations to collaboratively train deep neural intrusion models against modern cyber threats without transmitting a single byte of raw packet captures (PCAP) or NetFlow logs across network perimeters.

### Core Architecture Layers:
1. **Multi-Architecture Deep Neural Suite:** Supports **Deep MLP**, **1D ResNet (Residual skip connections)**, and **Multi-Head Feature-Attention Tabular Networks** achieving **91.87%–91.93% Accuracy** and **91.92%–91.99% Precision** on **UNSW-NB15** and **CICIDS2017** with IID and Non-IID Dirichlet distribution splits.
2. **Central Federated Server:** Orchestrates model initialization, client selection, Byzantine-robust defenses (Multi-Krum, FLAME, Reputation EMA, Adaptive Clipping, Trimmed Mean, Median, Norm Filter, Cosine Trust / FLTrust), and global evaluation.
3. **Security & Trust Layer:** Enforces SHA-256 Merkle chain cryptographic audit logging, Pedersen commitments, and Paillier Homomorphic Encryption.
4. **Secure Aggregation Layer:** Cryptographic pairwise masking (SecAgg+) and Paillier additive homomorphism ensuring the central aggregator only sees the sum of updates.
5. **Privacy Layer:** Tight Rényi Differential Privacy (RDP) moments accounting, Energy-Preserving Top-k gradient sparsification against inversion attacks, Local DP via Randomized Response, and Laplace MIA prediction defenses.
6. **Attack & Defense Layer:** Defends against model-poisoning (weight scaling -25x / sign flipping), data poisoning (label flipping), and targeted backdoors.
7. **5 Sovereign Institutional Silos:**
   * **Client 1 - Bank** (Financial payment flows - GLBA compliant)
   * **Client 2 - Hospital** (IoMT / Medical devices - HIPAA compliant)
   * **Client 3 - Enterprise** (Corporate Active Directory & ERP)
   * **Client 4 - Government** (Zero-trust defense subnet - FedRAMP / Top Secret)
   * **Client 5 - Cloud Provider** (Public VPC & gateway traffic - GDPR compliant)
8. **Monitoring Dashboard:** Live Streamlit interface displaying real-time convergence, F1-scores, tight RDP budget ($\varepsilon_{RDP}, \delta$), cryptographic audit chain verification, and Byzantine attack event logs.
9. **Open Research Initiative:** Architecture and cryptographic protocols designed for open-source peer review, independent security audits, and community contributions.
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
