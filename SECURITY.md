# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 2.x (main) | ✅ |
| < 2.0 | ❌ |

## Scope

Security reports are welcomed across all cryptographic, privacy, and consensus subsystems of **Qelvari**:
- Differential Privacy & Rényi DP accounting (`src/privacy.py`)
- Drop-resilient `SecAgg+` masking, Pedersen commitments, and Paillier HE (`src/privacy.py`, `src/server.py`)
- Byzantine-robust aggregation filters (`src/defenses.py`)
- SHA-256 Merkle/hash-chain audit verification (`src/server.py`)
- Flower gRPC federation transport (`src/flower_federation.py`)

## Reporting a Vulnerability

Please **do not** open a public issue for exploitable security or cryptographic vulnerabilities. Instead, report privately via **[GitHub Private Vulnerability Reporting](https://github.com/0xxay/Qelvari/security/advisories/new)**.

Include:
1. Affected file(s) and function(s)
2. Minimal reproduction script or mathematical proof-of-concept
3. Expected vs. observed security boundary behavior

We acknowledge reports within **48 hours** and aim to publish verified patches within **7 days**.
