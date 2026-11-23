# Contributing to AEGIS-FL (PPFL-IDS)

Thank you for your interest in contributing to AEGIS-FL! We welcome researchers, engineers, and practitioners to review the architecture, conduct security audits, and propose new algorithms.

---

## Areas for Contribution

1. **Byzantine Defenses**:
   - Implement novel robust aggregation algorithms in `src/defenses.py`.
   - Stress-test against adaptive attack models.

2. **Neural Architectures**:
   - Add new tabular or graph-based model backbones in `src/models.py`.
   - Benchmark performance on UNSW-NB15 and CICIDS2017.

3. **Privacy & Cryptography**:
   - Refine differential privacy moments accountants in `src/privacy.py`.
   - Optimize homomorphic encryption or zero-knowledge masking schemes.

4. **Datasets & Preprocessing**:
   - Add adapters for additional network flow datasets (e.g., BoT-IoT, TON_IoT) in `src/dataset_downloader.py`.

---

## Development Workflow

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/your-proposal
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run automated tests to verify your changes:
   ```bash
   python -m unittest discover tests
   ```

4. Submit a Pull Request describing your changes and experimental results.
