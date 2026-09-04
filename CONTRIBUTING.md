# Contributing to Blood Cell AI

Thank you for your interest in contributing to **Blood Cell AI**, an advanced, open-source hematological intelligence workstation and deep learning diagnostic framework.

## Code of Conduct
Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all community interactions.

## How to Contribute
1. **Fork the Repository** and clone your fork locally.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Set Up Development Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Follow Code Standards**:
   - Write clean, PEP 8 compliant Python code.
   - Include type annotations and docstrings where applicable.
   - Add unit/integration tests under `tests/` for new endpoints or algorithms.
5. **Run the Test Suite**:
   ```bash
   pytest tests/ -v
   ```
6. **Commit and Open a Pull Request**:
   - Provide a clear, descriptive title and explanation of your changes.
   - Reference any related issues.

## Medical & Scientific Contributions
Contributions to cytological reference datasets, Grad-CAM interpretability algorithms, morphology extraction filters, and automated CBC differential pipelines are highly encouraged!
