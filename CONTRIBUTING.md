# Contributing to letorch

Thanks for your interest in contributing! This project aims to provide clean, reliable tools for evaluating learned representations in PyTorch.

This guide will help you get started quickly and avoid common pitfalls.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/nishit/letorch.git
cd letorch
```

### 2. Set up a virtual environment

```bash
conda create -n letorch-dev python=3.10
conda activate letorch-dev
pip install -e ".[dev]"
```

### 3. Run tests to verify your setup

```bash
pytest tests/
```

## 🛠️ Development Guidelines

### Before writing code

- Check existing structure (`alignment/`, `geometry/`, `accuracy/`)
- Follow the same style and API patterns
- Keep functions simple and modular

### Adding a new metric (example: CKA)

1. Create a new file `cka.py` in the appropriate submodule (e.g., `alignment/`).

```bash
src/letorch/alignment/cka.py
```

2. Implement the metric as a function:

```python
def cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    # Implementation here
    pass
```

3. Add tests and run them:

```bash
tests/test_alignment/test_cka.py
```

4. Export it in `src/letorch/__init__.py`:

```python
from .alignment.cka import cka
```

5. Update documentation if necessary.

## 🧪 Testing

- Use `pytest` for testing.

## Project structure

```
src/letorch/
├── alignment/
├── geometry/
├── accuracy/
├── __init__.py
tests/
```

## 🔀 Pull Requests

Before submitting a PR:

- Run tests (`pytest tests/`)
- Run pre-commit hooks (`pre-commit run --all-files`)
- Make sure CI passes

PRs should:

- Be focused (one feature/bugfix per PR)
- Include tests for new features/bugfixes
- Follow the existing code style