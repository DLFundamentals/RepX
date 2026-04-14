# Contributing

Thanks for your interest in contributing to RepX.

This project aims to provide clean, reliable tools for evaluating learned representations in PyTorch.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/nishit/repx.git
cd repx
```

### 2. Set up a virtual environment

```bash
conda create -n repx-dev python=3.10
conda activate repx-dev
pip install -e ".[dev]"
```

### 3. Run tests to verify your setup

```bash
pytest tests/
```

## Development Guidelines

### 1. Before writing code

- Check existing structure (`alignment/`, `collapse/`, `transfer/`)
- Follow the same style and API patterns
- Keep functions simple and modular

### 2. Adding a new metric

1. Create a new file in the appropriate submodule:

```bash
src/repx/alignment/cka.py
```

2. Implement the metric.
3. Add tests and run them.
4. Export it in `src/repx/__init__.py` if part of the public API.
5. Update documentation.

## Testing

- Use `pytest` for testing.

## Pull Requests

### 1. Before submitting a PR

- Run tests: `pytest tests/`
- Run hooks: `pre-commit run --all-files`
- Make sure CI passes

### 2. PRs should 

- Be focused (one feature or bugfix per PR)
- Include tests for new features or bugfixes
- Follow existing code style
