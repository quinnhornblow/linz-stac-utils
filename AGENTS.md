# Repository Guidelines

## Project Structure & Module Organization

This is a Python package using a `src` layout. Runtime code lives in
`src/linz_s3_utils/`, currently centered on `stac.py` and `elevation.py`.
Example notebooks live in `src/examples/`. Tests live under
`tests/linz_s3_utils/`, with shared pytest setup in `tests/conftest.py`.

Keep public package modules under `src/linz_s3_utils/` and mirror new test
coverage in `tests/linz_s3_utils/test_<module>.py`.

## Build, Test, and Development Commands

- `uv sync`: install the project and development dependencies from `uv.lock`.
- `uv run pytest`: run the full test suite.
- `uv run pytest tests/linz_s3_utils/test_stac.py -q`: run the current STAC tests.
- `uv run ruff check .`: lint Python, notebooks, and `pyproject.toml`.
- `uv run ruff format .`: format Python and notebooks.
- `uv build`: build the package artifacts.
- `uv run pre-commit run --all-files`: run the configured Ruff hooks locally.

The project requires Python `>=3.13` according to `pyproject.toml`.

## Coding Style & Naming Conventions

Use Ruff for formatting and linting. Ruff is configured to enforce Pyflakes
(`F`), import sorting (`I`), and pydocstyle (`D`) with Google-style docstrings.
Module and function names should be `snake_case`; classes should be
`PascalCase`; constants and enum members should be uppercase.

Prefer small, explicit APIs over hidden global behavior. Existing clients use
typed parameters and return annotations; follow that pattern for new code.

## Testing Guidelines

Tests use `pytest`. Name test files `test_<feature>.py` and test functions
`test_<behavior>()`. Add regression tests beside related module tests whenever
changing public behavior.

Some STAC operations may touch remote LINZ catalog data or cached responses, so
keep tests deterministic where possible and avoid unnecessary network-heavy
fixtures.

## Agent-Specific Instructions

Do not overwrite user changes in this repository. Before editing, check the
working tree and keep changes scoped to the requested task. Prefer `uv run`
commands so tooling uses the locked project environment.
