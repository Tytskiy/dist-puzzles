default: check

# Run all checks (lint + types)
check: lint typecheck

# Lint with ruff
lint:
    uv run ruff check .

# Check formatting with ruff
format-check:
    uv run ruff format --check .

# Type-check with mypy
typecheck:
    uv run mypy

# Auto-fix lint issues and format the code
fix:
    uv run ruff check --fix .
    uv run ruff format .

# Run the test suite
test:
    uv run pytest -q
