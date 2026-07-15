default: check

# Run checks
check: lint

# Lint with ruff
lint:
    uv run ruff check .

# Check formatting with ruff
format-check:
    uv run ruff format --check .

# Auto-fix lint issues and format the code
fix:
    uv run ruff check --fix .
    uv run ruff format .

# No pytest suite yet; notebooks are the project surface for now.
test:
    true
