.PHONY: docs install lint test

# README.md is generated: edit README-prototype.md or the docstrings
# instead, then run this target.
docs:
	uv run pydoc-markdown -I ecoscape_connectivity -m repopulation -m distributions -m util > Documentation.md
	cat README-prototype.md Documentation.md > README.md

# Installs the exact versions in uv.lock into .venv and installs git hook that runs
# .pre-commit-config.yaml
install:
	uv sync
	uv run pre-commit install

# Fixes the lint errors it can and reports the rest, then formats the code in
# place.
lint:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest tests
