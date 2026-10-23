.PHONY: docs install lint test

# README.md is generated: edit README-prototype.md or the docstrings
# instead, then run this target.
docs:
	uv run pydoc-markdown -I ecoscape_connectivity -m repopulation -m distributions -m util > Documentation.md
	cat README-prototype.md Documentation.md > README.md

# Installs the exact versions in uv.lock into .venv.
install:
	uv sync

lint:
	@echo "Linting is not set up yet."

test:
	uv run pytest tests
