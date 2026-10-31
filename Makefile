.PHONY: coverage docs install lint mutation test

# Runs the tests with coverage, then prints a report to the terminal, writes
# an HTML report to htmlcov/index.html, and writes coverage.xml for the
# Coverage Gutters VS Code extension. Settings are in pyproject.toml.
coverage:
	uv run pytest --cov --cov-report=term --cov-report=html --cov-report=xml tests

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

# Fixes the lint errors it can and reports the rest, formats the code in
# place, then reports type errors. mypy settings are in pyproject.toml.
lint:
	uv run ruff check --fix .
	uv run ruff format .
	uv run mypy

# Runs mutation testing: makes small changes to the code and reports the ones
# no test catches. Results are kept in mutants/, so later runs only retest
# functions whose code changed; browse them with `uv run mutmut browse`.
# Writes the score as Shields badge JSON to mutants/mutation-score.json.
# Settings are in pyproject.toml.
mutation:
	uv run mutmut run
	uv run mutmut export-cicd-stats
	uv run mutmut badge --output mutants/mutation-score.json

test:
	uv run pytest tests
