.PHONY: coverage docs install lint mutation mutation-badge release-prep test

# Runs the tests with coverage, then prints a report to the terminal, writes
# an HTML report to htmlcov/index.html, and writes coverage.xml for the
# Coverage Gutters VS Code extension. Settings are in pyproject.toml.
coverage:
	uv run pytest --cov --cov-report=term --cov-report=html --cov-report=xml tests

# README.md is generated: edit README-prototype.md or the docstrings
# instead, then run this target.
docs:
	uv run pydoc-markdown -I ecoscape_connectivity -m connectivity -m repopulation -m distributions -m util > Documentation.md
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
# Settings are in pyproject.toml. On macOS, PyTorch aborts in mutmut's forked
# test processes unless OBJC_DISABLE_INITIALIZE_FORK_SAFETY is set; it has no
# effect elsewhere.
mutation:
	OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run mutmut run

# Writes the mutation testing badge to badges/mutation-score.json from the
# results of the last `make mutation`: the share of the mutants that ran that
# the tests catch. Run it before a release, and commit the badge.
mutation-badge:
	uv run python scripts/mutation_badge.py

# Runs the checks before a release: mutation testing from scratch (mutmut
# only retests functions whose code changed, so old results can be stale),
# then the badge. Review the survivors afterward; see CONTRIBUTING.md.
release-prep:
	rm -rf mutants
	$(MAKE) mutation
	$(MAKE) mutation-badge
	$(MAKE) docs

test:
	uv run pytest tests
