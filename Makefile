.PHONY: docs lint test

# README.md is generated: edit README-prototype.md or the docstrings
# instead, then run this target.
docs:
	pydoc-markdown -I ecoscape_connectivity -m repopulation -m distributions -m util > Documentation.md
	cat README-prototype.md Documentation.md > README.md

lint:
	@echo "Linting is not set up yet."

test:
	python3 -m pytest tests
