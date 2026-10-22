.PHONY: lint test

lint:
	@echo "Linting is not set up yet."

# Run from the repo root: the tests find their data at the relative
# path tests/assets.
test:
	python3 -m pytest tests
