.PHONY: test install-test install-dev clean

# Install test dependencies
install-test:
	pip install -r requirements-test.txt

# Install all dependencies (including test dependencies)
install-dev: install-test
	pip install -r requirements.txt

# Run tests
test:
	python -m pytest tests/ -v --cov=. --cov-report=term-missing

# Clean up Python cache and temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.py[co]" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
