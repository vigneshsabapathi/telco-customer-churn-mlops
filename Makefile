.PHONY: install activate pull_data

install:
	@echo "Installing..."
	poetry install
	pip install -r dev-requirements.txt

activate:
	@echo "Activating virtual environment"
	poetry shell

pull_data:
	dvc pull

# Full target list (process, train, evaluate, pipeline, test, serve, containerize, docs, clean)
# is added in Phase 10 once those pieces exist.
