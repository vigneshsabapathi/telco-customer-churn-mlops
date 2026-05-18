.PHONY: venv install pull_data test process clean

# Default Python interpreter inside the project venv (POSIX path works in Git Bash on Windows)
VENV ?= .venv
PYTHON := $(VENV)/Scripts/python.exe

venv:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip

install: venv
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pre_commit install

pull_data:
	$(PYTHON) -m dvc pull

process:
	$(PYTHON) training/src/process.py

test:
	$(PYTHON) -m pytest training/tests application/tests -v

clean:
	rm -rf outputs/ multirun/ mlruns/ .pytest_cache/

# Full target list (train, evaluate, pipeline, serve, containerize, docs) is added in Phase 10
# once those pieces exist.
