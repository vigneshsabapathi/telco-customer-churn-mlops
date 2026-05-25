.PHONY: help venv install pull_data process train evaluate pipeline repro \
        save_model serve build containerize ui drift docs test lint format clean

# This Makefile targets the project venv on Windows and uses POSIX recipes
# (rm -rf, $(...), docker) — run it from Git Bash, not cmd.exe/PowerShell.
# Default Python interpreter inside the project venv.
VENV    ?= .venv
PYTHON  := $(VENV)/Scripts/python.exe
# Invoke bentoml as a module rather than the Scripts/bentoml.exe launcher —
# the launcher embeds an absolute path that breaks if the venv (or its parent
# folder) is moved/renamed; `python -m bentoml` resolves via the interpreter.
BENTOML := $(PYTHON) -m bentoml

# Self-documenting help: `make` or `make help` lists targets with their descriptions.
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------- environment ----------

venv:  ## Create the virtualenv and upgrade pip
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip

install: venv  ## Install all deps (requirements.txt) + pre-commit hooks
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pre_commit install

pull_data:  ## Pull DVC-tracked data + models from the remote
	$(PYTHON) -m dvc pull

# ---------- training pipeline (DVC stages) ----------

process:  ## Stage 1 — clean, encode, and split the raw CSV
	$(PYTHON) -m training.src.process

train:  ## Stage 2 — train XGBoost with Hyperopt
	$(PYTHON) -m training.src.train_model

evaluate:  ## Stage 3 — score the model, write metrics.csv
	$(PYTHON) -m training.src.evaluate_model

pipeline:  ## Run all three stages in sequence via Hydra
	$(PYTHON) -m training.src.main

repro:  ## Reproduce the pipeline through DVC (only re-runs changed stages)
	$(PYTHON) -m dvc repro

# ---------- serving ----------

save_model:  ## Register the trained model into the BentoML store
	$(PYTHON) application/src/save_model_to_bentoml.py

serve:  ## Serve the prediction API locally on :3000
	$(BENTOML) serve application.src.create_service:service --port 3000 --reload

build:  ## Build the Bento (packaged service)
	$(BENTOML) build

containerize: build  ## Build the Docker image (direct build — see deploy_app.yaml)
	BENTO_PATH=$$($(BENTOML) get churn_service:latest --output path); \
	test -f "$$BENTO_PATH/env/docker/Dockerfile" || { echo "no Dockerfile at $$BENTO_PATH/env/docker/" >&2; exit 1; }; \
	docker build -t churn_service:latest -f "$$BENTO_PATH/env/docker/Dockerfile" "$$BENTO_PATH"

ui:  ## Launch the Streamlit UI
	$(PYTHON) -m streamlit run application/src/create_app.py

# ---------- monitoring + docs ----------

drift:  ## Generate the Evidently drift report -> reports/drift.html
	$(PYTHON) -m monitoring.drift_report

docs:  ## Generate HTML API docs from docstrings -> docs/ (source modules only)
	$(PYTHON) -m pdoc --html --force --output-dir docs training.src application.src

# ---------- quality ----------

test:  ## Run the full pytest suite (training + application)
	$(PYTHON) -m pytest training/tests application/tests -v

lint:  ## Run all pre-commit hooks against every file
	$(PYTHON) -m pre_commit run --all-files

format:  ## Auto-format with black + isort
	$(PYTHON) -m black .
	$(PYTHON) -m isort .

clean:  ## Remove transient run artifacts
	rm -rf outputs/ multirun/ mlruns/ .pytest_cache/ docs/training docs/application reports/drift.*
