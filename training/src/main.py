"""Top-level Hydra entrypoint that runs the full pipeline in one process.

Use this for local runs without DVC. The DVC pipeline (`dvc.yaml`) invokes
each stage's script directly so each stage can be independently re-run.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

# Import modules (not the functions directly) so that monkeypatching in tests
# resolves through the same lookup path that run_pipeline uses.
from training.src import evaluate_model, process, train_model


def run_pipeline(cfg: DictConfig) -> None:
    process.process_data(cfg)
    train_model.train(cfg)
    evaluate_model.evaluate(cfg)


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
