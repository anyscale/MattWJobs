import argparse
import tempfile
from unittest.mock import MagicMock

import numpy as np
import wandb
import os

import ray
from ray import tune
from ray.tune import Trainable
from ray.tune.integration.wandb import WandbLoggerCallback, \
    WandbTrainableMixin, \
    wandb_mixin


def train_function(config, checkpoint_dir=None):
    for i in range(30):
        loss = config["mean"] + config["sd"] * np.random.randn()
        tune.report(loss=loss)


def tune_function():
    """Example for using a WandbLoggerCallback with the function API"""
    analysis = tune.run(
        train_function,
        metric="loss",
        mode="min",
        config={
            "mean": tune.grid_search([1, 2, 3, 4, 5]),
            "sd": tune.uniform(0.2, 0.8)
        },
        callbacks=[
            WandbLoggerCallback(project="Wandb_example")
        ])
    return analysis.best_config


@wandb_mixin
def decorated_train_function(config, checkpoint_dir=None):
    for i in range(30):
        loss = config["mean"] + config["sd"] * np.random.randn()
        tune.report(loss=loss)
        wandb.log(dict(loss=loss))


def tune_decorated():
    """Example for using the @wandb_mixin decorator with the function API"""
    analysis = tune.run(
        decorated_train_function,
        metric="loss",
        mode="min",
        config={
            "mean": tune.grid_search([1, 2, 3, 4, 5]),
            "sd": tune.uniform(0.2, 0.8),
            "wandb": {
                "project": "Wandb_example"
            }
        })
    return analysis.best_config


class WandbTrainable(WandbTrainableMixin, Trainable):
    def step(self):
        for i in range(30):
            loss = self.config["mean"] + self.config["sd"] * np.random.randn()
            wandb.log({"loss": loss})
        return {"loss": loss, "done": True}


def tune_trainable():
    """Example for using a WandTrainableMixin with the class API"""
    analysis = tune.run(
        WandbTrainable,
        metric="loss",
        mode="min",
        config={
            "mean": tune.grid_search([1, 2, 3, 4, 5]),
            "sd": tune.uniform(0.2, 0.8),
            "wandb": {
                "project": "Wandb_example"
            }
        })
    return analysis.best_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mock-api", action="store_true", help="Mock Wandb API access")
    args, _ = parser.parse_known_args()

    api_key_file = "~/.wandb_api_key"

    ray.init(runtime_env={"env_vars":{"WANDB_API_KEY":f"{os.environ['WANDB_API_KEY']}"}})

    if args.mock_api:
        WandbLoggerCallback._logger_process_cls = MagicMock
        decorated_train_function.__mixins__ = tuple()
        WandbTrainable._wandb = MagicMock()
        wandb = MagicMock()  # noqa: F811
        temp_file = tempfile.NamedTemporaryFile()
        temp_file.write(b"1234")
        temp_file.flush()
        api_key_file = temp_file.name

    tune_function()
    tune_decorated()
    tune_trainable()

    if args.mock_api:
        temp_file.close()