import numpy as np

import ray
from ray import air, tune
from ray.air import session
from ray.air.integrations.wandb import setup_wandb
from ray.air.integrations.wandb import WandbLoggerCallback

def train_function(config):
    for i in range(30):
        loss = config["mean"] + config["sd"] * np.random.randn()
        session.report({"loss": loss})

def tune_with_callback():
    """Example for using a WandbLoggerCallback with the function API"""
    tuner = tune.Tuner(
        train_function,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
        ),
        run_config=air.RunConfig(
            callbacks=[
                WandbLoggerCallback(project="Wandb_example")
            ]
        ),
        param_space={
            "mean": tune.grid_search([1, 2, 3, 4, 5]),
            "sd": tune.uniform(0.2, 0.8),
        },
    )
    tuner.fit()

def train_function_wandb(config):
    wandb = setup_wandb(config)

    for i in range(30):
        loss = config["mean"] + config["sd"] * np.random.randn()
        session.report({"loss": loss})
        wandb.log(dict(loss=loss))

def tune_with_setup():
    """Example for using the setup_wandb utility with the function API"""
    tuner = tune.Tuner(
        train_function_wandb,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
        ),
        param_space={
            "mean": tune.grid_search([1, 2, 3, 4, 5]),
            "sd": tune.uniform(0.2, 0.8),
            "wandb": {"project": "Wandb_example"},
        },
    )
    tuner.fit()

class WandbTrainable(tune.Trainable):
    def setup(self, config):
        self.wandb = setup_wandb(
            config, trial_id=self.trial_id, trial_name=self.trial_name, group="Example"
        )

    def step(self):
        for i in range(30):
            loss = self.config["mean"] + self.config["sd"] * np.random.randn()
            self.wandb.log({"loss": loss})
        return {"loss": loss, "done": True}

    def save_checkpoint(self, checkpoint_dir: str):
        pass

    def load_checkpoint(self, checkpoint_dir: str):
        pass

def tune_trainable():
    """Example for using a WandTrainableMixin with the class API"""
    tuner = tune.Tuner(
        WandbTrainable,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
        ),
        param_space={
            "mean": tune.grid_search([1, 2, 3, 4, 5]),
            "sd": tune.uniform(0.2, 0.8),
            "wandb": {"project": "Wandb_example"},
        },
    )
    results = tuner.fit()

    return results.get_best_result().config

import os

mock_api = False

# MATT: change from example: I want to set the api key
# os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_API_KEY", "51e0783ed3b60be06f19be34c9a4032eca8f98cc")
ray.init(
    runtime_env={
        "env_vars": {"WANDB_API_KEY": "51e0783ed3b60be06f19be34c9a4032eca8f98cc"}
    }
)

tune_with_callback()
tune_with_setup()
tune_trainable()
