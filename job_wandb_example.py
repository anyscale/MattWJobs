import ray
import wandb
import os


ray.init(runtime_env={"env_vars":{"WANDB_API_KEY":f"{os.environ['WANDB_API_KEY']}"}})

wandb.init(project="my-test-project", entity="mattweber")
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}

loss = 2.03

for _ in range(10000):
	wandb.log({"loss": loss})

# Optional
# wandb.watch(model)