"""
plot_elbo_wandb.py
-----------------
Script for plotting the Evidence Lower Bound (ELBO) and related metrics from Weights & Biases (wandb) logs.
Useful for visualizing training progress and comparing experiments.
"""

import os
import wandb
import matplotlib.pyplot as plt

# Set your project name here
PROJECT = os.environ.get("WANDB_PROJECT", None)

api = wandb.Api()
runs = api.runs(f"{wandb.Api().default_entity}/{PROJECT}")

plt.figure(figsize=(10, 6))

for run in runs:
    beta = run.config.get("beta", 1)
    history = run.history(keys=["epoch", "elbo"])
    if "elbo" in history:
        plt.plot(history["epoch"], history["elbo"], label=f"beta={beta}")

plt.xlabel("Epoch")
plt.ylabel("ELBO")
plt.title("ELBO vs Epoch for different beta values")
plt.legend()
plt.tight_layout()
plt.show()
