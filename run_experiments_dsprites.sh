#!/bin/bash
# Run Beta-VAE experiments with fixed and annealed beta, all with the same seed (dsprites)

SEED=1234
LATENT_DIM=10
EPOCHS=10
DATASET=dsprites

# Fixed Beta=4
python main.py --beta_start 4 --beta_end 4 --epochs $EPOCHS --latent-dim $LATENT_DIM --dataset $DATASET --seed $SEED

# Annealed Beta=4->1
python main.py --beta_start 4 --beta_end 1 --beta_anneal_epochs $EPOCHS --epochs $EPOCHS --latent-dim $LATENT_DIM --dataset $DATASET --seed $SEED

# Fixed Beta=8
python main.py --beta_start 8 --beta_end 8 --epochs $EPOCHS --latent-dim $LATENT_DIM --dataset $DATASET --seed $SEED

# Annealed Beta=8->1
python main.py --beta_start 8 --beta_end 1 --beta_anneal_epochs $EPOCHS --epochs $EPOCHS --latent-dim $LATENT_DIM --dataset $DATASET --seed $SEED

# Fixed Beta=0.1
python main.py --beta_start 0.1 --beta_end 0.1 --epochs $EPOCHS --latent-dim $LATENT_DIM --dataset $DATASET --seed $SEED

# Annealed Beta=0.1->1
python main.py --beta_start 0.1 --beta_end 1 --beta_anneal_epochs $EPOCHS --epochs $EPOCHS --latent-dim $LATENT_DIM --dataset $DATASET --seed $SEED

# Plotting the results
python plot_experiment_results.py