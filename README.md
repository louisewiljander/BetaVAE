# Beta-VAE Experiments

This codebase is an adaptation of the code from Fil et al. (2021) [https://github.com/jonaswildberger/BetaVAE/tree/main](https://github.com/jonaswildberger/BetaVAE/tree/main).

This repository provides a framework for training, evaluating, and comparing standard VAE, fixed Beta-VAE, and annealed Beta-VAE models on datasets such as MNIST and dSprites.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- tqdm
- wandb (optional, for experiment tracking)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

### 1. Standard VAE (beta=1)

To train a standard VAE (beta=1) for 5 epochs on MNIST:

```bash
python main.py --dataset mnist --epochs 5 --beta_start 1.0 --beta_end 1.0 --beta_anneal_epochs 0 --log_wandb True
```

### 2. Annealed Beta-VAE (beta>1 → 1)

To train an annealed Beta-VAE (beta=4.0 → 1.0) for 5 epochs on MNIST:

```bash
python main.py --dataset mnist --epochs 5 --beta_start 4.0 --beta_end 1.0 --beta_anneal_epochs 5 --log_wandb True
```

- `--beta_start` and `--beta_end` control the beta schedule.
- `--beta_anneal_epochs` sets the number of epochs over which to anneal beta (set to 0 for fixed beta).
- `--log_wandb` enables logging to Weights & Biases (optional).

### 3. Other Datasets

To run on dSprites, change `--dataset mnist` to `--dataset dsprites`.

## Logging and Results
- If using wandb, all metrics and beta schedules are logged for experiment tracking.

## Plotting

You can use the provided plotting scripts (e.g., `plot_experiment_results.py` to plot training runs. The plotting scripts support both individual run visualization and aggregated (mean ± std) plots across multiple runs.

## Customization

- Beta scheduling is handled via the `--beta_start`, `--beta_end`, and `--beta_anneal_epochs` arguments.
- All loss components are available for analysis and plotting.
