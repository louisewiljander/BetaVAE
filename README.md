# Beta-VAE Experiments

This repository provides a framework for training, evaluating, and comparing standard VAE, fixed Beta-VAE, and annealed Beta-VAE models on datasets such as MNIST and dSprites. The codebase supports flexible beta scheduling, robust logging, and easy experiment management.

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

### 4. Skipping FID/InceptionV3 Evaluation

By default, FID/InceptionV3 evaluation is skipped for faster runs. If you want to enable it, check the relevant flags in `main.py`.

## Logging and Results

- Training losses (ELBO, recon loss, KL) are logged to `results/train_losses.log` in CSV format for easy plotting.
- If using wandb, all metrics and beta schedules are logged for experiment tracking.
- Model checkpoints are saved in the `results/` directory.

## Plotting

You can use the provided plotting scripts (e.g., `plot_elbo_wandb.py`) to visualize ELBO and other metrics. The CSV log format is compatible with most plotting tools.

## Customization

- Beta scheduling is handled via the `--beta_start`, `--beta_end`, and `--beta_anneal_epochs` arguments.
- All loss components are available for analysis and plotting.
- The codebase is modular and easy to extend for new datasets or VAE variants.

## Troubleshooting

- Ensure your dataset is available and preprocessed as expected (images resized to 64x64).
- For quick tests, reduce the number of epochs or use a smaller dataset.
- If you encounter errors, check the logs and ensure all dependencies are installed.
