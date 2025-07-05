"""
training.py
-----------
This module provides the main training loop and utilities for training Beta-VAE models.
It includes the Trainer class for orchestrating training, logging, and checkpointing,
as well as helper functions for saving models and logging losses.
"""

import imageio
import logging
import os
import json
from timeit import default_timer
from collections import defaultdict

from tqdm import trange
import torch
from torch.nn import functional as F
from evaluate import Evaluator

import wandb
from utils.beta_schedules import linear_beta_schedule

TRAIN_LOSSES_LOGFILE = "train_losses.log"
MODEL_FILENAME = "model.pt"
META_FILENAME = "specs.json"

class Trainer():
    """
    Trainer class to handle the training loop for Beta-VAE models.
    Manages model training, logging, checkpointing, and optional wandb integration.
    """
    def __init__(self, model, optimizer, scheduler = None, device=torch.device("cpu"),
                logger=logging.getLogger(__name__), metrics_freq =-2, sample_size = 64,
                save_dir ="results",
                dataset_size = 1000, all_latents = True, gif_visualizer = None, seed = None, dataset_name = None,
                beta_start=1.0, beta_end=1.0, beta_anneal_epochs=None):
        """
        Initialize the Trainer.

        Args:
            model: The model to train.
            optimizer: Optimizer for training.
            scheduler: Optional learning rate scheduler.
            device: Device to use for training.
            logger: Logger instance.
            metrics_freq: Frequency for metrics logging.
            sample_size: Number of samples for evaluation.
            save_dir: Directory to save results.
            dataset_size: Size of the dataset.
            all_latents: Whether to use all latents for evaluation.
            gif_visualizer: Optional visualizer for GIFs.
            seed: Random seed.
            dataset_name: Name of the dataset.
            beta_start: Initial beta value for annealing schedule.
            beta_end: Final beta value for annealing schedule.
            beta_anneal_epochs: Number of epochs over which to anneal beta (defaults to total epochs if None).
        """
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.logger = logger
        self.losses_logger = LossesLogger(os.path.join(save_dir, TRAIN_LOSSES_LOGFILE))
        self.metrics_freq =metrics_freq
        self.sample_size = sample_size
        self.dataset_size = dataset_size
        self.all_latents = all_latents
        self.gif_visualizer = gif_visualizer
        self.seed = seed
        self.dataset_name = dataset_name
        ### Beta annealing attributes ###
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_epochs = beta_anneal_epochs



    def __call__(self, data_loader, epochs=10, checkpoint_every=10, wandb_log=False, val_loader=None):
        """
        Run the training loop.

        Args:
            data_loader: DataLoader for training data.
            epochs: Number of epochs to train.
            checkpoint_every: Save model every N epochs.
            wandb_log: Whether to log metrics to Weights & Biases.
            val_loader: DataLoader for validation data.
        """
        start = default_timer()
        storers = []
        self.model.train()

        if wandb_log:
            wandb.config.update({"beta": getattr(self.model, "beta", 1)}, allow_val_change=True)
            train_evaluator = Evaluator(model=self.model, device=self.device, seed=self.seed,
                                        sample_size=self.sample_size, dataset_size=self.dataset_size, all_latents=self.all_latents)

        for epoch in range(epochs):
            # Anneal beta
            if hasattr(self.model, "beta") and hasattr(self, "beta_start") and hasattr(self, "beta_end") and hasattr(self, "beta_anneal_epochs"):
                total_anneal_epochs = self.beta_anneal_epochs or epochs
                self.model.beta = linear_beta_schedule(epoch, total_anneal_epochs, self.beta_start, self.beta_end)
                print(f"Epoch {epoch+1}: beta = {self.model.beta}")  # Log to terminal
                if wandb_log:
                    wandb.log({"epoch": epoch, "beta": self.model.beta})

            storer = defaultdict(list)
            epoch_loss = 0

            kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=False)
            with trange(len(data_loader), **kwargs) as t:

                for _, (data, _) in enumerate(data_loader):
                    batch_size, _, _, _ = data.size()
                    data = data.to(self.device)
                    self.optimizer.zero_grad()
                    recon_batch, mu, logvar = self.model(data)

                    # Unpack loss tuple and use only the total loss for backward
                    loss, recon_loss, kl_loss = self.model.loss_function(recon_batch, data, mu,logvar, storer=storer)
                    loss = loss / len(data)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(loss=loss)
                    t.update()

            mean_epoch_loss = epoch_loss / len(data_loader)

            # --- LOGGING: Printing outputs
            mean_train_recon = sum(storer['recon_loss']) / (len(storer['recon_loss']) * data_loader.batch_size) if 'recon_loss' in storer else float('nan')
            mean_train_kl = sum(storer['kl_loss']) / (len(storer['kl_loss']) * data_loader.batch_size) if 'kl_loss' in storer else float('nan')
            print(f"Epoch {epoch + 1} | Beta: {self.model.beta}")
            print(f"Train loss per image: {mean_epoch_loss:.2f} | Recon: {mean_train_recon:.2f} | KL: {mean_train_kl:.2f}")
            self.logger.info(f"Epoch: {epoch + 1} Training loss per image: {mean_epoch_loss:.2f}")
            if wandb_log:
                wandb.log({"epoch": epoch, "train_loss": mean_epoch_loss, "beta": getattr(self.model, "beta", None)})

            # --- LOGGING: Write training losses to CSV with 'train_' prefix ---
            train_losses_storer = {f"train_{k}": v for k, v in storer.items()}
            self.losses_logger.log(epoch, train_losses_storer)
            if self.gif_visualizer is not None:
                self.gif_visualizer()

            ### Validation Loss Logging ###
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                val_recon_losses = []
                val_kl_losses = []
                with torch.no_grad():
                    for data, _ in val_loader:
                        data = data.to(self.device)
                        recon, mu, logvar = self.model(data)
                        val_loss, val_recon_loss, val_kl_loss = self.model.loss_function(recon, data, mu, logvar)
                        val_losses.append(val_loss.item())
                        val_recon_losses.append(val_recon_loss.item())
                        val_kl_losses.append(val_kl_loss.item())
                # Normalize validation loss per image, matching training loss normalization
                mean_val_loss = sum(val_losses) / (len(val_losses) * val_loader.batch_size)
                mean_val_recon = sum(val_recon_losses) / (len(val_recon_losses) * val_loader.batch_size)
                mean_val_kl = sum(val_kl_losses) / (len(val_kl_losses) * val_loader.batch_size)
                # Logging validation loss
                print(f"Validation loss per image: {mean_val_loss:.2f} | Recon: {mean_val_recon:.2f} | KL: {mean_val_kl:.2f}")
                self.logger.info(f"Epoch: {epoch + 1} Validation loss per image: {mean_val_loss:.2f}")
                if wandb_log:
                    wandb.log({
                        "epoch": epoch,
                        "val_loss": mean_val_loss,
                        "val_recon_loss": mean_val_recon,
                        "val_kl_loss": mean_val_kl
                    })
                # Optionally log to CSV as well
                val_losses_storer = {
                    "val_loss": [mean_val_loss],
                    "val_recon_loss": [mean_val_recon],
                    "val_kl_loss": [mean_val_kl]
                }
                self.losses_logger.log(epoch, val_losses_storer)

            # Logging standard ELBO (beta=1) on training data, no gradients ---
            self.model.eval()
            standard_elbo_results = self.compute_standard_elbo(data_loader)
            import numpy as np
            batch_size = data_loader.batch_size
            mean_standard_elbo = np.mean([r['loss'] for r in standard_elbo_results]) / batch_size
            mean_standard_recon = np.mean([r['recon_loss'] for r in standard_elbo_results]) / batch_size
            mean_standard_kl = np.mean([r['kl_loss'] for r in standard_elbo_results]) / batch_size
            if wandb_log:
                wandb.log({
                    "epoch": epoch,
                    "standard_elbo": mean_standard_elbo,
                    "standard_recon_loss": mean_standard_recon,
                    "standard_kl_loss": mean_standard_kl
                })
            # Optionally log to CSV as well
            standard_elbo_storer = {
                "standard_elbo": [mean_standard_elbo],
                "standard_recon_loss": [mean_standard_recon],
                "standard_kl_loss": [mean_standard_kl]
            }
            self.losses_logger.log(epoch, standard_elbo_storer)
            self.model.train()

            if epoch % checkpoint_every == 0:
                save_model(self.model,self.save_dir,
                           filename="model-{}.pt".format(epoch))

            if self.scheduler is not None:
                self.scheduler.step()

            self.model.eval()
            self.model.train()           

        if self.gif_visualizer is not None:
            self.gif_visualizer.save_reset()

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        self.logger.info('Finished training after {:.1f} min.'.format(delta_time))


    def compute_standard_elbo(self, data_loader):
        """
        Evaluate the model on the given data_loader with beta=1 (standard ELBO), no gradients.
        Returns a list of dicts with 'loss', 'recon_loss', 'kl_loss' for each batch.
        """
        results = []
        was_training = self.model.training
        self.model.eval()
        current_beta = getattr(self.model, 'beta', 1.0)
        self.model.beta = 1.0
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                recon, mu, logvar = self.model(data)
                loss, recon_loss, kl_loss = self.model.loss_function(recon, data, mu, logvar)
                results.append({
                    'loss': loss.item(),
                    'recon_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item()
                })
        self.model.beta = current_beta
        if was_training:
            self.model.train()
        return results


class LossesLogger(object):
    """
    Logger for writing training losses to a CSV file for easy plotting and analysis.
    """
    def __init__(self, file_path_name):
        """
        Initialize the LossesLogger.

        Args:
            file_path_name: Path to the log file.
        """
        print(file_path_name)
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)
        # Write CSV header
        with open(file_path_name, "w") as f:
            f.write("Epoch,Loss,Value\n")

        self.file_path_name = file_path_name

    def log(self, epoch, losses_storer):
        """
        Write losses for a given epoch to the log file.

        Args:
            epoch: Current epoch number.
            losses_storer: Dictionary of loss lists.
        """
        # Log each loss type (e.g., recon_loss, kl_loss, etc.)
        with open(self.file_path_name, "a") as f:
            for k, v in losses_storer.items():
                f.write("{},{},{}\n".format(epoch, k, mean(v)))

def save_model(model, directory, metadata=None, filename=MODEL_FILENAME):
    """
    Save a model and corresponding metadata to disk.

    Args:
        model: The model to save.
        directory: Directory to save the model and metadata.
        metadata: Optional dictionary of metadata to save.
        filename: Filename for the saved model and metadata.
    """
    device = next(model.parameters()).device
    model.cpu()

    if metadata is None:
        # save the minimum required for loading
        metadata = dict(img_size=model.img_size, latent_dim=model.latent_dim,
                        model_type=model.model_type)
        # Optionally add beta if present in model
        if hasattr(model, "beta"):
            metadata["beta"] = model.beta

    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    path_to_model = os.path.join(directory, filename)
    torch.save(model.state_dict(), path_to_model)

    model.to(device)  # restore device

    
def mean(l):
    """Compute the mean of a list."""
    return sum(l) / len(l)