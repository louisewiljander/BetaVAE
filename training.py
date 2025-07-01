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
                dataset_size = 1000, all_latents = True, gif_visualizer = None, seed = None, dataset_name = None):
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



    def __call__(self, data_loader, epochs=10, checkpoint_every = 10, wandb_log = False):
        """
        Run the training loop.

        Args:
            data_loader: DataLoader for training data.
            epochs: Number of epochs to train.
            checkpoint_every: Save model every N epochs.
            wandb_log: Whether to log metrics to Weights & Biases.
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

            self.logger.info('Epoch: {} Average loss per image: {:.2f}'.format(epoch + 1,
                                                                               mean_epoch_loss))

            self.losses_logger.log(epoch, storer)
  
            if self.gif_visualizer is not None:
                self.gif_visualizer()

            if epoch % checkpoint_every == 0:
                save_model(self.model,self.save_dir,
                           filename="model-{}.pt".format(epoch))

            if self.scheduler is not None:
                self.scheduler.step()

            self.model.eval()

            if wandb_log:
                metrics, losses = {}, {}
                # Log ELBO (total loss) to wandb
                elbo = None
                if "loss" in storer:
                    elbo = mean(storer["loss"])
                    wandb.log({"epoch": epoch, "elbo": elbo, "beta": getattr(self.model, "beta", None)})
                
                if epoch % max(round(epochs/abs(self.metrics_freq)), 10) == 0 and abs(epoch-epochs) >= 5 and (epoch != 0 if self.metrics_freq < 0 else True):
                    metrics = train_evaluator.compute_metrics(data_loader, self.dataset_name)
                losses = train_evaluator.compute_losses(data_loader, batch_size=batch_size)
                wandb.log({"epoch":epoch,"metric":metrics, "loss":losses})

                ### Log standard ELBO (beta=1) on training data, no gradients ###
                original_beta = self.model.beta
                self.model.beta = 1.0
                standard_elbo_results = self.evaluate_no_grad(data_loader)
                self.model.beta = original_beta

                import numpy as np
                mean_standard_elbo = np.mean([r['loss'] for r in standard_elbo_results])
                mean_recon = np.mean([r['recon_loss'] for r in standard_elbo_results])
                mean_kl = np.mean([r['kl_loss'] for r in standard_elbo_results])
                wandb.log({
                    "epoch": epoch,
                    "standard_elbo": mean_standard_elbo,
                    "standard_recon_loss": mean_recon,
                    "standard_kl_loss": mean_kl
                })

            self.model.train()           

        if self.gif_visualizer is not None:
            self.gif_visualizer.save_reset()

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        self.logger.info('Finished training after {:.1f} min.'.format(delta_time))


    ### Added method for evaluation/testing without gradients: ###
    def evaluate_no_grad(self, data_loader):
        """
        Evaluate model on data_loader without gradients (for ELBO, recon, KL, etc).
        """
        self.model.eval()
        results = []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                recon, mu, logvar = self.model(data)
                result = self.model.loss_function(recon, data, mu, logvar)
                if isinstance(result, tuple) and len(result) == 3:
                    loss, recon_loss, kl_loss = result
                else:
                    loss = result
                    recon_loss = float('nan')
                    kl_loss = float('nan')
                results.append({
                    'loss': loss.item(),
                    'recon_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item()
                })
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
            # Always log the ELBO (total loss) as 'elbo'
            if "recon_loss" in losses_storer and "kl_loss" in losses_storer:
                # Try to get beta from storer if present, else assume 1
                beta = losses_storer.get("beta", [1])[0]
                elbo = mean(losses_storer["recon_loss"]) + beta * mean(losses_storer["kl_loss"])
                f.write("{},{},{}\n".format(epoch, "elbo", elbo))


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

def get_beta(epoch, beta_start, beta_end, anneal_epochs):
    """
    Compute the annealed beta value for the current epoch.

    Args:
        epoch: Current epoch number.
        beta_start: Initial beta value.
        beta_end: Final beta value.
        anneal_epochs: Number of epochs over which to anneal beta.
    Returns:
        The annealed beta value.
    """
    if anneal_epochs == 0:
        return beta_end
    if epoch >= anneal_epochs:
        return beta_end
    # Linear schedule
    return beta_start + (beta_end - beta_start) * (epoch / anneal_epochs)