"""
betaVAEHiggins.py
-----------------
Defines the BetaVAEHiggins class, an implementation of the Beta-VAE model as proposed by Higgins et al.
This class provides methods for encoding, decoding, sampling, and computing the loss for training a Beta-VAE.
"""

import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class BetaVAEHiggins(BaseVAE):
    """
    BetaVAEHiggins implements the Beta-VAE model with configurable latent dimension and beta parameter.
    Inherits from BaseVAE and provides methods for encoding, decoding, sampling, and loss computation.
    """
    num_iter = 0
    model_type = "BetaVAEHiggins"

    def __init__(self,
                latent_dim = 10,
                beta = 1,
                img_size = (1, 64, 64),
                latent_dist = 'bernoulli'):
        """
        Initialize the BetaVAEHiggins model.

        Args:
            latent_dim: Number of latent dimensions.
            beta: Weight for the KL divergence term.
            img_size: Size of the input images.
            latent_dist: Distribution type for the latent space ('bernoulli' or 'gaussian').
        """
        super(BetaVAEHiggins, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.latent_dist = latent_dist
        self.img_size = img_size

        input_dim = 4096

        self.line1 = nn.Linear(input_dim, 1200)
        self.line2 = nn.Linear(1200, 1200)
        self.mu_logvar_gen = nn.Linear(1200, self.latent_dim*2)

        self.lind1 = nn.Linear(latent_dim, 1200)
        self.lind2 = nn.Linear(1200, 1200)
        self.lind3 =nn.Linear(1200, 1200)
        self.lind4 = nn.Linear(1200, 4096)


    def encode(self, input):
        """
        Encode input images into latent mean and log-variance.

        Args:
            input: Input tensor of images.
        Returns:
            mu: Latent mean.
            logvar: Latent log-variance.
        """
        batch_size = input.size(0)
        input = input.view((batch_size, -1))
        result = torch.relu(self.line1(input))
        result = torch.relu(self.line2(result))
        mu_logvar = self.mu_logvar_gen(result)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar

    def decode(self, input):
        """
        Decode latent variables back to image space.

        Args:
            input: Latent tensor.
        Returns:
            Reconstructed images.
        """
        batch_size = input.size(0)
        x = torch.tanh(self.lind1(input))
        x = torch.tanh(self.lind2(x))
        x = torch.tanh(self.lind3(x))
        x = torch.sigmoid(self.lind4(x)).reshape((batch_size, 1, 64, 64))# Sigmoid because the distribution over pixels is supposed to be Bernoulli
        return x

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).

        Args:
            mu: Latent mean.
            logvar: Latent log-variance.
        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        """
        Forward pass through the VAE: encode, reparameterize, and decode.

        Args:
            input: Input tensor.
        Returns:
            Reconstructed images, latent mean, and log-variance.
        """
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon, x, mu, log_var, storer = None):
        """
        Compute the Beta-VAE loss (reconstruction + beta * KL divergence).

        Args:
            recon: Reconstructed images.
            x: Original images.
            mu: Latent mean.
            log_var: Latent log-variance.
            storer: Optional dictionary to store loss components for logging.
        Returns:
            Total loss for the batch.
        """
        self.num_iter += 1
        batch_size = x.size(0)
        if self.latent_dist == 'bernoulli':
            recon_loss =F.binary_cross_entropy(recon, x, reduction='sum')
        elif self.latent_dist  == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
            recon_loss = F.mse_loss(recon * 255, x * 255, reduction="sum") / 255
        #recon_loss/= batch_size
        latent_kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=0)
        kld_loss = torch.sum(latent_kl)

      
        loss = recon_loss + self.beta  * kld_loss

        if storer is not None:
            storer['recon_loss'].append(recon_loss.item())
            storer['kl_loss'].append(kld_loss.item())
            for i in range(self.latent_dim):
                storer['kl_loss_' + str(i)].append(latent_kl[i].item())
            storer['loss'].append(loss.item())

        return loss, recon_loss, kld_loss


    #smaple form latent sapce
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Sample images from the latent space.

        Args:
            num_samples: Number of samples to generate.
            current_device: Device to place the samples on.
        Returns:
            Generated samples.
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    #generate reconstructed image
    def generate(self, x):
        """
        Generate reconstructed images from input x.

        Args:
            x: Input tensor.
        Returns:
            Reconstructed images.
        """
        return self.forward(x)[0]