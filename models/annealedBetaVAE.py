"""
annealedBetaVAE.py
------------------
Implements a Beta-VAE with linearly annealed beta (β) as in Higgins et al.,
where beta decreases from a specified start value to 1 over a set number of epochs.
"""
from .base import BaseVAE
from torch import nn
import torch
import numpy as np

class AnnealedBetaVAE(BaseVAE):
    """
    Beta-VAE with linearly annealed beta (from beta_start to 1), compatible with dsprites and similar datasets.
    Architecture and interface matches BetaVAEHiggins for drop-in compatibility.
    """
    model_type = "AnnealedBetaVAE"
    num_iter = 0
    def __init__(self,
                latent_dim,
                beta=1,
                img_size=(1, 64, 64),
                latent_dist='bernoulli'):
        super(AnnealedBetaVAE, self).__init__()
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
        self.lind3 = nn.Linear(1200, 1200)
        self.lind4 = nn.Linear(1200, 4096)

    def encode(self, input):
        batch_size = input.size(0)
        input = input.view((batch_size, -1))
        result = torch.relu(self.line1(input))
        result = torch.relu(self.line2(result))
        mu_logvar = self.mu_logvar_gen(result)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)
        return mu, logvar

    def decode(self, input):
        batch_size = input.size(0)
        x = torch.tanh(self.lind1(input))
        x = torch.tanh(self.lind2(x))
        x = torch.tanh(self.lind3(x))
        x = torch.sigmoid(self.lind4(x)).reshape((batch_size, 1, 64, 64))
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon, x, mu, log_var, storer=None):
        self.num_iter += 1
        batch_size = x.size(0)
        if self.latent_dist == 'bernoulli':
            recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction='sum')
        elif self.latent_dist == "gaussian":
            recon_loss = nn.functional.mse_loss(recon * 255, x * 255, reduction="sum") / 255
        latent_kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=0)
        kld_loss = torch.sum(latent_kl)
        loss = recon_loss + self.beta * kld_loss
        if storer is not None:
            storer['recon_loss'].append(recon_loss.item())
            storer['kl_loss'].append(kld_loss.item())
            for i in range(self.latent_dim):
                storer['kl_loss_' + str(i)].append(latent_kl[i].item())
            storer['loss'].append(loss.item())
        return loss

    def sample(self, num_samples: int, current_device: int, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        return self.forward(x)[0]
