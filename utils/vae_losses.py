import torch
import torch.nn.functional as F

def beta_vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Compute the Beta-VAE loss (reconstruction + beta * KL divergence).
    Args:
        recon_x: Reconstructed input.
        x: Original input.
        mu: Latent mean.
        logvar: Latent log-variance.
        beta: Weight for KL term.
    Returns:
        total_loss: Scalar loss.
        recon_loss: Reconstruction loss.
        kl_loss: KL divergence loss.
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

# Beta scheduling utilities have moved to utils/beta_schedules.py
