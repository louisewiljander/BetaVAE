import torch
import torch.nn.functional as F

def beta_vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

def linear_beta_schedule(epoch, total_epochs, beta_start=5.0, beta_end=1.0):
    if total_epochs == 0:
        return beta_end
    if epoch >= total_epochs:
        return beta_end
    return beta_start + (beta_end - beta_start) * (epoch / total_epochs)
