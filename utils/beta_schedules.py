"""
beta_schedules.py
----------------
Utility functions for beta scheduling in Beta-VAE training.
"""

def linear_beta_schedule(epoch, total_epochs, beta_start=5.0, beta_end=1.0):
    """Linearly anneal beta from beta_start to beta_end over total_epochs."""
    if total_epochs == 0:
        return beta_end
    if epoch >= total_epochs:
        return beta_end
    return beta_start + (beta_end - beta_start) * (epoch / total_epochs)
