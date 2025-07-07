"""
beta_schedules.py
----------------
Utility functions for beta scheduling in Beta-VAE training.
"""

def linear_beta_schedule(epoch, total_epochs, beta_start=5.0, beta_end=1.0):
    """
    Linearly anneal beta from beta_start to beta_end over total_epochs,
    but keep beta fixed at beta_start for the first epoch (epoch 0).
    The last epoch (epoch=total_epochs-1) will use beta_end.
    """
    if total_epochs <= 1:
        return beta_end
    if epoch == 0:
        return beta_start
    if epoch >= total_epochs - 1:
        return beta_end
    # Anneal from epoch 1 to total_epochs-2
    return beta_start + (beta_end - beta_start) * ((epoch - 1) / (total_epochs - 2))
