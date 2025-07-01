"""
plot_elbo_vs_beta.py
-------------------
Simulate and plot ELBO for Beta-VAE with fixed beta, annealed beta, and standard VAE (beta=1).
"""
import numpy as np
import matplotlib.pyplot as plt
from utils.beta_schedules import linear_beta_schedule

# Simulated training parameters
epochs = 50
recon_loss = np.linspace(100, 80, epochs)  # Simulate improving reconstruction loss
kl = np.linspace(20, 10, epochs)           # Simulate decreasing KL divergence

# Standard VAE (beta=1)
beta_standard = 1
elbo_standard = -recon_loss - beta_standard * kl

# Fixed Beta-VAE (beta=10)
beta_fixed = 10
elbo_fixed = -recon_loss - beta_fixed * kl

# Annealed beta: use the same schedule as in training
beta_annealed = np.array([linear_beta_schedule(e, epochs, 10, 1) for e in range(epochs)])
elbo_annealed = -recon_loss - beta_annealed * kl

plt.plot(range(epochs), elbo_standard, label='Standard VAE (Beta=1)')
plt.plot(range(epochs), elbo_fixed, label='Beta-VAE (Beta=10)')
plt.plot(range(epochs), elbo_annealed, label='Annealed Beta (10→1)')
plt.xlabel('Epoch')
plt.ylabel('ELBO')
plt.title('ELBO: Standard VAE vs Beta-VAE vs Annealed Beta-VAE (Simulation)')
plt.legend()
plt.grid(True)
plt.show()
