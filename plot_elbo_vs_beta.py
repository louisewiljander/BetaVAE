"""
plot_elbo_vs_beta.py
-------------------
Simulate and plot ELBO for Beta-VAE with fixed beta, annealed beta, and standard VAE (beta=1).
"""
import numpy as np
import matplotlib.pyplot as plt

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

# Annealed beta: linearly decrease from 10 to 1
beta_annealed = np.linspace(10, 1, epochs)
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
