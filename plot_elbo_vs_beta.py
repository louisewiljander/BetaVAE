import pandas as pd
import matplotlib.pyplot as plt

# Adjust the path to your experiment directory if needed
logfile = 'results/your_experiment/train_losses.log'
df = pd.read_csv(logfile)

plt.figure(figsize=(8,5))
plt.plot(df[df['Loss'] == 'val_loss']['Epoch'], df[df['Loss'] == 'val_loss']['Value'], label='Beta-VAE Loss (val)')
plt.plot(df[df['Loss'] == 'val_standard_elbo']['Epoch'], df[df['Loss'] == 'val_standard_elbo']['Value'], 'o-', label='Standard ELBO (val, beta=1)')
plt.xlabel('Epoch')
plt.ylabel('Loss per image')
plt.legend()
plt.title('Validation Loss: Beta-VAE vs Standard ELBO')
plt.tight_layout()
plt.show()