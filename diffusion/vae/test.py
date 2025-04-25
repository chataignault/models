# %%
import numpy as np
import os
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms

torch.set_default_device("cuda")

from vae_utils import VAE

latent_dim = 2
depth = 4
device = "cuda"
SAVE_DIR = "models"

vae = VAE(latent_dim=latent_dim, depth=depth).to(device)

checkpoint = "vae_20250425-2036_2_4.pt"
vae.load_state_dict(torch.load(os.path.join(SAVE_DIR, checkpoint)))

vae.eval()


# %%

n_cols = 10
n_rows = 10
latent_dim = latent_dim

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)

for row, x in enumerate(np.linspace(-3.0, 3.0, 10)):
    for col, y in enumerate(np.linspace(-3.0, 3.0, 10)):
        print(row, col, x, y)
        z = torch.tensor([[[x, y]]]).to(device).to(torch.float32)
        print(z)
        # z = vae.reparametrise(mu_batch, logvar_batch)

        # pass latent variables through the decoder
        output = vae.decode(z)

        # reshape output of model into 28x28
        print(output.shape)
        images = output.reshape((28, 28))

        axs[row, col].imshow(images.detach().cpu().numpy(), cmap="gray")
        axs[row, col].axis("off")

plt.tight_layout()
img_name = f"{checkpoint}_generation_map.png"
img_path = img_name
fig.savefig(img_path)
plt.show()
