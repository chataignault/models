import torch
import torchvision
from torchvision import transforms
from argparse import ArgumentParser
import numpy as np
from vae_utils import VAE, ELBO_loss, train, sample_images

N_SAMPLES = 15

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--nepoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    batch_size = args.batch_size
    nepoch = args.nepoch
    lr = args.lr
    device = args.device

    torch.set_default_device(device)

    trainset = torchvision.datasets.MNIST(
        root="./",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        generator=torch.Generator(device=device),
    )

    vae = VAE().to(device)
    print(
        "Number of parameters :", np.sum([np.prod(p.size()) for p in vae.parameters()])
    )
    train(
        model=vae,
        nr_epochs=nepoch,
        optimizer=torch.optim.Adam(vae.parameters(), lr=lr),
        criterion=ELBO_loss,
        dataloader=trainloader,
        device=device,
    )

    vae.eval()

    sample_images(vae, N_SAMPLES)
