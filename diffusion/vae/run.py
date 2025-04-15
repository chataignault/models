import os
import datetime as dt
import torch
import torchvision
from torchvision import transforms
from argparse import ArgumentParser
import numpy as np
from vae_utils import VAE, ELBO_loss, train, sample_images
from functools import partial
import mlflow
from torchinfo import summary

N_SAMPLES = 15
SAVE_DIR = "models"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--nepoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument(
        "--cb",
        action="store_true",
        help="Whether to add the continuous Bernoulli distribution normalizing term",
    )

    args = parser.parse_args()

    batch_size = args.batch_size
    latent_dim = args.latent_dim
    nepoch = args.nepoch
    lr = args.lr
    device = args.device
    cb = args.cb
    dataset = args.dataset
    save = args.save
    load_checkpoint = args.load_checkpoint

    mlflow.set_experiment("VAE")

    torch.set_default_device(device)

    if dataset == "mnist":
        trainset = torchvision.datasets.MNIST(
            root="./",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    elif dataset == "fashion_mnist":
        trainset = torchvision.datasets.FashionMNIST(
            root="./",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.RandomHorizontalFlip()]
            ),
        )
    else:
        NotImplemented

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        generator=torch.Generator(device=device),
    )

    vae = VAE(latent_dim=latent_dim).to(device)
    print(
        "Number of parameters :", np.sum([np.prod(p.size()) for p in vae.parameters()])
    )
    if len(load_checkpoint) > 0:
        vae.load_state_dict(torch.load(os.path.join(SAVE_DIR, load_checkpoint)))

    with mlflow.start_run():
        mlflow.log_params(args.__dict__)

        train(
            model=vae,
            nr_epochs=nepoch,
            optimizer=torch.optim.Adam(vae.parameters(), lr=lr),
            criterion=partial(ELBO_loss, continuous_bernoulli=cb),
            dataloader=trainloader,
            device=device,
        )
        mlflow.pytorch.log_model(vae, "vae")

    vae.eval()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H")
    sample_images(vae, N_SAMPLES, "_".join([dataset, stamp]))

    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(vae)))
    mlflow.log_artifact("model_summary.txt")

    if save:
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        name = f"vae_{stamp}_{latent_dim}.pt"
        torch.save(vae.state_dict(), os.path.join(SAVE_DIR, name))
