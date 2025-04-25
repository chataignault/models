import os
import datetime as dt
import torch
import torchvision
from torchvision import transforms
from argparse import ArgumentParser
import numpy as np
from enum import Enum
from vae_utils import VAE, CVAE, ELBO_loss, train
from functools import partial
import mlflow
from torchinfo import summary
from lion_pytorch import Lion

N_SAMPLES = 15
SAVE_DIR = "models"
OUT_DIR = "out"


class AvailableVAE(Enum):
    VAE = "VAE"
    CVAE = "CVAE"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--model", type=AvailableVAE, default=AvailableVAE.VAE)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--nepoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="parameter referring to beta-VAE for disentangling the latent space",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--lion_optimizer", action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument(
        "--cb",
        action="store_true",
        help="Whether to add the continuous Bernoulli distribution normalizing term",
    )

    args = parser.parse_args()

    batch_size = args.batch_size
    model = args.model
    latent_dim = args.latent_dim
    depth = args.depth
    nepoch = args.nepoch
    lr = args.lr
    beta = args.beta
    device = args.device
    cb = args.cb
    dataset = args.dataset
    save = args.save
    load_checkpoint = args.load_checkpoint
    lion_optimizer = args.lion_optimizer

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
    match model:
        case AvailableVAE.VAE:
            vae = VAE(latent_dim=latent_dim, depth=depth).to(device)
        case AvailableVAE.CVAE:
            vae = CVAE(latent_dim=latent_dim)
        case _:
            NotImplementedError
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
            optimizer=torch.optim.Adam(vae.parameters(), lr=lr)
            if not lion_optimizer
            else Lion(vae.parameters(), lr=lr, weight_decay=1e-2),
            criterion=partial(ELBO_loss, continuous_bernoulli=cb, beta=beta),
            dataloader=trainloader,
            device=device,
        )
        mlflow.pytorch.log_model(vae, "vae")

    vae.eval()

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    vae.sample_images(N_SAMPLES, OUT_DIR, "_".join([dataset, stamp, model.value]))

    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(vae)))
    mlflow.log_artifact("model_summary.txt")
    if save:
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        name = f"vae_{stamp}_{latent_dim}_{depth}.pt"
        torch.save(vae.state_dict(), os.path.join(SAVE_DIR, name))
