# example run
# * python run_torch_unet.py --nepochs=2

import os
import datetime as dt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tqdm import tqdm
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np

from utils.fashion_mnist_dataloader import get_dataloader
from utils.logger import get_logger
from utils_torch.classes import SimpleUnet
from utils_torch.diffusion import sample, linear_beta_schedule
from utils_torch.training import get_loss

if __name__ == "__main__":
    script_name = "torch_unet"

    log_dir = os.path.join(os.getcwd(), "log")
    out_dir = os.path.join(os.getcwd(), "out")
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(log_dir, script_name + ".log")
    date_format = "%Y-%m-%d %H:%M:%S"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger_name = "run_jax_unet"

    logger = get_logger(logger_name, log_format, date_format, log_file)

    parser = ArgumentParser(description="Run Attention Unet")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--img_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--load_checkpoint", type=str, default="")

    args = parser.parse_args()
    logger.info(f"{args}")

    device = args.device
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    nepochs = args.nepochs
    lr = args.lr
    T = args.timesteps
    load_checkpoint = args.load_checkpoint

    torch.set_default_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    betas = linear_beta_schedule(timesteps=T, device=device)
    sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(1.0 - betas, -1))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - torch.cumprod(1.0 - betas, -1))
    posterior_variance = betas
    sqrt_recip_alphas = 1.0 / torch.sqrt(1 - betas)

    dataloader = get_dataloader(BATCH_SIZE, device, channels_last=False)

    logger.info(f"Checkpoint directory : {models_dir}")

    unet = SimpleUnet().to(device)

    if load_checkpoint:
        unet.load_state_dict(torch.load(os.path.join(models_dir, load_checkpoint)))

    unet.train()

    logger.info(
        f"Number of parameters : {np.sum([np.prod(t.shape) for t in list(unet.parameters())])}"
    )

    loss_history = []
    lr_history = []

    optimiser = Adam(unet.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=10, eta_min=5e-6)

    n_steps_refresh_progress_bar = 5

    loss_history = []
    lr_history = []

    for epoch in range(nepochs):
        pbar_batch = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
        for k, x in pbar_batch:
            x = x["pixel_values"]
            timestep = torch.randint(1, T, (x.shape[0],))
            optimiser.zero_grad()
            lr_history.append(optimiser.param_groups[0]["lr"])
            loss = get_loss(
                unet, x, timestep, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device
            )
            loss_history.append(float(loss.detach().cpu().numpy()))
            loss.backward()

            optimiser.step()
            if k % n_steps_refresh_progress_bar == 0:
                new_loss = float(loss.detach().cpu().mean().numpy())
                description = (
                    f'Epoch {epoch} | Step {k} | '
                    f'Loss {new_loss / n_steps_refresh_progress_bar:.4f} | '
                    f'grad norm {np.array(torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.).cpu()).mean():.2f} | '
                    f'learning rate {optimiser.param_groups[0]["lr"]:.9f}'
                )
                pbar_batch.set_description(description)
        logger.info(description)
        scheduler.step()

    datetime_str = dt.date.today().strftime("%Y%m%d-%H%M")
    img_base_name = f"{script_name}_{datetime_str}"
    _, ax = plt.subplots()
    ax.plot(range(len(loss_history)), loss_history)
    ax.grid()
    ax.set_title("Batch loss evolution")
    plt.savefig(os.path.join(out_dir, img_base_name + "_loss.png"), bbox_inches="tight")
    _, ax = plt.subplots()
    ax.plot(range(len(lr_history)), lr_history)
    ax.grid()
    ax.set_title("Learning Rate evolution")
    plt.savefig(os.path.join(out_dir, img_base_name + "_lr.png"), bbox_inches="tight")

    logger.info("Generate sample")
    unet.eval()
    sample_base_name = f"sample_{script_name}_{datetime_str}_"
    samp = sample(
        unet,
        (1, 1, 28, 28),
        posterior_variance,
        sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas,
        T,
    )

    i = 0
    for im in samp[:: (T // 4)]:
        plt.imsave(
            os.path.join(out_dir, sample_base_name + str(i) + ".png"),
            im.reshape(28, 28),
            cmap="gray",
        )
        i += 1
    plt.imsave(
        os.path.join(out_dir, sample_base_name + "final.png"),
        samp[-1].reshape(28, 28),
        cmap="gray",
    )

    name = f"unet_{dt.date.today().strftime("%Y%m%d")}.pt"
    location = os.path.join(models_dir, name)
    torch.save(unet.state_dict(), location)
