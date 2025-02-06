import os
import datetime as dt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    LinearLR,
    ConstantLR,
    ExponentialLR,
    SequentialLR,
)

from tqdm import tqdm
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np

from utils.fashion_mnist_dataloader import get_dataloader
from utils.logger import get_logger
from utils_torch.classes import SimpleUnet, Unet
from utils_torch.diffusion import sample, linear_beta_schedule
from utils_torch.training import get_loss

DEFAULT_IMG_SIZE = 28

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
    logger_name = "run_torch_unet"

    logger = get_logger(logger_name, log_format, date_format, log_file)

    parser = ArgumentParser(description="Run Attention Unet")
    parser.add_argument("--down_channels", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--time_emb_dim", type=int, default=4)
    parser.add_argument(
        "--zero_pad",
        action="store_true",
        help="Extend the iamge size to 32x32 to allow deeper network",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--model_tag", type=str, default="")
    parser.add_argument("--model_name", type=str, default="Unet")
    parser.add_argument("--load_checkpoint", type=str, default="")

    args = parser.parse_args()
    logger.info(f"{args}")

    down_channels = args.down_channels
    time_emb_dim = args.time_emb_dim
    zero_pad_images = args.zero_pad
    device = args.device
    BATCH_SIZE = args.batch_size
    nepochs = args.nepochs
    lr = args.lr
    T = args.timesteps
    load_checkpoint = args.load_checkpoint
    model_name = args.model_name
    model_tag = args.model_tag

    torch.set_default_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    betas = linear_beta_schedule(timesteps=T, device=device)
    sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(1.0 - betas, -1))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - torch.cumprod(1.0 - betas, -1))
    posterior_variance = betas
    sqrt_recip_alphas = 1.0 / torch.sqrt(1 - betas)

    dataloader = get_dataloader(
        BATCH_SIZE, device, channels_last=False, zero_pad_images=zero_pad_images
    )

    logger.info(f"Checkpoint directory : {models_dir}")

    match model_name:
        case "Unet":
            unet = Unet(down_channels=down_channels, time_emb_dim=time_emb_dim).to(
                device
            )
        case "SimpleUnet":
            unet = SimpleUnet(
                down_channels=down_channels, time_emb_dim=time_emb_dim
            ).to(device)
        case _:
            raise ValueError(f"{model_name} is not implemented")

    if load_checkpoint:
        unet.load_state_dict(torch.load(os.path.join(models_dir, load_checkpoint)))

    print(unet)

    unet.train()

    logger.info(
        f"Number of parameters : {np.sum([np.prod(t.shape) for t in list(unet.parameters())])}"
    )

    loss_history = []
    lr_history = []

    optimiser = Adam(unet.parameters(), lr=lr)
    scheduler = SequentialLR(
        optimiser,
        schedulers=[
            LinearLR(optimiser, 0.1, 1.0, 5),
            ConstantLR(optimiser, 1.0, 10),
            ExponentialLR(optimiser, 0.98),
        ],
        milestones=[5, 15],
    )

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
                unet,
                x,
                timestep,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                device,
            )
            loss_history.append(float(loss.detach().cpu().numpy()))
            loss.backward()

            optimiser.step()
            if k % n_steps_refresh_progress_bar == 0:
                description = (
                    f'Epoch {epoch} | Step {k} | '
                    f'Loss {(sum(loss_history[-n_steps_refresh_progress_bar:]) / n_steps_refresh_progress_bar):.4f} | '
                    f'grad norm {np.array(torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.).cpu()).mean():.2f} | '
                    f'learning rate {optimiser.param_groups[0]["lr"]:.9f}'
                )
                pbar_batch.set_description(description)
        logger.debug(description)
        scheduler.step()

    datetime_str = dt.datetime.today().strftime("%Y%m%d-%H%M")
    img_base_name = f"{script_name}_{datetime_str}"
    _, ax = plt.subplots()
    ax.plot(range(len(loss_history)), loss_history)
    ax.grid()
    ax.set_title("Batch loss evolution")
    plt.savefig(
        os.path.join(out_dir, "loss_" + img_base_name + ".png"), bbox_inches="tight"
    )
    _, ax = plt.subplots()
    ax.plot(range(len(lr_history)), lr_history)
    ax.grid()
    ax.set_title("Learning Rate evolution")
    plt.savefig(
        os.path.join(out_dir, "lr_" + img_base_name + ".png"), bbox_inches="tight"
    )

    name = (
        f"{unet._get_name()}{model_tag}_{dt.datetime.today().strftime("%Y%m%d-%H")}.pt"
    )
    location = os.path.join(models_dir, name)
    torch.save(unet.state_dict(), location)

    logger.info("Generate sample")
    unet.eval()
    sample_base_name = f"sample_{script_name}_{datetime_str}_"
    n_samp = 9

    _, axs = plt.subplots(nrows=n_samp // 3 + ((n_samp % 3) > 0), ncols=3)

    IMG_SHAPE = (1, 1, 32, 32) if zero_pad_images else (1, 1, 28, 28)
    for i in range(n_samp):
        samp = sample(
            unet,
            IMG_SHAPE,
            posterior_variance,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            T,
        )
        r, c = i // 3, i % 3
        axs[r, c].imshow(samp[-1][0, 0, :, :], cmap="gray")
        axs[r, c].axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, sample_base_name + ".png"),
    )
