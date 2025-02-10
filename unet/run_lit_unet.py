import os
import torch
import numpy as np
import datetime as dt
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from utils.logger import get_logger
from utils_torch.classes import LitUnet
from utils_torch.classes import SimpleUnet, Unet
from utils.fashion_mnist_dataloader import get_dataloader
from utils_torch.diffusion import sample, linear_beta_schedule

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
        help="Extend the image size to 32x32 to allow deeper network",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--model_tag", type=str, default="")
    parser.add_argument("--model_name", type=str, default="Unet")
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--only_generate_sample", action="store_true")

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
    only_generate_sample = args.only_generate_sample

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
    logger.info(
        f"Number of parameters : {np.sum([np.prod(t.shape) for t in list(unet.parameters())])}"
    )

    unet = LitUnet(
        unet=unet,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        T=T,
        device=device,
        lr=lr,
    )
    trainer = L.Trainer(
        max_epochs=nepochs,
        accelerator="auto",
        callbacks=[LearningRateMonitor(), DeviceStatsMonitor()],
        logger=TensorBoardLogger("tb_logs", name=model_name, log_graph=True),
    )
    if not only_generate_sample:
        trainer.fit(model=unet, train_dataloaders=dataloader)

    datetime_str = dt.datetime.today().strftime("%Y%m%d-%H%M")
    img_base_name = f"{script_name}_{datetime_str}"

    unet = unet.unet.to(device)
    unet.eval()
    name = (
        f"{unet._get_name()}{model_tag}_{dt.datetime.today().strftime("%Y%m%d-%H")}.pt"
    )
    location = os.path.join(models_dir, name)
    torch.save(unet.state_dict(), location)

    logger.info("Generate sample")
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
