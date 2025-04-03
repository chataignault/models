import os
import torch
import torchvision
import numpy as np
import datetime as dt
from rich import print
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

from utils.logger import get_logger
from utils_torch.unet import LitUnet, write_sample_to_board, load_model
from utils.dataloader import get_dataloader, DataSets
from utils_torch.diffusion import (
    sample,
    linear_beta_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
    NoiseSchedule,
)

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
    parser.add_argument("--downs", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--time_emb_dim", type=int, default=16)
    parser.add_argument(
        "--zero_pad",
        action="store_true",
        help="Extend the image size to 32x32 to allow deeper network",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DataSets.fashion_mnist,
        help="dataset name from HuggingFace",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--nepochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--model_tag", type=str, default="", help="name identifier")
    parser.add_argument("--model_name", type=str, default="Unet")
    parser.add_argument(
        "--noise_schedule", type=NoiseSchedule, default=NoiseSchedule.sigmoid
    )
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--only_generate_sample", action="store_true")

    args = parser.parse_args()
    logger.info(f"{args}")

    downs = args.downs
    time_emb_dim = args.time_emb_dim
    zero_pad_images = args.zero_pad
    device = args.device
    dataset_name = args.dataset
    BATCH_SIZE = args.batch_size
    nepochs = args.nepochs
    lr = args.lr
    T = args.timesteps
    load_checkpoint = args.load_checkpoint
    model_name = args.model_name
    noise_schedule = args.noise_schedule
    model_tag = args.model_tag
    only_generate_sample = args.only_generate_sample

    IMG_SIZE = 32 if zero_pad_images else DEFAULT_IMG_SIZE

    torch.set_default_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    match noise_schedule:
        case NoiseSchedule.linear:
            betas = linear_beta_schedule(timesteps=T, device=device)
        case NoiseSchedule.cosine:
            betas = cosine_beta_schedule(timesteps=T, device=device)
        case NoiseSchedule.sigmoid:
            betas = sigmoid_beta_schedule(timesteps=T, device=device)

    # define difffusion parameters
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, -1)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)

    dataloader = get_dataloader(
        BATCH_SIZE,
        device,
        channels_last=False,
        zero_pad_images=zero_pad_images,
        dataset_name=dataset_name,
    )

    print(f"Number of training examples : {len(dataloader.dataset)}")
    logger.info(f"Checkpoint directory : {models_dir}")

    unet = load_model(
        models_dir,
        logger,
        downs,
        time_emb_dim,
        device,
        load_checkpoint,
        model_name,
    )

    # plot original samples to board
    writer = SummaryWriter()
    dataiter = iter(dataloader)
    images = next(dataiter)
    images = images["pixel_values"]
    img_grid = torchvision.utils.make_grid(images[:16])
    imshow(np.transpose(img_grid.cpu().numpy(), (1, 2, 0)), aspect="auto")
    writer.add_image("original fMNIST samples", img_grid)

    # add model to board
    writer.add_graph(unet, [images, torch.ones(BATCH_SIZE)])

    # train
    unet = LitUnet(
        unet=unet,
        betas=betas,
        sqrt_recip_alphas=sqrt_recip_alphas,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        posterior_variance=posterior_variance,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        T=T,
        device=device,
        lr=lr,
        writer=writer,
        img_size=IMG_SIZE,
    )
    trainer = L.Trainer(
        max_epochs=nepochs,
        accelerator="auto",
        callbacks=[
            LearningRateMonitor(),
        ],
        logger=TensorBoardLogger("tb_logs", name=model_name, log_graph=True),
    )
    if not only_generate_sample:
        trainer.fit(model=unet, train_dataloaders=dataloader)

    datetime_str = dt.datetime.today().strftime("%Y%m%d-%H%M")
    img_base_name = f"{script_name}_{datetime_str}"

    unet = unet.unet.to(device)
    unet.eval()
    name = (
        f"{unet._get_name()}{model_tag}_{dt.datetime.today().strftime('%Y%m%d-%H')}.pt"
    )
    location = os.path.join(models_dir, name)
    if not only_generate_sample:
        torch.save(unet.state_dict(), location)

    # generate samples
    logger.info("Generate sample")
    sample_base_name = f"sample_{script_name}_{datetime_str}_"
    n_samp = 32
    n_cols = 8
    SAMP_SHAPE = (n_samp, 1, 32, 32) if zero_pad_images else (n_samp, 1, 28, 28)

    _, axs = plt.subplots(
        nrows=n_samp // n_cols + ((n_samp % n_cols) > 0), ncols=n_cols, figsize=(16, 8)
    )

    samp = sample(
        unet,
        SAMP_SHAPE,
        T,
        betas,
        alphas_cumprod,
        alphas_cumprod_prev,
        posterior_variance,
    )[-1]

    # log samples to board
    write_sample_to_board(samp, writer, "generated samples")

    samp = samp.cpu().numpy()
    for i in range(n_samp):
        r, c = i // n_cols, i % n_cols
        axs[r, c].imshow(samp[i, 0, :, :], cmap="gray")
        axs[r, c].axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, sample_base_name + ".png"),
    )
