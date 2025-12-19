import os
import jax
import torch
import datetime as dt
from jax import random
import jax.numpy as jnp
from tqdm import tqdm
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import orbax.checkpoint as ocp
from pathlib import Path

# https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html

from utils_jax.classes import UNetConv, UNet, SimpleUnet
from utils.dataloader import get_dataloader, get_grain_dataloader, Data
from utils_jax.training import (
    create_train_state,
    train_step,
    train_step_pmap,
    train_step_pmap_bf16,
    create_learning_rate_fn,
    SchedulerConfig,
)
from utils_jax.diffusion import sample, linear_beta_schedule, cosine_beta_schedule
from utils.logger import get_logger
from utils_jax.tpu_utils import (
    detect_tpu_environment,
    split_rng_for_devices,
)
from utils_jax.tensorboard_logger import DDPMTensorBoardLogger


def generate_samples_on_first_device(
    state,
    rng,
    num_devices,
    T,
    betas,
    alphas_cumprod,
    alphas_cumprod_prev,
    posterior_variance,
    channels: int,
    img_size: int,
):
    """Generate samples on first device only (for logging)."""
    # Unreplicate to first device if distributed
    if num_devices > 1:
        single_device_state = jax.tree.map(lambda x: x[0], state)
    else:
        single_device_state = state

    rng, subrng = random.split(rng)
    num_samples = 16
    sample_shape = (num_samples, img_size, img_size, channels)

    samples = sample(
        single_device_state,
        sample_shape,
        subrng,
        T,
        betas,
        alphas_cumprod,
        alphas_cumprod_prev,
        posterior_variance,
        pseudo_video=False,
    )[-1]

    return samples


def get_unet(model_name, channels, base_dim):
    if model_name == "UNet":
        unet = UNet(channels=channels, base_dim=base_dim)
    elif model_name == "UNetConv":
        unet = UNetConv(channels=channels)
    elif model_name == "SimpleUnet":
        unet = SimpleUnet(channels=channels)
    else:
        NotImplemented
    return unet


if __name__ == "__main__":
    script_name = "jax_unet"

    log_dir = os.path.join(os.getcwd(), "log")
    out_dir = os.path.join(os.getcwd(), "out")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(log_dir, script_name + ".log")
    date_format = "%Y-%m-%d %H:%M:%S"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger_name = "run_jax_unet"

    logger = get_logger(logger_name, log_format, date_format, log_file)

    parser = ArgumentParser(description="Run Attention Unet")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="SimpleUnet")
    parser.add_argument("--img_size", type=int, default=28)
    parser.add_argument("--base_dim", type=int, default=16)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--dataset", type=Data, default=Data.fashion_mnist)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--load-checkpoint", action="store_true")
    parser.add_argument("--checkpoint_name", type=str, default="")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=30,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1200,
        help="Generate samples every N steps",
    )
    # TPU-specific arguments
    parser.add_argument(
        "--use_tpu", type=bool, default=False, help="Use TPU if available"
    )
    parser.add_argument(
        "--use_mixed_precision",
        type=bool,
        default=True,
        help="Use bfloat16 mixed precision",
    )

    args = parser.parse_args()
    logger.info(f"{args}")

    device = args.device
    model_name = args.model_name
    channels = args.channels
    dataset = args.dataset
    base_dim = args.base_dim
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    nepochs = args.nepochs
    lr = args.lr
    T = args.timesteps
    load_checkpoint = args.load_checkpoint
    checkpoint_name = args.checkpoint_name

    datetime_str = dt.datetime.now().strftime("%Y%m%d%H%M")

    # TPU detection and setup
    if args.use_tpu:
        tpu_config = detect_tpu_environment()
        num_devices = tpu_config["device_count"] if tpu_config["is_tpu"] else 1
        logger.info(f"Detected {num_devices} devices: {tpu_config['device_type']}")

        if not tpu_config["is_tpu"]:
            logger.warning("TPU requested but not detected. Falling back to CPU/GPU.")
            args.use_tpu = False
            num_devices = 1
    else:
        num_devices = 1
        logger.info("Running in single-device mode")

    # Validate batch size for distributed training
    if num_devices > 1:
        assert BATCH_SIZE % num_devices == 0, (
            f"Batch size {BATCH_SIZE} must be divisible by num_devices {num_devices}"
        )
        per_device_batch_size = BATCH_SIZE // num_devices
        logger.info(f"Per-device batch size: {per_device_batch_size}")

    torch.set_default_device(device)

    # betas = linear_beta_schedule(timesteps=T)
    betas = cosine_beta_schedule(timesteps=T)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, -1)
    alphas_cumprod_prev = jnp.pad(
        alphas_cumprod[:-1], pad_width=(1, 0), mode="constant", constant_values=1.0
    )
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    sqrt_recip_alphas = 1.0 / jnp.sqrt(alphas)

    if args.use_tpu and num_devices > 1:
        logger.info("Using Grain dataloader for TPU")
        dataloader = get_grain_dataloader(
            batch_size=BATCH_SIZE,
            dataset_name=Data.fashion_mnist,
            num_devices=num_devices,
            num_epochs=nepochs,
        )
        # Estimate steps per epoch for grain (approximate)
        steps_per_epoch = 60000 // BATCH_SIZE  # FashionMNIST has ~60k training samples
    else:
        logger.info("Using PyTorch dataloader")
        dataloader = get_dataloader(BATCH_SIZE, device, dataset)
        steps_per_epoch = len(dataloader)

    unet = get_unet(model_name, channels, base_dim)

    config = SchedulerConfig(1, nepochs)
    learning_rate_fn = create_learning_rate_fn(config, lr, steps_per_epoch)

    rng = random.PRNGKey(56)
    rng_init, rng_train, rng_timestep, rng_final_sample = random.split(rng, 4)
    del rng

    state = create_train_state(
        rng_init, unet, learning_rate_fn, train=False, num_devices=num_devices
    )
    del rng_init
    ckpt_dir = Path("checkpoints").absolute()

    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    logger.info(f"Checkpoint directory: {ckpt_dir}")

    if load_checkpoint and len(checkpoint_name):
        logger.info(f"Loading checkpoint : {checkpoint_name}")
        tree = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
        state = ckptr.restore(ckpt_dir / checkpoint_name, tree)
        # print("Latest step: ", ckptr.latest_step())

    # Setup TensorBoard logger
    tb_log_dir = os.path.join(os.getcwd(), "tb_logs", datetime_str)
    tb_logger = DDPMTensorBoardLogger(log_dir=tb_log_dir)
    logger.info(f"TensorBoard logs: {tb_log_dir}")

    # Log parameter count (handle replicated state)
    if num_devices > 1:
        # Get first replica for counting
        params_for_count = jax.tree.map(lambda x: x[0], state.params)
        param_count = sum([p.size for p in jax.tree.leaves(params_for_count)])
    else:
        param_count = sum([p.size for p in jax.tree.leaves(state.params)])

    logger.info(f"Number of parameters: {param_count}")

    # Select training function based on settings
    if num_devices > 1:
        if args.use_mixed_precision:
            train_fn = train_step_pmap_bf16
            logger.info("Using distributed training with bfloat16 mixed precision")
        else:
            train_fn = train_step_pmap
            logger.info("Using distributed training with float32")

        # Split RNGs for devices
        rng_train_devices = split_rng_for_devices(rng_train, num_devices)
        rng_timestep_devices = split_rng_for_devices(rng_timestep, num_devices)
    else:
        train_fn = train_step
        logger.info("Using single-device training")

    global_step = 0
    loss_history = []
    lr_history = []

    for epoch in range(nepochs):
        if args.use_tpu and num_devices > 1:
            # Grain dataloader is an iterator
            pbar_batch = tqdm(enumerate(dataloader), total=steps_per_epoch, leave=True)
        else:
            pbar_batch = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)

        for k, batch in pbar_batch:
            if num_devices > 1:
                # Distributed training with pmap
                # batch shape: (num_devices, per_device_batch_size, H, W, C)

                # Split RNGs for this step
                rng_train_devices_split = jax.random.split(
                    rng_train_devices[0], num_devices + 1
                )
                rng_train_batch = rng_train_devices_split[1:]
                rng_train_devices = rng_train_devices_split[0:1]

                rng_timestep_devices_split = jax.random.split(
                    rng_timestep_devices[0], num_devices + 1
                )
                rng_timestep_batch = rng_timestep_devices_split[1:]
                rng_timestep_devices = rng_timestep_devices_split[0:1]

                # Generate per-device timesteps
                per_device_bs = BATCH_SIZE // num_devices
                timesteps = jax.random.randint(
                    rng_timestep_batch, (num_devices, per_device_bs), 1, T
                )

                diff_params = {
                    "t": timesteps,
                    "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
                    "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
                }

                state, train_loss, current_lr = train_fn(
                    state, batch, diff_params, rng_train_batch, learning_rate_fn
                )

                train_loss = float(train_loss[0])
                current_lr = float(current_lr[0])
            else:
                # Single-device training
                rng_train, rng_train_batch = random.split(rng_train)
                rng_timestep, rng_timestep_batch = random.split(rng_timestep)

                timestep = random.randint(rng_timestep_batch, (BATCH_SIZE,), 1, T)

                # Convert PyTorch batch to JAX (channel last)
                if args.use_tpu:
                    x = batch  # Grain already provides NHWC
                else:
                    x = jnp.permute_dims(
                        batch.numpy().astype(jnp.float32), (0, 2, 3, 1)
                    )

                diff_params = {
                    "t": timestep,
                    "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
                    "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
                }

                state, train_loss, current_lr = train_fn(
                    state, x, diff_params, rng_train_batch, learning_rate_fn
                )

                train_loss = float(train_loss)
                current_lr = float(current_lr)

            loss_history.append(train_loss)
            lr_history.append(current_lr)

            tb_logger.log_scalars(
                {
                    "train/loss": train_loss,
                    "train/learning_rate": current_lr,
                },
                global_step,
            )

            description = (
                f"Epoch {epoch} | "
                f"Step {global_step} | "
                f"Loss {train_loss:.7f} | LR {current_lr:.5f}"
            )
            pbar_batch.set_description(description)

            if global_step % args.sample_interval == 0 and global_step > 0:
                logger.info(f"Generating samples at step {global_step}")
                sample_images = generate_samples_on_first_device(
                    state,
                    rng,
                    num_devices,
                    T,
                    betas,
                    alphas_cumprod,
                    alphas_cumprod_prev,
                    posterior_variance,
                    channels,
                    IMG_SIZE,
                )
                tb_logger.log_images_grid(
                    "samples/generated", sample_images, global_step, nrow=4
                )

            global_step += 1

        # if epoch % args.checkpoint_interval == 0 and epoch > 1:
        #   pass

        logger.info(f"Epoch {epoch} complete | Loss {train_loss:.7f}")

    if nepochs > 0:
        logger.info("Saving final checkpoint")
        ckptr.save(
            ckpt_dir / "_".join([model_name, datetime_str]),
            args=ocp.args.StandardSave(state),
        )
        ckptr.wait_until_finished()

    img_base_name = f"{script_name}_{datetime_str}"

    _, ax = plt.subplots()
    ax.plot(range(len(loss_history)), loss_history)
    ax.grid()
    ax.set_title("Batch loss evolution")
    plt.savefig(os.path.join(out_dir, img_base_name + "_loss.png"), bbox_inches="tight")
    plt.close()

    _, ax = plt.subplots()
    ax.plot(range(len(lr_history)), lr_history)
    ax.grid()
    ax.set_title("Learning Rate evolution")
    plt.savefig(os.path.join(out_dir, img_base_name + "_lr.png"), bbox_inches="tight")
    plt.close()

    logger.info("Generating final samples")
    sample_base_name = f"sample_jax_{datetime_str}_"
    N_SAMPLE = 16
    N_COLS = 4

    samp = generate_samples_on_first_device(
        state,
        rng_final_sample,
        num_devices,
        T,
        betas,
        alphas_cumprod,
        alphas_cumprod_prev,
        posterior_variance,
        channels,
        IMG_SIZE,
    )

    tb_logger.log_images_grid("samples/final", samp, global_step, nrow=N_COLS)

    _, axs = plt.subplots(
        nrows=N_SAMPLE // N_COLS + ((N_SAMPLE % N_COLS) > 0),
        ncols=N_COLS,
        figsize=(16, 16),
    )

    for i in range(N_SAMPLE):
        r, c = i // N_COLS, i % N_COLS
        # Denormalize for display
        img_display = (samp[i, :, :, 0] + 1.0) / 2.0
        axs[r, c].imshow(img_display, cmap="gray")
        axs[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, sample_base_name + ".png"))
    plt.close()

    tb_logger.close()

    logger.info("Training completed successfully!")
    logger.info(f"Checkpoints saved to: {ckpt_dir}")
    logger.info(f"TensorBoard logs saved to: {tb_log_dir}")
    logger.info(f"Samples saved to: {out_dir}")
