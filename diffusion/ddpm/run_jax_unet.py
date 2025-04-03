import os
import jax
import shutil
import datetime as dt
import orbax.checkpoint
from tqdm import tqdm
from argparse import ArgumentParser
from jax import random
from flax.training import orbax_utils
import jax.numpy as jnp
from matplotlib import pyplot as plt

from utils_jax.classes import UNetConv
from utils.dataloader import get_dataloader, DataSets
from utils_jax.training import (
    linear_beta_schedule,
    create_train_state,
    train_step,
    create_learning_rate_fn,
    SchedulerConfig,
)
from utils_jax.diffusion import sample
from utils.logger import get_logger


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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--img_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--load_checkpoint", type=bool, default=False)

    args = parser.parse_args()
    logger.info(f"{args}")

    device = args.device
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    nepochs = args.nepochs
    lr = args.lr
    T = args.timesteps
    load_checkpoint = args.load_checkpoint

    betas = linear_beta_schedule(timesteps=T)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, -1)
    alphas_cumprod_prev = jnp.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # ! check
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    sqrt_recip_alphas = 1.0 / jnp.sqrt(alphas)

    dataloader = get_dataloader(BATCH_SIZE, device, DataSets.fashion_mnist)

    unet = UNetConv()

    rng = random.PRNGKey(0)
    rng_init, rng_train, rng_timestep = random.split(rng, 3)
    ckpt_dir = os.path.join(os.getcwd(), "flax_ckpt")
    save_path = os.path.join(ckpt_dir, "orbax", "single_save")
    logger.info(f"Checkpoint directory : {save_path}")
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    config = SchedulerConfig(1, nepochs)

    learning_rate_fn = create_learning_rate_fn(config, lr, 937)
    if load_checkpoint:
        raw_restored = orbax_checkpointer.restore(save_path)
        state = raw_restored["model"]
    else:
        state = create_train_state(rng_init, unet, learning_rate_fn, train=False)

    logger.info(
        f"Number of parameters : {sum([p.size for p in jax.tree.leaves(state.params)])}"
    )

    loss_history = []
    lr_history = []

    for epoch in range(nepochs):
        pbar_batch = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
        diff_params = {
            "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        }
        for k, batch in pbar_batch:
            rng_train, rng_train_batch = random.split(rng_train)
            rng_timestep, rng_timestep_batch = random.split(rng_timestep)

            timestep = random.randint(rng_timestep_batch, (BATCH_SIZE,), 1, T)
            diff_params["t"] = timestep

            x = batch["pixel_values"].numpy().astype(jnp.float32)
            state, train_loss, lr = train_step(
                state, x, diff_params, rng_train_batch, learning_rate_fn
            )
            loss_history.append(train_loss)
            lr_history.append(lr)
            description = (
                f"Epoch {epoch} | "
                f"Step {k} | "
                f"Loss {train_loss:.7f} | Learning rate {lr:.5f}"
            )
            pbar_batch.set_description(description)

        logger.info(description)

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
    sample_base_name = f"sample_{datetime_str}_"
    rng, subrng = random.split(rng)
    samp = sample(
        state,
        (1, 28, 28, 1),
        subrng,
        T,
        posterior_variance,
        sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas,
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

    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    ckpt = {"model": state, "sample": samp}
    save_args = orbax_utils.save_args_from_target(ckpt)
    # orbax_checkpointer.save(save_path, ckpt, save_args=save_args) # TODO
