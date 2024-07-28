import os
import jax
import shutil
import logging
import orbax.checkpoint
from tqdm import tqdm
from argparse import ArgumentParser
from jax import random
from flax.training import orbax_utils
import jax.numpy as jnp
from matplotlib import pyplot as plt

from source.unet_jax import UNetAttention
from source.fashion_mnist_dataloader import get_dataloader
from source.utils_jax import (
    linear_beta_schedule,
    create_train_state,
    train_step,
    sample,
)


if __name__ == "__main__":
    logging.basicConfig(
        filename=__file__.rstrip(".py") + ".log",
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    logger = logging.getLogger("run_jax_attention")

    parser = ArgumentParser(description="Run Attention Unet")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--img_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--load_checkpoint", type=bool, default=False)

    args = parser.parse_args()
    device = args.device
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    nepochs = args.nepochs
    lr = args.lr
    T = args.timesteps
    load_checkpoint = args.load_checkpoint

    betas = linear_beta_schedule(timesteps=T)
    sqrt_alphas_cumprod = jnp.sqrt(jnp.cumprod(1.0 - betas, -1))
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - jnp.cumprod(1.0 - betas, -1))
    posterior_variance = betas
    sqrt_recip_alphas = 1.0 / jnp.sqrt(1 - betas)

    dataloader = get_dataloader(BATCH_SIZE, device)

    unet = UNetAttention()
    rng = random.PRNGKey(0)
    rng_train, rng_timestep = random.split(rng)
    ckpt_dir = os.path.join(os.getcwd(), "flax_ckpt")
    save_path = os.path.join(ckpt_dir, "orbax", "single_save")
    logger.info(f"Checkpoint directory : {save_path}")
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    if load_checkpoint:
        raw_restored = orbax_checkpointer.restore(save_path)
        state = raw_restored["model"]
    else:
        state = create_train_state(rng, lr, unet)

    logger.info(
        f"Number of parameters : {sum([p.size for p in jax.tree.leaves(state.params)])}"
    )

    loss_history = []

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
            state, train_loss = train_step(state, x, diff_params, rng_train_batch)
            loss_history.append(train_loss)
            description = f"Epoch {epoch} | " f"Step {k} | " f"Loss {train_loss:.4f} | "
            logger.info(description)
            pbar_batch.set_description(description)

    _, ax = plt.subplots()
    ax.plot(range(len(loss_history)), loss_history)
    ax.grid()
    ax.set_title("Batch loss evolution")
    plt.show()

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

    for im in samp[:: (T // 4)]:
        plt.imshow(im.reshape(28, 28), cmap="gray")
        plt.show()
    plt.imshow(samp[-1].reshape(28, 28), cmap="gray")

    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    ckpt = {"model": state, "sample": samp}
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(save_path, ckpt, save_args=save_args)
