"""
Ray-based parallel inference script for DDPM sample generation.
This script loads a trained model and generates multiple samples in parallel using Ray.
"""
import os
import datetime as dt
import torch
import ray
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np

from utils_torch.unet import Unet, SimpleUnet
from utils_torch.diffusion import linear_beta_schedule


@ray.remote(num_gpus=0.25)  # Request 0.25 GPU per task (allows 4 parallel tasks per GPU)
def generate_sample_remote(
    model_state_dict,
    model_config,
    img_shape,
    T,
    betas,
    alphas_cumprod,
    alphas_cumprod_prev,
    posterior_variance,
    sample_id,
    device="cuda",
):
    """
    Remote function that generates a single sample using the diffusion model.

    Args:
        model_state_dict: State dict of the trained model
        model_config: Dict with model configuration (model_name, downs, time_emb_dim)
        img_shape: Shape of the image to generate (C, H, W)
        T: Number of diffusion timesteps
        betas: Beta schedule tensor
        alphas_cumprod: Cumulative product of alphas
        alphas_cumprod_prev: Previous cumulative product of alphas
        posterior_variance: Posterior variance tensor
        sample_id: Unique ID for this sample
        device: Device to run on

    Returns:
        Tuple of (sample_id, generated_image_numpy_array)
    """
    # Import here to avoid issues with Ray serialization
    from utils_torch.diffusion import sample

    # Set device
    torch.set_default_device(device)

    # Reconstruct the model
    if model_config["model_name"] == "Unet":
        model = Unet(
            downs=model_config["downs"],
            time_emb_dim=model_config["time_emb_dim"],
        ).to(device)
    elif model_config["model_name"] == "SimpleUnet":
        model = SimpleUnet(
            downs=model_config["downs"],
            time_emb_dim=model_config["time_emb_dim"],
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_config['model_name']}")

    # Load weights
    model.load_state_dict(model_state_dict)
    model.eval()

    # Move tensors to device
    betas = betas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    alphas_cumprod_prev = alphas_cumprod_prev.to(device)
    posterior_variance = posterior_variance.to(device)

    # Generate sample
    with torch.inference_mode():
        batch_shape = (1, *img_shape)  # Add batch dimension
        generated = sample(
            model,
            batch_shape,
            T,
            betas,
            alphas_cumprod,
            alphas_cumprod_prev,
            posterior_variance,
            pseudo_video=False,
        )

    # Return the final sample as numpy array (move to CPU first)
    final_sample = generated[-1][0].cpu().numpy()  # Remove batch dimension

    return sample_id, final_sample


def main():
    parser = ArgumentParser(description="Generate samples in parallel using Ray")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Unet",
        choices=["Unet", "SimpleUnet"],
        help="Model architecture",
    )
    parser.add_argument("--downs", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--time_emb_dim", type=int, default=4)
    parser.add_argument(
        "--img_size",
        type=int,
        default=28,
        help="Image size (28 for Fashion-MNIST, 32 if zero-padded)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Total number of samples to generate",
    )
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--ray_address",
        type=str,
        default="auto",
        help="Ray cluster address (use 'auto' for KubeRay, None for local)",
    )
    parser.add_argument(
        "--num_gpus_per_task",
        type=float,
        default=0.25,
        help="Number of GPUs per task (0.25 means 4 tasks per GPU)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./out",
        help="Output directory for generated samples",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Configuration: {args}")

    # Initialize Ray
    if args.ray_address == "None":
        # Local mode
        ray.init(ignore_reinit_error=True)
        print("Ray initialized in local mode")
    else:
        # Connect to Ray cluster (KubeRay)
        ray.init(address=args.ray_address, ignore_reinit_error=True)
        print(f"Ray initialized, connected to cluster at: {args.ray_address}")

    print(f"Ray cluster resources: {ray.cluster_resources()}")

    # Load model checkpoint (on CPU to avoid GPU memory issues)
    print(f"Loading model from {args.model_path}")
    model_state_dict = torch.load(args.model_path, map_location="cpu")

    # Prepare model config
    model_config = {
        "model_name": args.model_name,
        "downs": args.downs,
        "time_emb_dim": args.time_emb_dim,
    }

    # Prepare diffusion parameters
    T = args.timesteps
    betas = linear_beta_schedule(timesteps=T, device="cpu")  # Create on CPU
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=-1)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    posterior_variance = betas

    # Image shape (channels, height, width)
    img_shape = (1, args.img_size, args.img_size)

    # Put model state dict in Ray object store for efficient sharing
    model_state_ref = ray.put(model_state_dict)

    print(f"\nGenerating {args.n_samples} samples in parallel...")

    # Launch remote tasks
    # Update the remote function decorator dynamically
    generate_sample = ray.remote(num_gpus=args.num_gpus_per_task)(
        generate_sample_remote.options().remote
    )

    # Create tasks for all samples
    futures = [
        generate_sample_remote.remote(
            model_state_ref,
            model_config,
            img_shape,
            T,
            betas,
            alphas_cumprod,
            alphas_cumprod_prev,
            posterior_variance,
            i,
            args.device,
        )
        for i in range(args.n_samples)
    ]

    # Collect results as they complete
    print(f"Launched {len(futures)} tasks")
    results = []

    # Use ray.wait to get results as they complete
    remaining = futures
    while remaining:
        # Wait for at least one task to complete
        ready, remaining = ray.wait(remaining, num_returns=1)
        completed = ray.get(ready[0])
        results.append(completed)
        print(f"Progress: {len(results)}/{args.n_samples} samples generated")

    # Sort results by sample_id to maintain order
    results.sort(key=lambda x: x[0])
    samples = [img for _, img in results]

    print(f"\nGeneration complete! Generated {len(samples)} samples.")

    # Save samples
    datetime_str = dt.datetime.today().strftime("%Y%m%d-%H%M")
    sample_base_name = f"ray_samples_{datetime_str}"

    # Create a grid visualization
    grid_size = int(np.ceil(np.sqrt(min(args.n_samples, 100))))  # Max 100 in grid
    n_display = min(args.n_samples, grid_size * grid_size)

    fig, axs = plt.subplots(
        nrows=grid_size,
        ncols=grid_size,
        figsize=(grid_size * 2, grid_size * 2),
    )

    for i in range(n_display):
        r, c = i // grid_size, i % grid_size
        ax = axs[r, c] if grid_size > 1 else axs
        ax.imshow(samples[i][0, :, :], cmap="gray")
        ax.axis("off")

    # Hide unused subplots
    for i in range(n_display, grid_size * grid_size):
        r, c = i // grid_size, i % grid_size
        ax = axs[r, c] if grid_size > 1 else axs
        ax.axis("off")

    plt.tight_layout()
    output_path = os.path.join(args.out_dir, f"{sample_base_name}_grid.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved sample grid to: {output_path}")

    # Save all samples as numpy array
    samples_array = np.stack(samples, axis=0)
    np_output_path = os.path.join(args.out_dir, f"{sample_base_name}_all.npy")
    np.save(np_output_path, samples_array)
    print(f"Saved all samples to: {np_output_path}")

    # Shutdown Ray
    ray.shutdown()
    print("\nRay shutdown complete.")


if __name__ == "__main__":
    main()
