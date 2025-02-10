# https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2

import torch
from argparse import ArgumentParser
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str, help="comma-separated keywords")
    args = parser.parse_args()
    prompt = args.prompt

    # Load the motion adapter
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16
    )
    # load SD 1.5 based finetuned model
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipe = AnimateDiffPipeline.from_pretrained(
        model_id, motion_adapter=adapter, torch_dtype=torch.float16
    )
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    output = pipe(
        prompt=prompt,
        negative_prompt="bad quality, worse quality",
        num_frames=15,
        guidance_scale=7.5,
        num_inference_steps=25,
        generator=torch.Generator("cpu").manual_seed(42),
    )
    frames = output.frames[0]
    export_to_gif(frames, "animation.gif")
