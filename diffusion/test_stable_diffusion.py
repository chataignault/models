# https://huggingface.co/stabilityai/stable-diffusion-2-1

import torch
from argparse import ArgumentParser
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str, help="comma-seperated keywords")
    args = parser.parse_args()
    prompt = args.prompt

    model_id = "stabilityai/stable-diffusion-2-1"

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    image = pipe(prompt).images[0]

    image.save("image.png")
