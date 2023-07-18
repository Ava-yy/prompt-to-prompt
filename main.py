from time import time

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

import ptp_utils
from AttentionEdits import AttentionReplace, AttentionStore, EmptyControl
from main_utils import makedirs, show_cross_attention

# MY_TOKEN = '<replace with your token>'
LOW_RESOURCE = True
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
# MAX_NUM_WORDS = 77


def run_and_display(
    pipe,
    prompts,
    controller,
    latent=None,
    run_baseline=False,
    generator=None
):
    if run_baseline:
        controller = EmptyControl()

    images, x_t = ptp_utils.text2image_ldm_stable(
        pipe,
        prompts,
        controller,
        latent=latent,
        generator=generator,
        num_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        low_resource=LOW_RESOURCE,
    )
    return images, x_t


if __name__ == "__main__":
    device = (torch.device("cuda:0")
              if torch.cuda.is_available() else torch.device("cpu"))
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        #     use_auth_token=MY_TOKEN
    ).to(device)

    prompts = [
        "A photo of a squirrel eating a burger",
        "A photo of a lion eating a burger",
    ]
    assert len(prompts) == 2, f'need two prompts (orignal, edit), got {len(prompts)}'

    timestamp = int(time())
    out_dir = (f'data_out/{"_".join(prompts[0].split())}-t{timestamp}')
    makedirs(out_dir)

    for seed in range(5):
        generator = torch.Generator().manual_seed(seed)
        controller = AttentionReplace(
            prompts, NUM_DIFFUSION_STEPS,
            cross_replace_steps=.8,
            self_replace_steps=.4,
            tokenizer=pipe.tokenizer,
        )
        images, x_t = run_and_display(
            pipe,
            prompts,
            controller,
            latent=None,
            run_baseline=False,
            generator=generator
        )
        for i, image in enumerate(images):
            pil_img = Image.fromarray(image)
            pil_img.save(f"{out_dir}/seed{seed:02d}-image{i}.png")
        # show_cross_attention(controller, res=16, from_where=("up", "down"))
