import abc
import shutil
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as nnf
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from torch.optim.adam import Adam
from tqdm.notebook import tqdm

import inversion_utils
import main_utils
import ptp_utils
import seq_aligner
from AttentionControl import AttentionStore, make_controller
from NullInversion import NullInversion

if __name__ == "__main__":
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    MY_TOKEN = ""
    LOW_RESOURCE = True
    NUM_DDIM_STEPS = 20
    GUIDANCE_SCALE = 7.5
    # MAX_NUM_WORDS = 77

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=MY_TOKEN,
        scheduler=scheduler,
        torch_dtype=torch.float32,
    ).to(device)
    for module in [
        pipe.unet,
        pipe.vae,
        pipe.text_encoder,
    ]:
        for p in module.parameters():
            p.requires_grad_(False)
    # pipe.enable_attention_slicing()
    # pipe.enable_sequential_cpu_offload()
    # pipe.enable_vae_slicing()
    tokenizer = pipe.tokenizer

    image_path = "./example_images/gnochi_mirror.jpeg"
    prompt = "a cat sitting next to a mirror"

    null_inversion = NullInversion(pipe, NUM_DDIM_STEPS, GUIDANCE_SCALE)
    (image_gt, image_ddim), x_t, uncond_embeddings = null_inversion.invert(
        image_path,
        prompt,
        offsets=(0, 0, 200, 0),  ## Modify or remove offsets according to your image!
        verbose=True,
        num_inner_steps=10,
    )
    torch.save(dict(x_t=x_t, uncond_embeddings=uncond_embeddings), 'invert.pth')

    # save images from inverted latents
    print(
        f"null_inversion_images.shape: {null_inversion_images.shape}",
        f"x_t.shape: {x_t.shape}",
    )
    # for i, image in enumerate(null_inversion_images):
    #     image_pil = Image.fromarray(image)
    #     image_pil.save(f"image-{i}.png")
    image_grid_pil = ptp_utils.view_images(
        [image_gt, image_ddim, null_inversion_images[0]]
    )
    image_grid_pil.save("image-invertion.png")
    # main_utils.show_cross_attention(controller, 16, ["up", "down"])

