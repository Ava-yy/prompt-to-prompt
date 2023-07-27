import abc
import shutil
from typing import Callable, Dict, List, Optional, Tuple, Union
from glob import glob
from natsort import natsorted
import re

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

# from AttentionControl import AttentionStore
from AttentionControl import EmptyControl, make_controller
from constants import MAX_NUM_WORDS
from NullInversion import NullInversion


# for i, image in enumerate(null_inversion_images):
#     image_pil = Image.fromarray(image)
#     image_pil.save(f"image-silver-{i}.png")

if __name__ == "__main__":
    MY_TOKEN = ""
    LOW_RESOURCE = True
    NUM_DDIM_STEPS = 50
    GUIDANCE_SCALE = 7.5
    seed = 0
    dtype = torch.float16
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=MY_TOKEN,
        scheduler=scheduler,
        torch_dtype=dtype,
    ).to(device)
    for module in [
        pipe.unet,
        pipe.vae,
        pipe.text_encoder,
    ]:
        for p in module.parameters():
            p.requires_grad_(False)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    # pipe.enable_sequential_cpu_offload()
    tokenizer = pipe.tokenizer

    main_utils.mem()

    # # Test 1 ptp replace
    # prompts = [
    #     # "A photo of a white male face",
    #     # "A photo of a Asian male face",
    #     "A photo of a cat riding a bike",
    #     "A photo of a cat riding a car",
    # ]
    # cross_replace_steps = {
    #     "default_": (0.0, 0.8),
    # }
    # self_replace_steps = 0.4
    # blend_word = None
    # eq_params = None
    # # eq_words = replacement_prompt.split()
    # # eq_params = {
    # #     "words": eq_words,
    # #     "values": (1.0,) * len(eq_words),
    # # }
    # controller = make_controller(
    #     prompts,
    #     True,
    #     cross_replace_steps,
    #     self_replace_steps,
    #     blend_word,
    #     eq_params,
    #     tokenizer,
    #     num_ddim_steps=NUM_DDIM_STEPS,
    # )

    # Test 2: ptp refine + reweight
    prompts = [
        # "a scene consisting of tables and chairs",
        "a coffee shop scene consisting of tables and chairs",
        "a coffee shop scene consisting of tables and chairs",
    ]
    cross_replace_steps = {
        "default_": (0.0, 0.8),
    }
    self_replace_steps = 0.4
    blend_word = None
    eq_params = {
        "words": ("coffee", "shop"),
        "values": (20.0, 20.0),
    }
    controller = make_controller(
        prompts,
        False,
        cross_replace_steps,
        self_replace_steps,
        blend_word,
        eq_params,
        tokenizer,
        num_ddim_steps=NUM_DDIM_STEPS,
    )

    # main process
    print(prompts)
    print(eq_params)
    ptp_images, _ = ptp_utils.text2image_ldm_stable(
        pipe,
        prompts,
        controller,
        num_inference_steps=NUM_DDIM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=torch.Generator().manual_seed(seed),
        # latent=x_t,
        low_resource=True,
    )

    # save figures
    image_grid_pil = ptp_utils.view_images([*ptp_images])
    image_grid_pil.save("test.png")
    controller.reset()
