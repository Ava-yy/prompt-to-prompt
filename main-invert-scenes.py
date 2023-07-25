import abc
import shutil
import sys
from glob import glob
from typing import Callable, Dict, List, Optional, Tuple, Union
import shutil

import numpy as np
import torch
import torch.nn.functional as nnf
from diffusers import DDIMScheduler, StableDiffusionPipeline
from natsort import natsorted
from PIL import Image
from torch.optim.adam import Adam
from tqdm.auto import tqdm

import inversion_utils
import main_utils
import ptp_utils
import seq_aligner
from AttentionControl import AttentionStore, EmptyControl, make_controller
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
    NUM_DDIM_STEPS = 50
    NUM_INNER_STEPS = 10
    GUIDANCE_SCALE = 7.5

    device = (
        torch.device("cuda:0") if torch.cuda.is_available(
        ) else torch.device("cpu")
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

    output_dir = "data/inverted_scenes"
    main_utils.makedirs(output_dir)

    image_paths = [
        f"/home/zhaoy32/Desktop/understandingbdl/datasets/places365_10c/val/{subpath}"
        for subpath in [
            # "banquet_hall/Places365_val_00005925.jpg",
            # "banquet_hall/Places365_val_00007819.jpg",
            # "banquet_hall/Places365_val_00008383.jpg",
            # "banquet_hall/Places365_val_00033235.jpg",
            # "banquet_hall/Places365_val_00035008.jpg",
            # "banquet_hall/Places365_val_00035479.jpg",
            # DOING new set
            "banquet_hall/Places365_val_00035008.jpg",
            "bar/Places365_val_00001763.jpg",
            "cafeteria/Places365_val_00025659.jpg",
            "dining_hall/Places365_val_00005023.jpg"
            "dining_hall/Places365_val_00020896.jpg",
            "dining_hall/Places365_val_00024614.jpg",
        ]
    ]
    for image_path in image_paths:
        main_utils.mem()
        soure_image_fn = image_path.split("/")[-1].split(".")[0]
        prompt = "A scene consisting of tables and chairs"
        print("inverting...")
        null_inversion = NullInversion(pipe, NUM_DDIM_STEPS, GUIDANCE_SCALE)
        (image_gt, image_ddim), x_t, uncond_embeddings = null_inversion.invert(
            image_path,
            prompt,
            # Modify or remove offsets according to your image!
            offsets=(0, 0, 0, 0),
            verbose=True,
            num_inner_steps=NUM_INNER_STEPS,
        )
        torch.save(
            dict(x_t=x_t, uncond_embeddings=uncond_embeddings, prompt=prompt),
            f"{output_dir}/inverted-{soure_image_fn}.pth",
        )

        # print("reconstruct...")
        # prompts = [prompt]
        # controller = EmptyControl()
        # null_inversion_images, _ = inversion_utils.text2image_ldm_stable(
        #     pipe,
        #     prompts,
        #     controller,
        #     latent=x_t,
        #     generator=None,
        #     num_inference_steps=NUM_DDIM_STEPS,
        #     guidance_scale=GUIDANCE_SCALE,
        #     uncond_embeddings=uncond_embeddings,
        # )
        # image_grid_pil = ptp_utils.view_images(
        #     [image_gt, *null_inversion_images])
        # image_grid_pil.save(f"{output_dir}/reconstructed-{soure_image_fn}.png")
        # del controller, null_inversion_images

        # main_utils.show_cross_attention(controller, 16, ["up", "down"])

        null_inversion.reset()
        for embed in uncond_embeddings:
            del embed
        del image_gt, image_ddim, x_t
        del null_inversion
        main_utils.mem()
        main_utils.free()
