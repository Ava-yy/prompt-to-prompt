import abc
import shutil
from typing import Callable, Dict, List, Optional, Tuple, Union
from glob import glob
from natsort import natsorted

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
from AttentionControl import AttentionStore, EmptyControl, make_controller
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
    dtype = torch.float16
    device = (
        torch.device("cuda:0") if torch.cuda.is_available(
        ) else torch.device("cpu")
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

    # pipe.enable_attention_slicing()
    # pipe.enable_vae_slicing()
    # pipe.enable_sequential_cpu_offload()
    tokenizer = pipe.tokenizer

    output_dir = "figs/scene-example-2"
    main_utils.makedirs(output_dir)
    for data_path in glob("data/inverted_scenes/inverted-*.pth"):
        source_image_fn = data_path.split("/")[-1].split(".")[0].split("-")[1]
        shutil.copy(
            f"/home/zhaoy32/Desktop/understandingbdl/datasets/val_256/{source_image_fn}.jpg",
            output_dir + "/",
        )
        print(source_image_fn)
        continue

        # Load data from disk
        print(f"loading invesion at {data_path}...")
        data = torch.load(f"{data_path}", map_location="cpu")
        x_t = data.get("x_t").type(dtype)
        uncond_embeddings = [
            d.type(dtype)[:, :MAX_NUM_WORDS] for d in data.get("uncond_embeddings")
        ]
        prompt = data.get("prompt", "A scene consisting of tables and chairs")

        main_utils.mem()

        # Reconstruction from latent (x_t)
        # and learned null-text embeddings(uncond_embeddings)
        print("reconstruct...")
        prompts = [prompt]
        # controller = AttentionStore()
        controller = EmptyControl()
        null_inversion_images, x_t = inversion_utils.text2image_ldm_stable(
            pipe,
            prompts,
            controller,
            latent=x_t,
            generator=None,
            num_inference_steps=NUM_DDIM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            uncond_embeddings=uncond_embeddings,
        )
        image_grid_pil = ptp_utils.view_images([*null_inversion_images])
        image_grid_pil.save(f"{output_dir}/reconstruct-{source_image_fn}.png")
        # main_utils.show_cross_attention(controller, 16, ["up", "down"])
        del controller, null_inversion_images

        main_utils.mem()

        # make some edit
        class_labels = [
            "banquet_hall",
            "bar",
            "beer_hall",
            "cafeteria",
            "coffee_shop",
            "dining_hall",
            "fastfood_restaurant",
            "food_court",
            "restaurant_patio",
            "sushi_bar",
        ]
        for class_label in class_labels:
            print(f"{class_label}...")

            # prompt
            replacement_prompt = f"{''.join(class_label.split('_'))} scene"
            prompts = [prompt, prompt.replace("scene", replacement_prompt)]
            print(prompts)

            # config ptp
            cross_replace_steps = {
                "default_": 0.8,
            }
            self_replace_steps = 0.6
            blend_word = None
            eq_params = {
                "words": replacement_prompt.split(),
                "values": (10.0,) * len(replacement_prompt.split()),
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
            print(eq_params)

            # do ptp from inversion
            ptp_images, _ = inversion_utils.text2image_ldm_stable(
                pipe,
                prompts,
                controller,
                latent=x_t,
                generator=None,
                num_inference_steps=NUM_DDIM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                uncond_embeddings=uncond_embeddings,
            )

            # save figures
            image_grid_pil = ptp_utils.view_images([*ptp_images])
            image_grid_pil.save(
                f"{output_dir}/edit-{source_image_fn}-{class_label}.png"
            )
            controller.reset()
