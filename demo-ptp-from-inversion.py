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
from AttentionControl import AttentionStore, EmptyControl, make_controller
from constants import MAX_NUM_WORDS
from NullInversion import NullInversion


def mem(prefix=""):
    print(f"{prefix}{torch.cuda.memory_allocated()/1024**2:.2f}MB")


def free():
    torch.cuda.empty_cache()


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

    # Load data from disk
    data = torch.load("invert.pth", map_location="cpu")
    x_t = data.get("x_t").type(dtype)
    uncond_embeddings = [
        d.type(dtype)[:, :MAX_NUM_WORDS] for d in data.get("uncond_embeddings")
    ]

    # Vanilla reconstruction from latent (x_t)
    # and learned null-text embeddings(uncond_embeddings)
    mem()
    print("reconstruct...")
    prompt = "a cat sitting next to a mirror"
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
    image_grid_pil.save("image-invertion.png")
    # main_utils.show_cross_attention(controller, 16, ["up", "down"])
    del controller, null_inversion_images

    # replace
    mem()
    print("tiger...")
    prompts = ["a cat sitting next to a mirror",
               "a tiger sitting next to a mirror"]
    cross_replace_steps = {
        "default_": 0.8,
    }
    self_replace_steps = 0.5
    # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    blend_word = (
        ("cat",),
        ("tiger",),
    )
    eq_params = {
        "words": ("tiger",),
        "values": (2,),
    }  # amplify attention to the word "tiger" by *2
    controller = make_controller(
        prompts,
        True,
        cross_replace_steps,
        self_replace_steps,
        blend_word,
        eq_params,
        tokenizer,
        num_ddim_steps=NUM_DDIM_STEPS,
    )
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
    image_grid_pil = ptp_utils.view_images([*ptp_images])
    image_grid_pil.save("image-tiger.png")
    controller.reset()
    # del controller, ptp_images
    # free()

    # refine
    mem()
    print("silver...")
    prompts = [
        "a cat sitting next to a mirror",
        "a silver cat sculpture sitting next to a mirror",
    ]
    cross_replace_steps = {
        "default_": 0.8,
    }
    self_replace_steps = 0.6
    blend_word = (("cat",), ("cat",))  # for local edit
    eq_params = {
        "words": (
            "silver",
            "sculpture",
        ),
        "values": (
            2,
            2,
        ),
    }  # amplify attention to the words "silver" and "sculpture" by *2
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
    image_grid_pil = ptp_utils.view_images([*ptp_images])
    image_grid_pil.save("image-silver.png")
    controller.reset()
    # del controller, ptp_images
    # free()

    # refine
    mem()
    print("watercolor...")
    prompts = [
        "a cat sitting next to a mirror",
        "watercolor painting of a cat sitting next to a mirror",
    ]
    cross_replace_steps = {
        "default_": 0.8,
    }
    self_replace_steps = 0.7
    blend_word = None
    eq_params = {
        "words": ("watercolor",),
        "values": (
            5,
            2,
        ),
    }  # amplify attention to the word "watercolor" by 5
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
    image_grid_pil = ptp_utils.view_images([*ptp_images])
    image_grid_pil.save("image-watercolor.png")
    controller.reset()
    # del controller, ptp_images
    # free()
