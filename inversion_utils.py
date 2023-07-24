import abc
import shutil
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as nnf
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from torch.optim.adam import Adam
from tqdm.auto import tqdm

import ptp_utils
import seq_aligner
from constants import MAX_NUM_WORDS


# ## Null Text Inversion code
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top: h - bottom, left: w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset: offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset: offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


# ## Inference Code
@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type="image",
):
    dtype = model.unet.dtype
    device = model.device

    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        # max_length=model.tokenizer.model_max_length,
        max_length=MAX_NUM_WORDS,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings_ = model.text_encoder(
            uncond_input.input_ids.to(model.device)
        )[0]
    else:
        uncond_embeddings_ = None
    latent, latents = ptp_utils.init_latent(
        latent, model, height, width, generator, batch_size
    )
    # MOD: half()
    latent = latent.type(dtype)
    latents = latents.type(dtype)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            uncond = uncond_embeddings[i].expand(
                *text_embeddings.shape).to(device)
            context = torch.cat(
                [
                    uncond,
                    text_embeddings,
                ]
            )
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(
            model, controller, latents, context, t, guidance_scale, low_resource=True
        )
        latents = latents.type(dtype)  # MOD: half()
    if return_type == "image":
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


# def run_and_display(
#     pipe,
#     prompts,
#     controller,
#     latent=None,
#     run_baseline=False,
#     generator=None,
#     uncond_embeddings=None,
#     verbose=True,
#     num_inference_steps=50,
#     guidance_scale=7.5,
# ):
#     if run_baseline:
#         print("w.o. prompt-to-prompt")
#         images, latent = run_and_display(
#             prompts,
#             EmptyControl(),
#             latent=latent,
#             run_baseline=False,
#             generator=generator,
#         )
#         print("with prompt-to-prompt")
#     images, x_t = text2image_ldm_stable(
#         pipe,
#         prompts,
#         controller,
#         latent=latent,
#         num_inference_steps=num_inference_steps,
#         guidance_scale=guidance_scale,
#         generator=generator,
#         uncond_embeddings=uncond_embeddings,
#     )
#     if verbose:
#         ptp_utils.view_images(images)
#     return images, x_t
