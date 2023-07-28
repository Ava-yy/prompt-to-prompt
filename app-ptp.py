from io import BytesIO
from base64 import b64encode
import abc
import re
import shutil
from glob import glob
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as nnf
from diffusers import DDIMScheduler, StableDiffusionPipeline
from flask import Flask, jsonify, request
from flask_cors import CORS
from natsort import natsorted
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

app = Flask(__name__)
CORS(app)


def encode_image(array):
    """
    Convert numpy array to base64 string that is ready for html image
    <img src='data:image/png;base64,'+src>
    Parameters
    ----------
    x: numpy array
        x.shape == [h,w,3]
        x.dtype == np.uint8
    Returns
    -------
    src: string
        a src string encoding image in base64
        ready for <img src='data:image/png;base64,'+src>
    """
    image = Image.fromarray(array)
    buff = BytesIO()
    image.save(buff, format="PNG")
    string = b64encode(buff.getvalue()).decode("utf-8")
    return string


@app.route("/", methods=["GET"])
def index():
    return "Hello!"


# save figures
# image_grid_pil = ptp_utils.view_images([*ptp_images])
# image_grid_pil.save("test.png")


# def ptp_refine():
#     # Test 2: ptp refine + reweight
#     prompts = [
#         # "a scene consisting of tables and chairs",
#         "a coffee shop scene consisting of tables and chairs",
#         "a coffee shop scene consisting of tables and chairs",
#     ]
#     cross_replace_steps = {
#         "default_": (0.0, 0.8),
#     }
#     self_replace_steps = 0.4
#     blend_words = None
#     eq_params = {
#         "words": ("coffee", "shop"),
#         "values": (20.0, 20.0),
#     }
#     controller = make_controller(
#         prompts,
#         False,
#         cross_replace_steps,
#         self_replace_steps,
#         blend_words,
#         eq_params,
#         tokenizer,
#         num_ddim_steps=NUM_DDIM_STEPS,
#     )


@app.route("/text2image", methods=["GET", "POST"])
def text2image():
    if request.method == "GET":
        # for testing
        prompt = "A photo of a cat riding a bike"
        seed = 0
        num_steps = 15
        guidance_scale = 7.5
    elif request.method == "POST":
        req = request.get_json()
        prompt = req.get("prompt")
        seed = req.get("seed", 0)
        num_steps = req.get("num_steps", 15)
        guidance_scale = req.get("guidance_scale", 7.5)

    ptp_images, _ = ptp_utils.text2image_ldm_stable(
        pipe,
        [prompt],
        EmptyControl(),
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        low_resource=True,
    )
    image = encode_image(ptp_images[0])
    return jsonify(
        dict(
            image=image,
        )
    )


@app.route("/replace", methods=["GET", "POST"])
def ptp_replace(
    prompts=[
        "A photo of a cat riding a bike",
        "A photo of a cat riding a car",
    ],
):
    if request.method == "POST":
        req = request.get_json()
        prompts = [req.get("prompt1"), req.get("prompt2")]
        seed = req.get("seed", 0)
        num_steps = req.get("num_steps", 15)
        guidance_scale = req.get("guidance_scale", 7.5)
        cross_replace_steps = {
            "default_": (0.0, 0.8),
        }
        self_replace_steps = 0.4
        blend_words = None
        eq_params = {
            "words": ("car", "riding"),
            "values": (5.0, 1.0),
        }
        # eq_params = None
    print(prompts)

    controller = make_controller(
        prompts,
        True,
        cross_replace_steps,
        self_replace_steps,
        blend_words,
        eq_params,
        tokenizer,
        num_ddim_steps=num_steps,
    )
    ptp_images, _ = ptp_utils.text2image_ldm_stable(
        pipe,
        prompts,
        controller,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        # latent=x_t,
        low_resource=True,
    )
    controller.reset()
    print("ptp_images.shape", ptp_images.shape)
    images = [encode_image(img) for img in ptp_images]
    return jsonify(
        dict(
            images=images,
        )
    )


if __name__ == "__main__":
    MY_TOKEN = ""
    LOW_RESOURCE = True
    NUM_DDIM_STEPS = 15
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
    # for module in [
    #     pipe.unet,
    #     pipe.vae,
    #     pipe.text_encoder,
    # ]:
    #     for p in module.parameters():
    #         p.requires_grad_(False)
    # pipe.enable_attention_slicing()
    # pipe.enable_vae_slicing()
    # pipe.enable_sequential_cpu_offload()
    tokenizer = pipe.tokenizer
    main_utils.mem()
    app.run(debug=True)
