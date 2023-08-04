# from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
import torch
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
)
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from glob import glob
from natsort import natsorted

import inversion_utils


class EDICTScheduler(DDIMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def guide(self, pred_uncond, pred_text, guidance_scale):
        noise_pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
        return noise_pred

    def epsilon(self, unet, latent, t, text_embeddings, guidance_scale):
        latents = latent.expand(2, -1, -1, -1)
        # latents = scheduler.scale_model_input(latents, timestep=t) # This does nothing in DDIM
        raw_noise_pred = unet(latents, t, encoder_hidden_states=text_embeddings).sample
        return self.guide(
            pred_uncond=raw_noise_pred[0:1],
            pred_text=raw_noise_pred[1:2],
            guidance_scale=guidance_scale,
        )

    def guided_step(
        self,
        x_t,  # the latent intermediate step x_t, to be updated to x_{t-1} here
        y_t,  # the latent intermediate step y_t, to be updated to x_{t-1} here
        timestep,
        timestep_prev,
        text_embeddings,
        unet=None,
        guidance_scale=7.5,
        mixing_factor=0.93,
    ):
        def epsilon(x, t):
            """encapsulate some variables for conciseness"""
            return self.epsilon(
                unet, x.type(unet.dtype), t, text_embeddings, guidance_scale
            )

        # Ref: EDICT(https://arxiv.org/abs/2211.12446)
        # 1. Get previous step value (=t-1)
        t = timestep - 1
        t_prev = timestep_prev - 1
        # print(t, t_prev)
        # 2. Alphas (cumulative)
        T = self.alphas_cumprod.shape[0] - 1
        alpha_t = self.alphas_cumprod[t] if t <= T else self.alphas_cumprod[T]
        alpha_t_prev = (
            self.alphas_cumprod[t_prev] if t_prev >= 0 else self.alphas_cumprod[0]
        )
        # alpha_t = alpha_t.double()
        # alpha_t_prev = alpha_t_prev.double()
        # see eq(3) and (4) from EDICT (https://arxiv.org/abs/2211.12446)
        a_t = (alpha_t_prev / alpha_t) ** 0.5
        b_t = -((alpha_t_prev * (1 - alpha_t) / alpha_t) ** 0.5) + (
            (1 - alpha_t_prev) ** 0.5
        )
        # print(t.item(), t_prev.item(), a_t.item(), b_t.item())

        # eq (14)
        p = mixing_factor  # shortname
        x_inter = a_t * x_t + b_t * epsilon(y_t, t)
        y_inter = a_t * y_t + b_t * epsilon(x_inter, t)
        x_prev = p * x_inter + (1 - p) * y_inter
        y_prev = p * y_inter + (1 - p) * x_prev

        # print("t", t.item(), t_prev.item())
        # print("a_t, b_t", a_t.item(), b_t.item())
        # print("epsilon", epsilon(x_inter, t)[0, 0, 0, :2].tolist())
        # print("x_t", x_t[0, 0, 0, :2].tolist())
        # print("y_t", y_t[0, 0, 0, :2].tolist())
        # print("x_inter", x_inter[0, 0, 0, :2].tolist())
        # print("y_inter", y_inter[0, 0, 0, :2].tolist())
        # print("x_prev", x_prev[0, 0, 0, :2].tolist())
        # print("y_prev", y_prev[0, 0, 0, :2].tolist())

        return dict(
            x_prev=x_prev,
            y_prev=y_prev,
        )

    def guided_step_invert(
        self,
        x_t,  # the latent intermediate step x_t, to be updated to x_{t-1} here
        y_t,  # the latent intermediate step x_t, to be updated to x_{t-1} here
        timestep,
        timestep_next,
        text_embeddings,
        unet=None,
        guidance_scale=7.5,
        mixing_factor=0.93,
    ):
        def epsilon(x, t):
            """encapsulate some variables for conciseness"""
            return self.epsilon(
                unet, x.type(unet.dtype), t, text_embeddings, guidance_scale
            )

        # Ref: EDICT(https://arxiv.org/abs/2211.12446)
        # 1. Get previous step value (=t-1)
        T = self.alphas_cumprod.shape[0] - 1
        t = timestep - 1
        t_next = timestep_next - 1
        # 2. Alphas (cumulative)
        alpha_t = self.alphas_cumprod[t] if t >= 0 else self.alphas_cumprod[0]
        alpha_t_next = self.alphas_cumprod[t_next] if t <= T else self.alphas_cumprod[T]

        # alpha_t = alpha_t.double()
        # alpha_t_next = alpha_t_next.double()
        # see eq(3) and (4) from EDICT (https://arxiv.org/abs/2211.12446)
        a_t_next = (alpha_t / alpha_t_next) ** 0.5
        b_t_next = -((alpha_t * (1 - alpha_t_next) / alpha_t_next) ** 0.5) + (
            (1 - alpha_t) ** 0.5
        )
        # print(t.item(), t_next.item(), a_t_next.item(), b_t_next.item())

        # eq (15)
        p = mixing_factor  # shortname
        y_inter = (y_t - (1 - p) * x_t) / p
        x_inter = (x_t - (1 - p) * y_inter) / p
        y_next = (y_inter - b_t_next * epsilon(x_inter, t_next)) / a_t_next
        x_next = (x_inter - b_t_next * epsilon(y_next, t_next)) / a_t_next

        # print("t", t.item(), t_next.item())
        # print("a_t, b_t", a_t_next.item(), b_t_next.item())
        # print("epsilon", epsilon(x_inter, t_next)[0, 0, 0, :2].tolist())
        # print("x_t", x_t[0, 0, 0, :2].tolist())
        # print("y_t", y_t[0, 0, 0, :2].tolist())
        # print("y_inter", y_inter[0, 0, 0, :2].tolist())
        # print("x_inter", x_inter[0, 0, 0, :2].tolist())
        # print("y_next", y_next[0, 0, 0, :2].tolist())
        # print("x_next", x_next[0, 0, 0, :2].tolist())

        return dict(
            x_next=x_next,
            y_next=y_next,
        )


#  PROCESSING & VIS UTILS
def process_prompts(prompts, tokenizer, text_encoder, device="cuda"):
    with torch.no_grad():
        text = tokenizer(
            prompts,
            max_length=20,  # tokenizer.model_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",  # return data type = pytorch tensor
        )
        text_embeddings = text_encoder(text.input_ids.to(device))[0]
    return text_embeddings


def decode(latents, vae, scale=1 / 0.18215):
    # scale and decode the image latents with vae
    with torch.no_grad():
        return vae.decode(scale * latents).sample


def encode(image, vae, scale=0.18215):
    latents = vae.encode(image)["latent_dist"].mean
    latents = latents * scale
    return latents


def show_images(x):
    """
    x: Tensor
    shape = [n_image, n_channels, height, width]
    ranges normally from -1 to 1
    """
    images = (x / 2 + 0.5).clamp(0, 1)
    images = make_grid(images.cpu(), padding=2)
    plt.imshow(images.permute(1, 2, 0))
    plt.axis("off")


#  HIGH-LEVEL APIs
def denoise(
    pipe=None,
    latents=None,
    x_t=None,
    y_t=None,
    text_embeddings=None,
    timesteps=None,
    guidance_scale=None,
    mixing_factor=None,
):
    if x_t is None and y_t is None:
        x_t = latents
        y_t = latents

    for i, [t, t_prev] in enumerate(tqdm(list(zip(timesteps[:-1], timesteps[1:])))):
        with torch.no_grad():
            step_output = scheduler.guided_step(
                x_t,
                y_t,
                timestep=t,
                timestep_prev=t_prev,
                text_embeddings=text_embeddings,
                unet=unet,
                guidance_scale=guidance_scale,
                mixing_factor=mixing_factor,
            )
            x_t, y_t = step_output.get("x_prev"), step_output.get("y_prev")
    return x_t, y_t


def invert(
    pipe=None,
    latents=None,
    x_t=None,
    y_t=None,
    text_embeddings=None,
    timesteps=None,
    guidance_scale=None,
    mixing_factor=None,
):
    if x_t is None and y_t is None:
        x_t = latents
        y_t = latents

    for i, [t, t_next] in enumerate(
        tqdm(list(zip(timesteps[::-1][:-1], timesteps[::-1][1:])))
    ):
        with torch.no_grad():
            step_output = pipe.scheduler.guided_step_invert(
                x_t,
                y_t,
                timestep=t,
                timestep_next=t_next,
                text_embeddings=text_embeddings,
                unet=pipe.unet,
                guidance_scale=guidance_scale,
                mixing_factor=mixing_factor,
            )
            x_t, y_t = step_output.get("x_next"), step_output.get("y_next")
    return x_t, y_t


#  UNIT TESTS
def test_invert_synthetic(
    pipe,
    generator,
    text_embeddings,
    guidance_scale=7.5,
    num_steps=50,
    mixing_factor=0.93,
    image_filepath_out="EDICT-invert-synthetic.png",
    device="cuda",
):
    latents = torch.randn(
        [1, unet.config.in_channels, 64, 64],
        generator=generator,
        dtype=dtype,
    )
    latents = latents * scheduler.init_noise_sigma
    latents = latents.to(device)

    # generation
    scheduler.set_timesteps(num_steps)
    timesteps = [torch.tensor(1000), *list(scheduler.timesteps)]
    # timesteps = list(scheduler.timesteps)

    x0, y0 = denoise(
        latents=latents,
        text_embeddings=text_embeddings,
        pipe=pipe,
        timesteps=timesteps,
        guidance_scale=guidance_scale,
        mixing_factor=mixing_factor,
    )
    z0 = x0.type(dtype)  # for vis to decode

    # inversion
    print("inversion")
    xT, yT = invert(
        latents=x0,
        text_embeddings=text_embeddings,
        pipe=pipe,
        timesteps=timesteps,
        guidance_scale=guidance_scale,
        mixing_factor=mixing_factor,
    )

    print("Inversion average per-pixel errors: ")
    print(
        "|X_T - init_latent|.mean() =",
        (xT - latents).abs().mean().item(),
    )
    print(
        "|Y_T - init_latent|.mean() =",
        (yT - latents).abs().mean().item(),
    )

    x0_recon, y0_recon = denoise(
        x_t=xT,
        y_t=yT,
        text_embeddings=text_embeddings,
        pipe=pipe,
        timesteps=timesteps,
        guidance_scale=guidance_scale,
        mixing_factor=mixing_factor,
    )
    z1 = x0_recon.type(dtype)  # for vis to decode

    # vis
    plt.figure(figsize=[12, 6])
    plt.subplot(121)
    images = decode(z0, vae)
    show_images(images)
    plt.subplot(122)
    images = decode(z1, vae)
    show_images(images)
    plt.savefig(image_filepath_out, bbox_inches="tight")
    plt.close()
    # plt.show()


def test_invert_real(
    pipe,
    image_filepath,
    text_embeddings,
    guidance_scale=7.5,
    num_steps=50,
    mixing_factor=0.93,
    image_filepath_out="EDICT-invert-real.png",
):
    # [x] load image
    # [x] encode image
    # invert image to x_T, y_T
    # denoise from x_T, y_T,
    # compare denoised to real, savefig
    dtype = pipe.vae.dtype
    device = pipe.vae.device
    image = inversion_utils.load_512(image_filepath)
    image = torch.from_numpy(image).type(dtype) / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)
    latents = encode(
        image,
        pipe.vae,
    )
    z0 = latents.type(dtype)  # for vis to decode

    scheduler.set_timesteps(num_steps)
    timesteps = [torch.tensor(1000), *list(scheduler.timesteps)]

    # inversion
    print("inversion...")
    xT, yT = invert(
        latents=latents,
        text_embeddings=text_embeddings,
        pipe=pipe,
        timesteps=timesteps,
        guidance_scale=guidance_scale,
        mixing_factor=mixing_factor,
    )

    print("reconstruction...")
    x0_recon, y0_recon = denoise(
        x_t=xT,
        y_t=yT,
        latents=None,
        text_embeddings=text_embeddings,
        pipe=pipe,
        timesteps=timesteps,
        guidance_scale=guidance_scale,
        mixing_factor=mixing_factor,
    )
    z1 = x0_recon.type(dtype)  # for vis to decode

    # vis
    plt.figure(figsize=[12, 6])
    plt.subplot(121)
    images = decode(z0, vae)
    show_images(images)
    plt.subplot(122)
    images = decode(z1, vae)
    show_images(images)
    plt.savefig(image_filepath_out, bbox_inches="tight")
    plt.close()


# ------------ MAIN ---------------
if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float32  # torch.float16 or torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",  # 512x512
        torch_dtype=dtype,
        # scheduler=DDPMScheduler(),
        scheduler=EDICTScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        ),
    )
    pipe = pipe.to(device)
    for module in [
        pipe.unet,
        pipe.vae,
        pipe.text_encoder,
    ]:
        for p in module.parameters():
            p.requires_grad_(False)
    # pipe.unet.type(torch.float64)
    # pipe.set_progress_bar_config(disable=True)
    # pipe.enable_attention_slicing()
    # pipe.enable_vae_slicing()
    # pipe.enable_sequential_cpu_offload()
    scheduler = pipe.scheduler
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    guidance_scale = 7.5
    num_steps = 50
    mixing_factor = 0.93  # EDICT param "p", typical: 0.93, recommended: [0.9, 0.97]

    # # TEST 1
    # prompts = ["", "A photo of a cat sitting next to a dog"]
    # generator = torch.manual_seed(8)
    # text_embeddings = process_prompts(prompts, tokenizer, text_encoder)
    # test_invert_synthetic(
    #     pipe, generator, text_embeddings, guidance_scale, num_steps, mixing_factor
    # )

    # # TEST 2
    # prompts = ["", "Whatever prompt used here seems not matter for inversion"]
    # text_embeddings = process_prompts(prompts, tokenizer, text_encoder)
    # test_invert_real(
    #     pipe=pipe,
    #     image_filepath="example_images/gnochi_mirror.jpeg",
    #     text_embeddings=text_embeddings,
    #     guidance_scale=7.5,
    #     num_steps=50,
    #     mixing_factor=0.93,
    # )

    prompts = ["", "Whatever prompt used here seems not matter for inversion"]
    text_embeddings = process_prompts(prompts, tokenizer, text_encoder)
    image_fns = natsorted(
        glob("/home/zhaoy32/Desktop/understandingbdl/datasets/val_256/*.jpg")
    )
    for image_filepath in image_fns[:5]:
        print(image_filepath)
        fn_out = image_filepath.split("/")[-1].split(".")[0]
        test_invert_real(
            pipe=pipe,
            image_filepath=image_filepath,
            text_embeddings=text_embeddings,
            guidance_scale=7.5,
            num_steps=50,
            mixing_factor=0.93,
            image_filepath_out=f"EDICT-reconstruct-images/{fn_out}.png",
        )
