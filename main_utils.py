import os
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

import ptp_utils

# from AttentionControl import AttentionStore
import torch


def makedirs(d):
    os.makedirs(d, exist_ok=True)


def mem(prefix=""):
    print(f"{prefix}{torch.cuda.memory_allocated()/1024**2:.2f}MB")


def free():
    torch.cuda.empty_cache()


# def aggregate_attention(
#     attention_store: AttentionStore,
#     res: int,
#     from_where: List[str],
#     is_cross: bool,
#     select: int,
# ):
#     out = []
#     attention_maps = attention_store.get_average_attention()
#     num_pixels = res**2
#     for location in from_where:
#         for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
#             if (
#                 item.shape[1] == num_pixels
#             ):  # only aggregate those with the supplied resolution (e.g., 16)
#                 cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[
#                     select
#                 ]
#                 out.append(cross_maps)
#     out = torch.cat(out, dim=0)
#     out = out.sum(0) / out.shape[0]
#     return out.cpu()


# def show_cross_attention(
#     attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0
# ):
#     tokens = tokenizer.encode(prompts[select])
#     decoder = tokenizer.decode
#     attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
#     images = []
#     for i in range(len(tokens)):
#         image = attention_maps[:, :, i]
#         image = 255 * image / image.max()
#         image = image.unsqueeze(-1).expand(*image.shape, 3)
#         image = image.numpy().astype(np.uint8)
#         image = np.array(Image.fromarray(image).resize((256, 256)))
#         image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
#         images.append(image)
#     ptp_utils.view_images(np.stack(images, axis=0))


# def show_self_attention_comp(
#     attention_store: AttentionStore,
#     res: int,
#     from_where: List[str],
#     max_com=10,
#     select: int = 0,
# ):
#     attention_maps = (
#         aggregate_attention(attention_store, res, from_where, False, select)
#         .numpy()
#         .reshape((res**2, res**2))
#     )
#     u, s, vh = np.linalg.svd(
#         attention_maps - np.mean(attention_maps, axis=1, keepdims=True)
#     )
#     images = []
#     for i in range(max_com):
#         image = vh[i].reshape(res, res)
#         image = image - image.min()
#         image = 255 * image / image.max()
#         image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
#         image = Image.fromarray(image).resize((256, 256))
#         image = np.array(image)
#         images.append(image)
#     ptp_utils.view_images(np.concatenate(images, axis=1))
