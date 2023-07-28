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

import ptp_utils
import seq_aligner
from constants import MAX_NUM_WORDS


# ## Prompt-to-Prompt code
class MaskBlend:
    def __call__(self, x_t):
        if x_t.device != self.mask.device:
            self.mask = self.mask.to(x_t.device)
        if x_t.dtype != self.mask.dtype:
            self.mask = self.mask.type(x_t.dtype)
        x_t[1:] = x_t[1:] * self.mask + x_t[:1] * (1 - self.mask)
        return x_t

    def __init__(self, mask=torch.ones(64, 64)):
        self.mask = mask


class LocalBlend:
    def get_mask(self, maps, x_t, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1 - int(use_pool)])
        mask = mask[:1] + mask
        return mask

    # TODO make a manual mask version of this
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [
                item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS)
                for item in maps
            ]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, x_t, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, x_t, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.type(x_t.dtype)
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(
        self,
        prompts: List[str],
        words: [List[List[str]]],
        substruct_words=None,
        start_blend=0.2,
        th=(0.3, 0.3),
        tokenizer=None,
        num_ddim_steps=50,
        device="cuda",
    ):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0
        self.th = th


class EmptyControl:
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.low_resource else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.low_resource:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(
        self,
        low_resource=True,
    ):
        self.low_resource = low_resource
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class SpatialReplace(EmptyControl):
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float, num_ddim_steps=50):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * num_ddim_steps)


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if self.should_store_attn(attn):
            self.step_store[key].append(attn.to(self.store_device))
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        # for attention_list in self.step_store.values():
        #     for a in attention_list:
        #         del a
        # for attention_list in self.attention_store.values():
        #     for a in attention_list:
        #         del a
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(
        self,
        should_store_attn=lambda x: False,  # lambda x: x.shape[2] <= 32**2
        store_device="cpu",
        low_resource=True,
    ):
        super(AttentionStore, self).__init__(
            low_resource=low_resource,
        )
        # gather attentions of the same part of U-net (down, mid, or up) in a list
        # e.g., self.step_store['down_cross'] = [down_64x64, down_32x32, ...]
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.should_store_attn = should_store_attn
        self.store_device = store_device


class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        if self.local_blend is not None:
            print("AttentionControlEdit: self.local_blend()")
            x_t = self.local_blend(x_t, self.attention_store)
        if self.mask is not None:
            crs = self.cross_replace_steps
            if crs[0] <= self.cur_step < crs[1]:
                # print("AttentionControlEdit: self.mask()")
                x_t = self.mask(x_t)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[3] <= 64**2:
            attn_base = attn_base.unsqueeze(0).expand(
                att_replace.shape[0], *attn_base.shape
            )
            return attn_base
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (
            self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]
        ):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                # print("replace_cross_attention...")
                attn_repalce_new = (
                    self.replace_cross_attention(attn_base, attn_repalce) * alpha_words
                    + (1 - alpha_words) * attn_repalce
                )
                # attn.shape == torch.Size([2, 2, 8, 256, 77])
                # attn.shape == [prompts, null/new, heads, features, tokens]
                attn[1:] = attn_repalce_new
            else:
                # print("replace_self_attention...")
                attn[1:] = self.replace_self_attention(
                    attn_base, attn_repalce, place_in_unet
                )
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: Union[
            float, Tuple[float, float], Dict[str, Tuple[float, float]]
        ],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
        mask=None,
        tokenizer=None,
        device="cuda",
        low_resource=True,
    ):
        print("AttentionControlEdit init", mask)
        super(AttentionControlEdit, self).__init__(low_resource=low_resource)
        self.batch_size = len(prompts)
        self.num_steps = num_steps
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, tokenizer
        ).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        if type(cross_replace_steps) is float:
            cross_replace_steps = 0, int(cross_replace_steps * num_steps)
        self.cross_replace_steps = cross_replace_steps

        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(
            num_steps * self_replace_steps[1]
        )
        self.local_blend = local_blend
        self.mask = mask


class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        # print(attn_base.shape, self.mapper.shape)
        return torch.einsum(
            "...hpw,bwn->bhpn", attn_base, self.mapper.type(attn_base.dtype)
        )

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        tokenizer=None,
        device="cuda",
        low_resource=True,
    ):
        super(AttentionReplace, self).__init__(
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
            low_resource=low_resource,
        )
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        # print("AttentionRefine.replace_cross_attention()")
        # print("attn_base.shape", attn_base.shape)
        # print(self.mapper.shape)
        # print(attn_base[:, :, self.mapper].shape)

        attn_base_replace = attn_base[0, :, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        tokenizer=None,
        device="cuda",
        low_resource=True,
        mask=None,
    ):
        super(AttentionRefine, self).__init__(
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
            low_resource=low_resource,
            mask=mask,
        )
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(
                attn_base, att_replace
            )
        if self.equalizer.dtype != attn_base.dtype:
            self.equalizer = self.equalizer.type(attn_base.dtype)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        equalizer,
        local_blend: Optional[LocalBlend] = None,
        controller: Optional[AttentionControlEdit] = None,
        device="cuda",
        low_resource=True,
        mask=None,
    ):
        super(AttentionReweight, self).__init__(
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
            low_resource=low_resource,
            mask=mask,
        )
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(
    text: str,
    word_select: Union[int, Tuple[int, ...]],
    values: Union[List[float], Tuple[float, ...]],
    tokenizer=None,
):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, MAX_NUM_WORDS)

    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


def make_controller(
    prompts: List[str],
    is_replace_controller: bool,
    cross_replace_steps: Dict[str, float],
    self_replace_steps: float,
    blend_words=None,
    equilizer_params=None,
    tokenizer=None,
    num_ddim_steps=50,
    mask=torch.ones(64, 64),
) -> AttentionControlEdit:
    if mask is not None:
        mask = MaskBlend(mask)
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer=tokenizer)

    if is_replace_controller:
        print("make_controller(): creating AttentionReplace")
        controller = AttentionReplace(
            prompts,
            num_ddim_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            local_blend=lb,
            tokenizer=tokenizer,
        )
    else:
        print("make_controller(): creating AttentionRefine")
        controller = AttentionRefine(
            prompts,
            num_ddim_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            local_blend=lb,
            tokenizer=tokenizer,
        )

    if equilizer_params is not None:
        print("make_controller(): creating AttentionReweight")
        eq = get_equalizer(
            prompts[1],
            equilizer_params["words"],
            equilizer_params["values"],
            tokenizer=tokenizer,
        )
        controller = AttentionReweight(
            prompts,
            num_ddim_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            equalizer=eq,
            local_blend=lb,
            controller=controller,
            mask=mask,
        )
    return controller
