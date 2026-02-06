# Copyright (c) Meta Platforms, Inc. and affiliates.
import time
from collections import defaultdict
from safetensors.torch import load_file

import torch
import math

from torch.nn import functional as F
import torch.nn as nn
from omegaconf import OmegaConf
from pydantic import BaseModel
from enum import Enum

from dynamic_tokenization.models.gpt import GPT_models


class PatchingModeEnum(str, Enum):
    token = "token" # Each discrete token is a segment in itself
    static = "static" # Fixed patch size. The sequence is divided into seq_len / patch_size segments
    entropy = "entropy" # Dynamic entropy based patching
    entropy_row_boundary = "entropy_row_boundary" # Dynamic entropy based patching with a new patch starting at every row

class PatcherArgs(BaseModel):
    patching_mode: PatchingModeEnum 
    entropy_model_checkpoint_config: str | None = None # Required if patching_mode is entropy 
    entropy_model_checkpoint: str | None = None # Required if patching_mode is entropy 
    threshold: float | None = None # Required if patching_mode is entropy 
    max_patch_length: int | None = None
    patch_size: float | None = None # Required if patching_mode is fixed 
    monotonicity: bool = False
    log_time: bool = False
    block_size: int | None = None
    use_preprocessed_entropy: bool = False

    def build(self) -> "Patcher":
        return Patcher(self)


def load_entropy_model(config_path: str, ckpt_path: str) -> nn.Module:
    config = OmegaConf.load(config_path)
    if config.model.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = config.model.dropout_p
    latent_size = config.dataset.image_size // config.model.downsample_size
    model: nn.Module =  GPT_models[config.model.gpt_model](
        vocab_size=config.model.codebook_size,
        block_size=latent_size**2,
        num_classes=config.dataset.num_classes,
        cls_token_num=config.model.cls_token_num,
        model_type=config.model.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=config.model.drop_path_rate,
        token_dropout_p=config.model.token_dropout_p,
    )
    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def entropy(scores):
    """
    scores: [bs, seq_len, vocab]
    returns [bs, seq_len]

    Computes the entropy for each token in the batch.
    Note: uses natural log.
    """
    log_probs = F.log_softmax(scores, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim=-1)
    return entropy

def patch_start_mask_from_entropy_with_monotonicity(entropies, t):
    """
    entropies: [bs, seq_len] torch tensor of entropies
    t: threshold
    returns [bs, seq_len] mask where True indicates the start of a patch
    """
    bs, seq_len = entropies.shape

    if seq_len == 0:
        return entropies > t

    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    # Calculate differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]

    # Calculate conditions for all elements except the first one in each sequence
    condition = differences > t

    # Update the mask based on the condition
    mask[:, 1:] = condition

    return mask


def patch_start_mask_global_and_monotonicity(entropies, t, t_add=0):
    """
    entropies: [bs, seq_len] torch tensor of entropies
    t: threshold
    returns [bs, seq_len] mask where True indicates the start of a patch
    """
    bs, seq_len = entropies.shape

    if seq_len == 0:
        return entropies > t

    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    # Calculate differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]

    # Calculate conditions for all elements except the first one in each sequence
    condition = (differences > t_add) & (entropies[:, 1:] > t) & (~mask[:, :-1])

    # Update the mask based on the condition
    mask[:, 1:] = condition

    return mask


def patch_start_ids_from_patch_start_mask(patch_start_mask):
    bs, trunc_seq_len = patch_start_mask.shape
    max_patches = patch_start_mask.sum(dim=1).max()
    if max_patches == 0:
        patch_start_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
    else:
        patch_ids = (
            torch.arange(trunc_seq_len, device=patch_start_mask.device)
            .unsqueeze(0)
            .repeat(bs, 1)
        )
        extra_patch_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
        all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
        patch_start_mask_padded = torch.cat(
            (patch_start_mask, ~patch_start_mask), dim=1
        )
        patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(
            bs, trunc_seq_len
        )[:, :max_patches]
    return patch_start_ids


def check_non_zero_after_zero(tensor):
    zero_mask = tensor == 0
    shifted_mask = torch.cat(
        [
            torch.zeros(tensor.shape[0], 1, dtype=torch.bool, device=tensor.device),
            zero_mask[:, :-1],
        ],
        dim=1,
    )
    non_zero_after_zero = (tensor != 0) & shifted_mask
    return non_zero_after_zero.any()


def patch_lengths_from_start_ids(patch_start_ids, seq_len):
    """
    Calculate patch lengths from start ids.
    start ids: ex: [0, 1, 7, 7, 7, 7, 7], it has the start ids of the patches (here 0, 1), and then
        the rest are filled to the seq len.
    seq_len: ex: 7 length of the sequence

    returns the patch lengths:
    [1, 6] for the above example.
    """
    last_ids = torch.full_like(patch_start_ids[:, :1], seq_len - 1)
    patch_end_ids = torch.cat((patch_start_ids[:, 1:] - 1, last_ids), dim=1)
    patch_lengths = patch_end_ids - patch_start_ids + 1
    assert torch.all(patch_lengths >= 0), f"{patch_lengths}"
    assert not check_non_zero_after_zero(patch_lengths), f"{patch_lengths}"
    return patch_lengths


def find_entropy_patch_start_ids(
    entropies,
    patch_size=None,
    threshold=None,
    threshold_add=None,
    monotonicity=False,
    include_next_token=True,
    block_size=None,
    include_eoi_token=False,
):
    """
    Use entropies to find the start ids of each patch.
    Use patch_size or threshold to figure out the total number of patches to allocate.

    When threshold is not None the number of patches is not constant between
    different sequences, but patches can be identified incrementally rather than
    decided globally using the entire sequence.
    """
    bs, seq_len = entropies.shape[:2]

    first_ids = (
        torch.tensor([0, 1], dtype=torch.long, device=entropies.device)
        .unsqueeze(0)
        .repeat(bs, 1)
    )
    preds_truncation_len = first_ids.shape[
        1
    ]  # remove the first preds because they will be start of patches.
    entropies = entropies[:, 1:]
    if threshold is None:
        num_patches = seq_len // patch_size
        patch_start_ids = entropies.topk(num_patches - 2, dim=1).indices
        patch_start_ids = patch_start_ids.sort(dim=1).values
    else:
        # Assumes that there is at least one token going over the threshold
        if monotonicity:
            patch_start_mask = patch_start_mask_from_entropy_with_monotonicity(
                entropies, threshold
            )
        elif threshold_add is not None and threshold is not None:
            patch_start_mask = patch_start_mask_global_and_monotonicity(
                entropies, threshold, threshold_add
            )
        else:
            patch_start_mask = entropies > threshold
        if block_size:
            row_start_idxs = (torch.arange(patch_start_mask.shape[-1]) + preds_truncation_len) % block_size == 0
            patch_start_mask[:, row_start_idxs] = True
        if not include_next_token:
            patch_start_mask = patch_start_mask[:, :-1]
        if include_eoi_token:
            patch_start_mask = torch.cat([patch_start_mask, torch.ones((bs,1), dtype=torch.bool, device=patch_start_mask.device)], dim=-1)
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)

    patch_start_ids = torch.cat(
        (first_ids, patch_start_ids + preds_truncation_len), dim=1
    )
    return patch_start_ids


def rightpad(seq, pad_id, max_len):
    return seq + [pad_id] * (max_len - len(seq))


def split_large_numbers(lst, m):
    new_lst = []
    for i in lst:
        if i > m:
            while i > m:
                new_lst.append(m)
                i -= m
            new_lst.append(i)
        else:
            new_lst.append(i)
    assert sum(new_lst) == sum(lst), f"{sum(new_lst)} != {sum(lst)}"
    return new_lst

@torch.no_grad()
def patch_ids_from_lengths(patch_lengths, seq_len):
    bs, num_patches = patch_lengths.shape
    # Create a tensor of cumulative sums of the patch lengths
    cum_d = torch.cat(
        [
            torch.zeros(bs, 1, dtype=patch_lengths.dtype, device=patch_lengths.device),
            patch_lengths.cumsum(dim=-1),
        ],
        dim=-1,
    )
    patch_ids = (cum_d.unsqueeze(-1) <= torch.arange(seq_len, device=cum_d.device)).sum(
        dim=-2
    ) - 1
    assert patch_ids.shape[-1] == 0 or not (
        torch.max(patch_ids) > patch_lengths.shape[-1] or torch.min(patch_ids) < 0
    ), f"{torch.max(patch_ids)} > {patch_lengths.shape[-1]} or {torch.min(patch_ids)} < 0"
    return patch_ids


def create_patch_mask_from_ids(
    patch_ids, num_patches, window=None, patches_as_queries=False
):
    """
    Creates a tensor of shape [bs, seq_len, num_patches] where each element at position (i, j, k)
    is True if the patch id at position (i, j) is less than or equal to k.
    Args:
        patch_ids (torch.Tensor): Tensor of shape [bs, seq_len] containing patch ids.
        num_patches (int): Total number of patches.
        window (int): If not None, only considers patches within a window of size window.
        patches_as_queries (bool): If True, the patches are used as queries
    Returns:
        torch.Tensor: Tensor of shape [bs, q_len, kv_len] with the desired mask.
    """
    bs, seq_len = patch_ids.shape
    if not patches_as_queries:
        q_ids = patch_ids.unsqueeze(-1).expand(bs, seq_len, num_patches)
        kv_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, seq_len, num_patches)
        )
    else:
        kv_ids = patch_ids.unsqueeze(1).expand(bs, num_patches, seq_len)
        q_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bs, num_patches, seq_len)
        )
    if window is None:
        mask = q_ids == kv_ids
    else:
        mask = (kv_ids <= q_ids) & (q_ids < kv_ids + window)
    return mask


class Patcher(nn.Module):
    def __init__(self, patcher_args: PatcherArgs):
        super().__init__()
        self.patcher_args = patcher_args
        self.patching_mode = patcher_args.patching_mode
        self.threshold = patcher_args.threshold
        self.max_patch_length = patcher_args.max_patch_length
        self.patch_size = patcher_args.patch_size
        self.monotonicity = patcher_args.monotonicity
        self.log_time = patcher_args.log_time

        # load model for entropy patching
        if patcher_args.patching_mode == PatchingModeEnum.entropy or patcher_args.patching_mode == PatchingModeEnum.entropy_row_boundary:
            assert patcher_args.entropy_model_checkpoint_config, "entropy_model_checkpoint_config cannot be None"
            assert patcher_args.entropy_model_checkpoint, "entropy_model_checkpoint cannot be None"
            assert patcher_args.threshold, "threshold cannot be none"

            self.entropy_model = None
            if not patcher_args.use_preprocessed_entropy:
                self.entropy_model = load_entropy_model(patcher_args.entropy_model_checkpoint_config, patcher_args.entropy_model_checkpoint)
        
        if patcher_args.patching_mode == PatchingModeEnum.entropy_row_boundary:
            assert patcher_args.block_size, "block_size cannot be None"

        if self.log_time:
            self.log = defaultdict(float)

    def forward(
        self,
        tokens: torch.Tensor,
        cond: torch.Tensor | None = None,
        include_next_token: bool = False,
        preds: torch.Tensor | None = None,
        entropies: torch.Tensor | None = None,
        threshold: float | None = None,
        include_eoi_token: bool = False,
        attn_impl: str = "xformers"
    ) -> torch.Tensor:
        """
        tokens: 2D tensor of shape [batch_size, seq_len] that needs to be patched
        Returns patch lengths and optionally scores associated with the tokens (i.e. entropies, logprobs etc.)
        -> output tensor: [batch_size, max_num_patches]
            each tensor is processed independently and gets right padded with zeros.

        Patching with the following modes:
        1. patching_mode = "token": Patches of size 1
        2. patching_mode = "static": Patches of size `patch size` 
        3. patching_mode = "entropy":
            calculate entropy of each token, allocate patches so that the total
            number of patches is the same as static patching but choose to begin
            patches on tokens where the model is most uncertain (highest entropy).

            When threshold is provided, it uses the threshold to decide when to
            start a new patch.

        To correctly patch the last token, it may be necessary to include the next token in the patch
        lengths calculations. This is controlled by the include_next_token argument.
        """
        bs, seq_len = tokens.shape
        seq_len_next_tok = seq_len + 1 if include_next_token else seq_len # NOTE: We need to add patch for the class condition token 
        seq_len_next_tok = seq_len_next_tok + 1 if include_eoi_token else seq_len_next_tok
        scores = None

        # STATIC
        if self.log_time:
            s = time.time()
        if self.patching_mode == PatchingModeEnum.token:
            patch_lengths = torch.ones((tokens.shape[0], seq_len_next_tok), dtype=tokens.dtype, device=tokens.device)
        elif self.patching_mode == PatchingModeEnum.static:
            patch_lengths = torch.zeros(
                (bs, math.ceil(seq_len_next_tok / self.patch_size)),
                # dtype=tokens.dtype,
                device=tokens.device,
            ).fill_(self.patch_size)
            if seq_len_next_tok % self.patch_size != 0:
                patch_lengths[:, -1] = seq_len_next_tok % self.patch_size
        elif self.patching_mode == PatchingModeEnum.entropy or self.patching_mode == PatchingModeEnum.entropy_row_boundary:
            if entropies is not None:
                scores = entropies.to(dtype=torch.float32)
            elif preds is not None:
                scores = entropy(preds)
            else:
                # calculate entropy in this scenario
                scores = entropy(self.entropy_model(idx=tokens, cond_idx=cond, attn_impl=attn_impl)[0])
            if self.log_time:
                self.log["calculate_entropies"] += time.time() - s
                s = time.time()
            if self.patching_mode == PatchingModeEnum.entropy_row_boundary:
                patch_start_ids = find_entropy_patch_start_ids(
                    scores,
                    self.patch_size,
                    include_next_token=include_next_token,
                    threshold=threshold if threshold is not None else self.threshold,
                    monotonicity=self.monotonicity,
                    block_size=self.patcher_args.block_size,
                    include_eoi_token=include_eoi_token
                )
            else:
                patch_start_ids = find_entropy_patch_start_ids(
                scores,
                self.patch_size,
                include_next_token=include_next_token,
                threshold=threshold if threshold is not None else self.threshold,
                monotonicity=self.monotonicity,
                include_eoi_token=include_eoi_token
            )
            if self.log_time:
                self.log["find_entropy_patch_start_ids"] += time.time() - s
                s = time.time()
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_len_next_tok
            )
            if self.log_time:
                self.log["patch_lengths_from_start_ids"] += time.time() - s
                s = time.time()
        else:
            raise NotImplementedError(f"self.patching_mode {self.patching_mode}")

        # Apply any processing to patch lengths
        if self.max_patch_length is not None:
            # TODO: avoid going back to a list here.
            patch_lengths = [
                split_large_numbers(pl, self.max_patch_length)
                for pl in patch_lengths.tolist()
            ]
            max_len = max([len(pl) for pl in patch_lengths])
            patch_lengths = [rightpad(pl, 0, max_len=max_len) for pl in patch_lengths]
            patch_lengths = torch.tensor(
                patch_lengths, dtype=tokens.dtype, device=tokens.device
            )
        assert not check_non_zero_after_zero(patch_lengths)
        
        # Find the last non-zero column index using argmax on a reversed version of the tensor
        last_non_zero_col_reversed = (
            (patch_lengths != 0).flip(dims=[1]).int().argmax(dim=1).min()
        )
        
        # Slice the tensor up to the last non-zero column
        patch_lengths = patch_lengths[
            :, :patch_lengths.shape[1] - last_non_zero_col_reversed
        ]
        if self.patching_mode != PatchingModeEnum.static:
            assert (
                torch.sum(patch_lengths)
                == tokens.numel() + include_next_token * tokens.shape[0] + include_eoi_token * tokens.shape[0]
            ), f"{torch.sum(patch_lengths)} != {tokens.numel() + include_next_token * tokens.shape[0] + include_eoi_token * tokens.shape[0]}"
        if self.log_time:
            self.log["postprocessing_patch_lengths"] += time.time() - s
            self.log["tokens"] += patch_lengths.sum().item()
        
        return patch_lengths, scores

    def move_buffers_to_device(self, device) -> None:
        if self.patching_mode != PatchingModeEnum.static and self.entropy_model:
            self.entropy_model.move_buffers_to_device(device)

