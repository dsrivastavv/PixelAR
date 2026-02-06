from dataclasses import dataclass
import torch
import logging

import torch.nn as nn
from typing import List, Optional, Any
from enum import Enum

from torch.utils.checkpoint import checkpoint

from PixelAR.models.gpt import apply_rotary_emb, get_start_pos_from_seqlens, map_pos_to_freq, precompute_freqs_cis_2d, TransformerBlock, RMSNorm
from PixelAR.models.patcher import create_patch_mask_from_ids
from xformers.ops import fmha, LowerTriangularMask
from xformers.ops import fmha, LowerTriangularMask
from xformers.ops.fmha import MemoryEfficientAttentionCutlassOp
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

@torch.compiler.disable
def _call_xformers_cross_attention(
    q_seqlens: torch.Tensor,
    k_seqlens: torch.Tensor,
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    attn_dropout_p: float = 0.0,
):
    x_seqlens_cpu = q_seqlens.detach().to(dtype=torch.int32)
    kv_seqlens_cpu = k_seqlens.detach().to(dtype=torch.int32)
    attn_bias = BlockDiagonalMask.from_seqlens(
        q_seqlen=x_seqlens_cpu.tolist(),
        kv_seqlen=kv_seqlens_cpu.tolist(),
        device=xq.device,
    )
    return fmha.memory_efficient_attention(
        xq,
        xk,
        xv,
        attn_bias=attn_bias,
        p=attn_dropout_p,
        op=MemoryEfficientAttentionCutlassOp,
    )
    
logger = logging.getLogger()


class InitStdFactor(str, Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096



def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class SingleKeyCrossAttention(nn.Module):
    """
    Optimized Cross attention to copy the interface when a query attends to a single key value.
    """

    def __init__(
        self,
        dim: int,
        norm_eps: float,
        packed_inputs: bool = False
    ):
        super().__init__()

        self.dim = dim
        self.packed_inputs = packed_inputs

        self.cross_attn_norm_kv = RMSNorm(dim, eps=norm_eps)
        self.wo = nn.Linear(
            dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor, 
        kv: torch.Tensor,
        *args,
        **kwargs
    ):
        if self.packed_inputs:
            output = self._forward_packed(x, kv, *args, **kwargs)
            return output
        else:
            return self._forward_simple(x, kv, *args, **kwargs)

    def _forward_simple(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # B S D
        bsz, _, _ = x.shape

        # check first non-zero in each row of the mask
        kv = self.cross_attn_norm_kv(kv)
        out = self.wo(kv)

        # Note: mimic cross-attention for each query mapped to a single key value
        mask = mask.squeeze(1)
        patch_id = torch.argmax(mask, dim=-1)  # [bsz, seq_len]
        patch_embeds_mapped = out[torch.arange(bsz).unsqueeze(-1), patch_id]  # [bsz, seq_len, dim]       
        return x + patch_embeds_mapped
    
    def _forward_packed(
        self,
        x: torch.Tensor,          # [Tq, D], (bf16/fp16 under AMP)
        kv: torch.Tensor,         # [Tk, D]
        patch_lens: torch.Tensor, # [num_patches]
        patch_seqlens: torch.Tensor, # [B]
        token_seqlens: torch.Tensor  # [B]
        
    ) -> torch.Tensor:        
        # pre-norms are AMP/compile-safe (keep eps as python float in the module)
        out = self.cross_attn_norm_kv(kv)

        # projections (AMP-friendly)
        out = self.wo(out)

        # Note: mimic cross-attention for each query mapped to a single key value
        assert torch.all(token_seqlens == token_seqlens[0]), "All token_seqlens must be the same for DummyCrossAttention"
        token_batch_start_pos = get_start_pos_from_seqlens(patch_seqlens).view(-1, 1).repeat(1, token_seqlens[0]).view(-1)
        token_batch_rel_pos = torch.repeat_interleave(patch_lens).reshape(-1, token_seqlens[0])
        token_batch_rel_pos = (token_batch_rel_pos - token_batch_rel_pos[:, :1]).view(-1)
        token_patch_id = token_batch_start_pos + token_batch_rel_pos
        
        # output projection; match activation dtype for the residual add
        patch_embeds_mapped = out[token_patch_id]

        return x + patch_embeds_mapped

    def init_weights(self, base_std: float, factor: float = 1.0):
        std = base_std or (self.dim ** (-0.5)) / factor
        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )


class CrossAttention(nn.Module):
    """
    CrossAttention block to attend to the encoder states from the decoder.
    Rope is not supported.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        packed_inputs: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads
        self.packed_inputs = packed_inputs
        self.use_rope = use_rope

        self.cross_attn_norm_q = RMSNorm(dim, eps=norm_eps)
        self.cross_attn_norm_kv = RMSNorm(dim, eps=norm_eps)

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

        # log state for CrossAttention
        logger = logging.getLogger(self.__class__.__name__)
        logger.debug(f"Initialized CrossAttention: dim={dim}, head_dim={head_dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, packed_inputs={packed_inputs}, use_rope={use_rope}")

    def forward(
        self,
        x: torch.Tensor, 
        kv: torch.Tensor,
        *args,
        **kwargs
    ):
        if self.packed_inputs:
            output = self._forward_packed(x, kv, *args, **kwargs)
            return output
        else:
            return self._forward_simple(x, kv, *args, **kwargs)

    def _forward_simple(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[Any] = None,
        attn_impl: str = "xformers",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, _ = x.shape
        _, slen_kv, _ = kv.shape
        x_norm = self.cross_attn_norm_q(x)
        kv = self.cross_attn_norm_kv(kv)

        xq = self.wq(x_norm)
        xk = self.wk(kv)
        xv = self.wv(kv)

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)
        
        if self.use_rope:
            assert freqs_cis is not None, "freqs_cis must be provided when use_rope is True"
            xq = apply_rotary_emb(xq, freqs_cis[:seq_len])
            xk = apply_rotary_emb(xk, freqs_cis[:slen_kv])

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        # attention
        if attn_impl == "sdpa":
            # sdpa
            # assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            inf_rows = torch.all(torch.isinf(mask), dim=-1, keepdim=True)
            mask = torch.where(inf_rows, -1e9, mask)
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        elif attn_impl == "xformers":
            # This uses B S H D instead of B H S D of pytorch
            attn_bias = LowerTriangularMask() if mask is None else CrossAttention.align_tensor_stride(mask.repeat(1, self.n_heads, 1, 1,).to(xq), fill_value=float("-inf"))
            output = self.apply_xformer_attn(
                xq, xk, xv, 
                attn_bias=attn_bias,
            )
            output = output.contiguous().view(bsz, seq_len, self.dim)
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )
        
        # self.wq.ca_inp_q = xq
        # self.wq.ca_inp_k = xk
        # self.wq.ca_inp_v = xv
        # output = flex_attention_comp(xq, xk, xv, block_mask=mask)
        # def score_mod(score, b, h, q_idx, kv_idx):
        #     return torch.where(mask[b,0,q_idx,kv_idx]==0, score, -float("inf"))

        # self.wq.ca_output, lse_score = flex_attention_comp(self.wq.ca_inp_q, self.wq.ca_inp_k, self.wq.ca_inp_v, return_lse=True, score_mod=score_mod)
        # self.wq.ca_output = torch.nn.functional.scaled_dot_product_attention(self.wq.ca_inp_q, self.wq.ca_inp_k, self.wq.ca_inp_v, attn_mask=mask)
        # self.wq.score = score
        # self.wq.block_mask = mask
        # self.wq.ca_inp_q.retain_grad()
        # self.wq.ca_inp_k.retain_grad()
        # self.wq.ca_inp_v.retain_grad()
        # self.wq.ca_output.retain_grad()
        # output = self.wq.ca_output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        output = self.wo(output.reshape(output_shape))

        return x + output
    
    
    def _forward_packed(
        self,
        x: torch.Tensor,          # [Tq, D], (bf16/fp16 under AMP)
        kv: torch.Tensor,         # [Tk, D]
        x_seqlens: torch.Tensor,  # [num_blocks], int32
        kv_seqlens: torch.Tensor,  # [num_blocks], int32
        x_freqs_cis: Optional[torch.Tensor] = None, 
        kv_freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # same number of blocks
        assert x_seqlens.shape[0] == kv_seqlens.shape[0], \
            f"Should have same number of blocks {x_seqlens.shape} vs {kv_seqlens.shape}"

        # pre-norms are AMP/compile-safe (keep eps as python float in the module)
        x_norm = self.cross_attn_norm_q(x)
        kv_norm = self.cross_attn_norm_kv(kv)

        # projections (AMP-friendly)
        xq = self.wq(x_norm)
        xk = self.wk(kv_norm)
        xv = self.wv(kv_norm)

        # split heads – derive sizes from tensors (no threaded ints)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # 1, T, H, Hd
            return t.view(1, t.shape[0], self.n_heads, self.head_dim).contiguous()
        xq = split_heads(xq)
        xk = split_heads(xk)
        xv = split_heads(xv)
        
        # apply rotary embeddings if provided
        if self.use_rope:
            assert x_freqs_cis is not None and kv_freqs_cis is not None, "Both x_freqs_cis and kv_freqs_cis must be provided"
            xq = apply_rotary_emb(xq, x_freqs_cis)
            xk = apply_rotary_emb(xk, kv_freqs_cis)

        # repeat kv heads (ensure your repeat_kv uses reshape/expand, no .item())
        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        # cross-attention (prefer SDPA backend if you can; otherwise keep this)
        out = _call_xformers_cross_attention(
            q_seqlens=x_seqlens,
            k_seqlens=kv_seqlens,
            xq=xq, xk=xk, xv=xv,
        )  # [1, Tq, H, Hd], dtype may be fp32 depending on kernel

        # merge heads back – no captured SymInts
        out = out.contiguous().view(x.shape[0], x.shape[1])

        # output projection; match activation dtype for the residual add
        out = self.wo(out).to(x.dtype)

        return x + out

    def init_weights(self, base_std: float, factor: float = 1.0):
        std = base_std or (self.dim ** (-0.5)) / factor

        nn.init.trunc_normal_(
            self.wq.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wk.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wv.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        # self.cross_attn_norm_q.reset_parameters()
        # self.cross_attn_norm_kv.reset_parameters()

    @torch.compiler.disable
    def apply_xformer_attn(
        self,
        xq: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor,
        attn_bias: torch.Tensor     
    ):
        output = fmha.memory_efficient_attention(
            xq, keys, values, 
            attn_bias=attn_bias,
        )
        return output

    @torch.no_grad()
    @staticmethod
    def align_tensor_stride(
        tensor: torch.Tensor, 
        multiple: int = 8, 
        fill_value: Any = 0
    ) -> torch.Tensor:
        """
        Pads the last two dimensions of a tensor to be a multiple of `multiple`.

        This is done by creating a new, larger tensor and copying the original
        data into it, which ensures the new tensor's memory layout (stride)
        is well-aligned for high-performance kernels.

        Args:
            tensor: The input tensor.
            multiple: The multiple to pad the dimensions to. Defaults to 8.
            fill_value: The value to use for padding. Defaults to 0.

        Returns:
            A new tensor with padded dimensions and aligned strides.
        """
        # 1. Get original shape and calculate the new, padded shape
        original_shape = tensor.shape
        h, w = original_shape[-2:]
        
        new_h = h + (multiple - h % multiple) % multiple + multiple
        new_w = w + (multiple - w % multiple) % multiple + multiple
        
        padded_shape = (*original_shape[:-2], new_h, new_w)

        # 2. Create a new continuous tensor with the target padded shape
        padded_tensor = torch.full(
            padded_shape, fill_value, dtype=tensor.dtype, device=tensor.device
        )

        # 3. Copy the original data into the top-left slice of the new tensor
        padded_tensor[..., :h, :w] = tensor
        return padded_tensor[..., :h, :w]

@torch.no_grad()
def cross_attn_mask(
    patch_ids,
    num_patches,
    N,
    patches_as_queries=False,
    cross_attn_k=1,
    window=None,
    block_mask=True,
):
    bs = patch_ids.shape[0]
    with torch.no_grad():
        # Create the patch mask
        cross_mask = create_patch_mask_from_ids(
            patch_ids,
            num_patches,
            window=window,
            patches_as_queries=patches_as_queries,
        ).repeat_interleave(cross_attn_k, dim=1 if patches_as_queries else -1)
        q_len = num_patches * cross_attn_k if patches_as_queries else N
        kv_len = N if patches_as_queries else num_patches * cross_attn_k
        assert cross_mask.shape == (
            bs,
            q_len,
            kv_len,
        ), f"{cross_mask.shape} != {(bs, q_len, kv_len)}"
        if block_mask:

            def patch_mask(b, h, q_idx, kv_idx):
                return cross_mask[b, q_idx, kv_idx]

            block_mask = create_block_mask(
                patch_mask,
                B=bs,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len
            )
            return block_mask
        else:
            return torch.where(
                cross_mask, torch.tensor(0.0), torch.tensor(float("-inf"))
            ).unsqueeze(
                1
            )  # [bs, 1, q_len, kv_len]

def patch_reduce(h, max_num_patches, reduction, patch_ids):
    """
    Reduce variable length patches to single embedding per patch
    Note: this works with variable number of patches for different sequences in the batch
    It handles variable length patches by assuming that patch_lengths will be 0 for any
    extra patches on the *right*. Since there can be a variable number of patches
    this function also return the number of patches for each sequence in the batch.
    Any embeddings on the right that are not allocated to a patch
    (i.e. if the sum(patch_lengths[i]) < seq_len for any i)
    will be sent to a dummy patch, which is trimmed before returning.
    """
    bs, seq_len, emb_dim = h.shape

    patch_ids = patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1])

    reduced_embs = torch.zeros(
        (bs, max_num_patches, emb_dim), dtype=h.dtype, device=h.device
    )
    reduced_embs = reduced_embs.scatter_reduce(
        src=h,
        dim=1,
        index=patch_ids,
        reduce=reduction,
        include_self=False,
    )
    reduced_embs = reduced_embs[:, :max_num_patches, :]

    return reduced_embs

def concat_downsample(h, patch_lengths, patch_size):
    # The assumption in this function is that seq_len = patch_size * num_patches.
    bs, seq_len, emb_dim = h.shape
    patch_end_ids = torch.cumsum(patch_lengths, dim=1)
    patch_ids = patch_end_ids.unsqueeze(-1) - torch.arange(patch_size, 0, -1).to(
        patch_end_ids.device
    )
    # Is clamp ok here?
    patch_ids = patch_ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, h.shape[-1])
    patch_ids = patch_ids.view(bs, -1, emb_dim)
    # after gather h.shape = [batch_size, seq_len, dim]
    h = torch.gather(h, 1, patch_ids)
    h = h.reshape(bs, patch_lengths.shape[1], patch_size * h.size(-1))
    return h


def pooling_downsample(h, max_num_patches, pooling_mode, patch_ids):
    cat = []
    if "avg" in pooling_mode or "mean" in pooling_mode:
        cat.append(patch_reduce(h, max_num_patches, "mean", patch_ids))
    if "min" in pooling_mode:
        cat.append(patch_reduce(h, max_num_patches, "amin", patch_ids))
    if "max" in pooling_mode:
        cat.append(patch_reduce(h, max_num_patches, "amax", patch_ids))
    assert len(cat) > 0
    h = torch.cat(cat, dim=-1)
    return h


def downsample(
    h,
    num_patches,
    patch_lengths=None,
    patch_ids=None,
    downsampling_by_pooling=None,
    patch_size=4,
):
    """
    Downsampling:
        a. concatenating embeddings in the patch
            Note: with dynamic patching, patch the last patch_size tokens.
        b. pooling embeddings in the patch
    """
    # input: h.shape = [batch_size, seq_len, dim]
    # input: pool h.shape = [batch_size, seq_len / patch_size, dim]
    # if we don't use the cros_attn, we pool so that we convert bytes rep to patch rep
    if downsampling_by_pooling is not None and len(downsampling_by_pooling) > 0:
        # By pooling
        max_num_patches = num_patches
        assert patch_ids is not None
        h = pooling_downsample(h, max_num_patches, downsampling_by_pooling, patch_ids)
    else:
        # TODO: remove this condition
        # By concatenating (fixed lengths patching)
        assert patch_lengths is not None
        h = concat_downsample(h, patch_lengths, patch_size)
    return h

@dataclass
class EncoderModelArgs:
    dim: int = 4096
    n_layer: int = 1
    n_head: int = 16
    n_cross_attn_head: int = 16
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: int = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    patch_embedding_projection_layer: bool = False
    gradient_checkpointing: bool = False
    use_ca_rope: bool = False

class LocalEncoder(nn.Module):
    def __init__(self, config: EncoderModelArgs, packed_inputs: bool = False):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.packed_inputs = packed_inputs

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        self.cross_attn_layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id], packed_inputs=self.packed_inputs))
            # cross attention
            self.cross_attn_layers.append(
                CrossAttention(
                    dim=config.dim,
                    head_dim=config.dim // config.n_cross_attn_head,
                    n_heads=config.n_cross_attn_head,
                    n_kv_heads=config.n_cross_attn_head,
                    norm_eps=config.norm_eps,
                    packed_inputs=packed_inputs,
                    use_rope=config.use_ca_rope,
                )
            )            

        # NOTE: needed when encoder/decoder dimension is different from global model
        self.patch_embedding_projection = None
        if config.patch_embedding_projection_layer:
            self.patch_embedding_projection = nn.Linear(
                in_features=config.dim,
                out_features=config.dim,
                bias=False,
            )

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size

        # NOTE: Local encoder does not use condition, hence no need for condition freq embedding
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, 0)

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        for depth, layer in enumerate(self.cross_attn_layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.config.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[InitStdFactor.DISABLED] # NOTE: Force set to 1.0 following BLT
            layer.init_weights(None, factor)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        *args,
        **kwargs
    ):
        if self.packed_inputs:
            return self._forward_packed(token_embeddings, *args, **kwargs)
        else:
            return self._forward_simple(token_embeddings, *args, **kwargs)
        
    def _forward_simple(
        self,
        token_embeddings: torch.Tensor,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        attn_impl: str = "xformers",
    ):
        if self.training or input_pos is None:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        
        h = token_embeddings
        patch_embeds = None
        for i, layer in enumerate(self.layers):
            if self.config.gradient_checkpointing:
                h = checkpoint(layer, h, freqs_cis, input_pos, mask, attn_impl=attn_impl, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, input_pos, mask, attn_impl=attn_impl)
            patch_embeds = self.apply_cross_attention_simple(
                h, patch_embeds, i, num_patches, patch_ids, cross_mask, attn_impl=attn_impl
            )

        return h, patch_embeds
    
    
    def _forward_packed(
        self,
        token_embeddings: torch.Tensor, # [T, D]
        patch_lengths: torch.Tensor, # [num_blocks]
        token_seqlens: torch.Tensor, # [B]
    ):
        h = token_embeddings

        # freq embeddings
        _freq_cis = map_pos_to_freq(token_seqlens, self.freqs_cis)
        
        # transformer blocks
        patch_embeds = None
        for i, layer in enumerate(self.layers):
            # self attention
            if self.config.gradient_checkpointing:
                h = checkpoint(layer, h, _freq_cis, seqlens=token_seqlens, use_reentrant=False)
            else:
                h = layer(h, _freq_cis, seqlens=token_seqlens)
                
            # cross attention
            patch_embeds = self.apply_cross_attention_padded(
                h, patch_embeds, i, patch_lengths
            )
        return h, patch_embeds
    
    def apply_cross_attention_simple(
        self, h, patch_embeds, layer_idx, num_patches, patch_ids, cross_mask, attn_impl
    ):
        # apply pooling and project
        if patch_embeds is None:
            patch_embeds = downsample(
                h,
                num_patches,
                patch_ids=patch_ids,
                downsampling_by_pooling='max',
                patch_size=None, # type: ignore
            )
            if self.patch_embedding_projection is not None:
                patch_embeds = self.patch_embedding_projection(patch_embeds)

        patch_embeds_cross = self.cross_attn_layers[layer_idx](
            x=patch_embeds,
            kv=h,
            freqs_cis=self.freqs_cis,
            mask=cross_mask,
            attn_impl=attn_impl
        )
        return patch_embeds_cross
    
    
    def apply_cross_attention_padded(
        self, h: torch.Tensor, patch_embeds: torch.Tensor | None, layer_idx: int, patch_lengths: torch.Tensor
    ):
        # apply pooling and project
        if patch_embeds is None:
            patch_ids = torch.repeat_interleave(patch_lengths)
            T, D = patch_lengths.shape[0], h.shape[1]
            patch_embeds = downsample(
                h.unsqueeze(0),
                num_patches=patch_lengths.shape[0],
                patch_ids=patch_ids.unsqueeze(0),
                downsampling_by_pooling='max',
                patch_size=None, # type: ignore
            ).view(T, D)
            if self.patch_embedding_projection is not None:
                patch_embeds = self.patch_embedding_projection(patch_embeds)

        # x seqlens
        x_seqlens = torch.tensor([1], dtype=torch.int32, device=h.device).repeat(patch_embeds.shape[0])

        # get freq embeddings for patches
        _x_freqs_cis = map_pos_to_freq(x_seqlens, self.freqs_cis)
        _kv_freqs_cis = map_pos_to_freq(patch_lengths, self.freqs_cis)

        # cross attention
        patch_embeds_cross = self.cross_attn_layers[layer_idx](
            x=patch_embeds,
            kv=h,
            x_seqlens=x_seqlens,
            kv_seqlens=patch_lengths,
            x_freqs_cis=_x_freqs_cis,
            kv_freqs_cis=_kv_freqs_cis,
        )
        return patch_embeds_cross
    
    def move_buffers_to_device(self, device) -> None:
        self.freqs_cis = self.freqs_cis.to(device)


@dataclass
class DecoderModelArgs:
    dim: int = 4096
    n_layer: int = 1
    n_head: int = 16
    n_cross_attn_head: int = 16
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: int = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    patch_embedding_projection_layer: bool = False
    gradient_checkpointing: bool = False

class LocalDecoder(nn.Module):
    def __init__(self, config: DecoderModelArgs, packed_inputs: bool = False):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.packed_inputs = packed_inputs

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        self.cross_attn_layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            # self attention
            self.layers.append(TransformerBlock(config, dpr[layer_id], packed_inputs=self.packed_inputs))
            # cross attention
            # if self.packed_inputs:
            self.cross_attn_layers.append(
                SingleKeyCrossAttention(
                    dim=config.dim,
                    norm_eps=config.norm_eps,
                    packed_inputs=self.packed_inputs
                )
            )
            # else:
            #     self.cross_attn_layers.append(
            #         CrossAttention(
            #             dim=config.dim,
            #             head_dim=config.dim // config.n_cross_attn_head,
            #             n_heads=config.n_cross_attn_head,
            #             n_kv_heads=config.n_cross_attn_head,
            #             norm_eps=config.norm_eps,
            #             packed_inputs=self.packed_inputs
            #         )
            #     )

        # NOTE: needed when encoder/decoder dimension is different from global model
        self.patch_embedding_projection = None
        if config.patch_embedding_projection_layer:
            self.patch_embedding_projection = nn.Linear(
                in_features=config.dim,
                out_features=config.dim,
                bias=False,
            )

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, config.cls_token_num)

         # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        if self.cross_attn_layers is not None:
            for depth, layer in enumerate(self.cross_attn_layers):
                if isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                    if layer.bias is not None:
                        layer.bias.data.zero_()
                else:
                    factor = {
                        InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                        InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                        InitStdFactor.DIM_RATIO: self.config.dim / 4096,
                        InitStdFactor.DISABLED: 1.0,
                    }[InitStdFactor.DISABLED] # NOTE: Force set to 1.0 following BLT
                    layer.init_weights(None, factor)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


    def forward(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        *args,
        **kwargs
    ):
        if self.packed_inputs:
            return self._forward_packed(embeds, patch_embeds, *args, **kwargs)
        else:
            return self._forward_simple(embeds, patch_embeds, *args, **kwargs)

    def _forward_simple(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        attn_impl: str = "xformers",
    ):
        # setup hidden embeddings
        h = embeds
        if self.training or input_pos is None:
            freqs_cis = self.freqs_cis[:h.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        
        for i, layer in enumerate(self.layers):
            # Use cross attention to extract info from patch_embeds into h
            h = self.cross_attn_layers[i](
                x=h,
                kv=patch_embeds,
                mask=cross_mask,
            )

            # self attention
            if self.config.gradient_checkpointing:
                h = checkpoint(layer, h, freqs_cis=freqs_cis, start_pos=input_pos, mask=mask, use_reentrant=False)
            else:
                h = layer(h, freqs_cis=freqs_cis, start_pos=input_pos, mask=mask)
    
        return h
    
    def _forward_packed(
        self,
        embeds: torch.Tensor, # [T, D]
        patch_embeds: torch.Tensor, # [T_patch, D]
        patch_lens: torch.Tensor, # [num_patches]
        patch_seqlens: torch.Tensor, # [B]
        token_seqlens: torch.Tensor, # [B]
    ):
        h = embeds
        
        # freq embeddings
        _freq_cis = map_pos_to_freq(token_seqlens, self.freqs_cis)
        
        # map token to correponding patch. size is the same as embeds 
         
        for i, layer in enumerate(self.layers):
            # cross attention
            h = self.cross_attn_layers[i](h, patch_embeds, patch_lens, patch_seqlens, token_seqlens)
            
            # self attention
            if self.config.gradient_checkpointing:
                h = checkpoint(layer, h, _freq_cis, seqlens=token_seqlens, use_reentrant=False)
            else:
                h = layer(h, _freq_cis, seqlens=token_seqlens)
    
        return h
    
    def move_buffers_to_device(self, device) -> None:
        self.freqs_cis = self.freqs_cis.to(device)
