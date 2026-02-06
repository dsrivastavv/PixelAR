# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List


import torch
import torch.nn as nn
from torch.nn import functional as F
from dynamic_tokenization.models.drop_path import DropPath
from xformers.ops import fmha, LowerTriangularMask
from xformers.ops.fmha import MemoryEfficientAttentionCutlassOp
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

@torch.compiler.disable
def _call_xformers_attention(
    seqlens: torch.Tensor,
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    attn_dropout_p: float = 0.0,
):
    seqlens_cpu = seqlens.detach().to(dtype=torch.int32)
    attn_bias = BlockDiagonalCausalMask.from_seqlens(
        seqlens_cpu.tolist(),
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

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


@torch.no_grad()
def map_pos_to_freq(seqlens: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # seqlens: [num_blocks] (int64), freqs_cis: [max_T, ...] (any dtype incl. complex/bf16)
    seqlens = seqlens.to(torch.long)
    total = seqlens.sum()

    start_offsets = torch.cumsum(seqlens, dim=0) - seqlens  # [num_blocks], int64
    per_token_offsets = torch.repeat_interleave(start_offsets, seqlens)  # [total], int64
    global_index = torch.arange(total, device=seqlens.device)  # [total], int64
    relative_index = global_index - per_token_offsets  # [total], int64
    return torch.index_select(freqs_cis, dim=0, index=relative_index)



def scatter_at_pos(input: torch.Tensor, data: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    Insert `data` rows into a new tensor at output indices `pos`, preserving the
    original order of `input` for the remaining slots.

    Args:
        input: [T1, D]  (base sequence, any floating dtype incl. bf16/fp16/fp32)
        data:  [T2, D]  (rows to insert; will be cast to input.dtype)
        pos:   [T2]     (indices in range [0, T1 + T2), unique; where to place `data`)

    Returns:
        output: [T1 + T2, D]
    """
    T1, D = input.shape
    T2 = data.shape[0]
    device = input.device
    dtype = input.dtype

    pos = pos.to(device=device, dtype=torch.long).flatten()
    
    # light runtime guards (kept tensor-y to be compile-friendly)
    assert pos.numel() == T2, "Position tensor must align with data length"
    
    # (Optional) uniqueness/range checks could be added if needed.

    T = T1 + T2

    # mask for where data goes
    data_mask = torch.zeros(T, dtype=torch.bool, device=device)
    data_mask[pos] = True
    input_mask = ~data_mask  # where input should fill, preserving order

    # prefix ranks: for each output position, what's the 0-based index we should take from input/data?
    in_rank  = torch.cumsum(input_mask.to(torch.int64), dim=0) - 1   # [-1 .. T1-1]
    dt_rank  = torch.cumsum(data_mask.to(torch.int64),  dim=0) - 1   # [-1 .. T2-1]

    # clamp negatives so gathers are always valid (values at those places will be discarded by where)
    in_rank  = torch.clamp(in_rank, min=0)
    dt_rank  = torch.clamp(dt_rank, min=0)

    # gather rows (shape: [T, D])
    picked_input = input[in_rank]                      # bf16-safe gather
    picked_data  = data[dt_rank].to(dtype)             # cast once to match AMP dtype

    # select per position without in-place writes / advanced index-put
    output = torch.where(input_mask.unsqueeze(1), picked_input, picked_data)
    return output

@torch.no_grad()
def get_start_pos_from_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor of sequence lengths, return the start positions for each sequence.

    Args:
        seqlens: [B] (int64) tensor of sequence lengths

    Returns:
        start_pos: [B] (int64) tensor of start positions
    """
    assert seqlens.dim() == 1, "seqlens must be a 1D tensor"
    start_pos = torch.cumsum(seqlens, dim=-1) - seqlens
    return start_pos

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    packed_inputs: bool = False


#################################################################################
#                      Embedding Layers for no Class Labels                        #
#################################################################################
class UcondEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, labels, train, force_drop_ids=None):
        embeddings = torch.empty((labels.shape[0], 0, self.hidden_size), device=labels.device)
        return embeddings


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        hidden = F.silu(self.w1(x)) * self.w3(x)

        w2_dtype = self.w2.weight.dtype
        if hidden.dtype != w2_dtype:
            hidden = hidden.to(dtype=w2_dtype)

        out = self.w2(hidden)

        if out.dtype != x.dtype:
            out = out.to(dtype=x.dtype)

        return self.ffn_dropout(out)


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val.float()
        v_out[:, :, input_pos] = v_val.float()

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, packed_inputs: bool = False):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim
        self.packed_inputs = packed_inputs

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x, freqs_cis, *args, **kwargs
    ):
        if self.packed_inputs:
            return self._forward_packed(x, freqs_cis, *args, **kwargs)
        else:
            return self._forward_simple(x, freqs_cis, *args, **kwargs)

    def _forward_simple(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
        attn_impl: str = "xformers",
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        if attn_impl == "sdpa":
            output = F.scaled_dot_product_attention(
                xq, keys, values, 
                attn_mask=mask, 
                is_causal=True if mask is None else False, # is_causal=False is for KV cache
                dropout_p=self.attn_dropout_p if self.training else 0)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        elif attn_impl == "xformers":
            # This uses B S H D instead of B H S D of pytorch
            attn_bias = LowerTriangularMask() if mask is None else mask
            xq, keys, values = map(lambda x: x.transpose(1, 2), (xq, keys, values))
            output = fmha.memory_efficient_attention(
                xq, keys, values, 
                attn_bias=attn_bias,
                p=self.attn_dropout_p if self.training else 0
            )
            output = output.contiguous().view(bsz, seqlen, self.dim)
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        output = self.resid_dropout(self.wo(output))
        return output

    def _forward_packed(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, seqlens: torch.Tensor
    ):
        T, dim = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)  # each [T, D]

        # reshape to [T, H, Hd] for xformers
        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(1, T, self.n_head, self.head_dim).contiguous()
        xq, xk, xv = split_heads(xq), split_heads(xk), split_heads(xv)
        
        # apply rotatory embeddings
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        # attention
        output = _call_xformers_attention(
            seqlens,
            xq=xq,
            xk=xk,
            xv=xv,
            attn_dropout_p=self.attn_dropout_p if self.training else 0.0,
        )  # [1, T, H, Hd]

        # reshape and ffn
        output = output.contiguous().view(T, dim)
        output = self.resid_dropout(self.wo(output))

        return output

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float, packed_inputs: bool = False):
        super().__init__()
        self.attention = Attention(config, packed_inputs=packed_inputs)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.packed_inputs = packed_inputs

    def forward(
        self, x, freqs_cis, *args, **kwargs
    ):
        if self.packed_inputs:
            return self._forward_packed(x, freqs_cis, *args, **kwargs)
        else:
            return self._forward_simple(x, freqs_cis, *args, **kwargs)

    def _forward_packed(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, seqlens: torch.Tensor
    ):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, seqlens))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out

    def _forward_simple(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None, attn_impl: str = "xformers"):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask, attn_impl=attn_impl))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        elif self.model_type == "ucond":
            assert self.cls_token_num == 0, "cls_token_num must be 0 for ucond model"
            self.cls_embedding = UcondEmbedder(hidden_size=config.dim)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)
        self.packed_inputs = config.packed_inputs

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id], packed_inputs=self.packed_inputs))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

    def forward(
        self,
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        targets: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        if self.packed_inputs:
            assert targets is not None, f"packing is currently only valid for training"
            output = self._forward_packed(idx, cond_idx, targets, *args, **kwargs)
            return output
        else:
            return self._forward_simple(idx, cond_idx, targets, *args, **kwargs)


    def _forward_simple(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        targets: Optional[torch.Tensor] = None,
        input_pos:  Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        attn_impl: str = "xformers",
    ):
        if idx is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
        else:
            if cond_idx is not None: # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)
            
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis
        
        if self.training or input_pos is None:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask, attn_impl=attn_impl)
        
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()
        
        if self.training:
            logits = logits[:, max(0, self.cls_token_num - 1):].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1))

        return logits, loss
    
    def _forward_packed(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        targets: torch.Tensor,
        seqlens: torch.Tensor,
    ):
        assert self.cls_token_num == 1, f"Only 1 class token number is supported"

        # create token embeddings
        cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num].view(-1, self.config.dim)
        token_embeddings = self.tok_embeddings(idx)

        # add condition emebddings
        seqlens = seqlens + 1
        start_pos = torch.cumsum(seqlens, dim=-1) - seqlens
        token_embeddings = scatter_at_pos(token_embeddings, cond_embeddings, start_pos)

        # token dropout
        h = self.tok_dropout(token_embeddings)

        # freq embeddings
        _freqs_cis = map_pos_to_freq(seqlens, self.freqs_cis)
        
        # transformer blocks
        for layer in self.layers:
            h = layer(h, _freqs_cis, seqlens=seqlens)
        
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()

        # if we are given some desired targets also calculate the loss
        loss = F.cross_entropy(logits, targets)

        return logits, loss


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)
    
    def move_buffers_to_device(self, device) -> None:
        self.freqs_cis = self.freqs_cis.to(device)




#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis_bsz = freqs_cis.shape[0] if freqs_cis.dim() == 4 else 1
    freqs_cis = freqs_cis.view(freqs_cis_bsz, xshaped.size(1), 1, xshaped.size(3), 2) # (bsz, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        
def GPT_T(**kwargs):
    return Transformer(ModelArgs(n_layer=6, n_head=8, dim=512, **kwargs)) # 37M

GPT_models = {
    'GPT-T': GPT_T, 'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}