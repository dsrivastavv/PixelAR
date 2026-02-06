# Base Model Cloned from LLaMAGEN
# Not used in our RandAR
# Modified from:
#   LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
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
from torch.utils.checkpoint import checkpoint

from dynamic_tokenization.models.local_models import DecoderModelArgs, EncoderModelArgs, LocalEncoder, LocalDecoder, cross_attn_mask
from dynamic_tokenization.models.patcher import patch_ids_from_lengths
from dynamic_tokenization.models.gpt import RMSNorm, get_start_pos_from_seqlens, precompute_freqs_cis, precompute_freqs_cis_2d, LabelEmbedder, TransformerBlock, scatter_at_pos, map_pos_to_freq
from typing import Optional



@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 25
    n_local_encoder_layers: int = 1
    n_local_decoder_layers: int = 9
    n_head: int = 16
    # n_local_encoder_head: int = 16
    # n_local_decoder_head: int = 16
    # n_local_encoder_cross_attn_head: int = 16
    # n_local_decoder_cross_attn_head: int = 16
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

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    predict_eoi_token: bool = False # add end of image token prediction by global model
    embedding_type: str | None = None
    encoder_block_causal: bool = False

    packed_inputs: bool = False
    use_ca_rope: bool = False
    gradient_checkpointing: bool = False

def get_block_causal_mask(patch_ids: torch.Tensor, is_causal: bool = True):
    bsz, seqlen = patch_ids.shape
    patch_block_mask = (patch_ids.reshape(bsz, 1, 1, seqlen) == patch_ids.reshape(bsz, 1, seqlen, 1))
    causal_mask = torch.tril(torch.ones((bsz, 1, seqlen, seqlen), device=patch_ids.device)) > 0
    mask = patch_block_mask & causal_mask
    return mask    
        
class DynamicTransformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.packed_inputs = config.packed_inputs
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        
        self.EOI_TOKEN = -1
        if config.predict_eoi_token:
            self.EOI_TOKEN = self.vocab_size
            self.vocab_size = self.vocab_size + 1
        self.tok_embeddings = nn.Embedding(self.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # HACK: code for more tokens is not checked
        assert self.cls_token_num == 1, "Code has not been checked for more than 1 token"
        if config.n_local_encoder_layers > 0 or config.n_local_decoder_layers > 0:
            assert config.n_local_encoder_layers and config.n_local_decoder_layers > 0, "Both encoder and decoder should be present if either is present"

        # local embedder 
        self.local_encoder = None
        if config.n_local_encoder_layers > 0:
            self.local_encoder = LocalEncoder(
                EncoderModelArgs(
                    dim=config.dim,
                    n_layer=config.n_local_encoder_layers,
                    n_head=config.n_head,
                    n_cross_attn_head=config.n_head,
                    multiple_of=config.multiple_of,
                    ffn_dim_multiplier=config.ffn_dim_multiplier,
                    rope_base=config.rope_base,
                    norm_eps=config.norm_eps,
                    initializer_range=config.initializer_range,
                    token_dropout_p=0.0,
                    attn_dropout_p=0.0,
                    resid_dropout_p=0.0,
                    ffn_dropout_p=0.0,
                    drop_path_rate=0.0,
                    block_size=config.block_size,
                    gradient_checkpointing=config.gradient_checkpointing,
                    use_ca_rope=config.use_ca_rope,
                ),
                packed_inputs=self.packed_inputs
            )

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id], packed_inputs=self.packed_inputs))

        # local decoder
        self.local_decoder = None
        if config.n_local_decoder_layers > 0:
            self.local_decoder = LocalDecoder(
                DecoderModelArgs(
                    dim=config.dim,
                    n_layer=config.n_local_decoder_layers,
                    n_head=config.n_head,
                    n_cross_attn_head=config.n_head,
                    n_kv_head=config.n_kv_head,
                    multiple_of=config.multiple_of,
                    ffn_dim_multiplier=config.ffn_dim_multiplier,
                    rope_base=config.rope_base,
                    norm_eps=config.norm_eps,
                    initializer_range=config.initializer_range,
                    token_dropout_p=0.0,
                    attn_dropout_p=0.0,
                    resid_dropout_p=0.0,
                    ffn_dropout_p=0.0,
                    drop_path_rate=config.drop_path_rate,
                    cls_token_num=config.cls_token_num,
                    block_size=config.block_size,
                    gradient_checkpointing=config.gradient_checkpointing,
                ),
                packed_inputs=self.packed_inputs
            )
            self.decoder_start_embedding = nn.Embedding(1, self.config.dim * self.cls_token_num)    

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, self.vocab_size, bias=False)

        # 1D position embedding
        self.grid_size = int(self.block_size ** 0.5)
        assert self.grid_size * self.grid_size == self.block_size
        self.freqs_cis = None
        if config.embedding_type is None:
            if config.n_local_encoder_layers == 0:
                print(f"Using 2D Rope Embeddings with type {config.embedding_type}")
                self.freqs_cis = precompute_freqs_cis_2d(self.grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
            else:
                print("Using 1D Rope Embeddings")
                self.freqs_cis = precompute_freqs_cis(self.grid_size * self.grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        else:
            print(f"Using dynamic embeddings with type {config.embedding_type}")


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


    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        patch_lens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        if self.packed_inputs:
            assert targets is not None, f"packing is currently only valid for training"
            output = self._forward_packed(idx, cond_idx, patch_lens, targets, *args, **kwargs)
            return output
        else:
            return self._forward_simple(idx, cond_idx, patch_lens, targets, *args, **kwargs)

    def _forward_simple(
        self,        
        idx: torch.Tensor,
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        patch_lens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        input_pos:  Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        attn_impl: str = "xformers",
    ):
        bsz, seq_len = idx.shape # (bsz, block_size)
        if idx is not None and cond_idx is not None: # training or naive inference
            # condition embedding
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)
            assert cond_embeddings.shape[1] == self.cls_token_num, f"class embedding has incorrect token numbers"
            
            # token embedding
            assert not (idx == self.EOI_TOKEN).any(), f"EOI token should not be in the input"
            token_embeddings = self.tok_embeddings(idx)

            # local encoder logic
            h_encoder = None
            if self.local_encoder:
                # get patch ids
                patch_ids = patch_ids_from_lengths(
                    patch_lens, seq_len
                ) # (bsz, seq_len)

                # setup masks
                self_attn_mask = get_block_causal_mask(patch_ids) if self.config.encoder_block_causal else None # (bsz, seq_len, seq_len)
                cross_attn_mask_enc = cross_attn_mask(
                    patch_ids,
                    min(patch_lens.shape[-1], seq_len), # we need to deal with maximum of seq_len patches
                    seq_len,
                    patches_as_queries=True,
                    cross_attn_k=1,
                    block_mask=False, # to be compatible with flex attention
                ) # (bsz, 1, max_patches, seq_len)

                # local encoder
                h_encoder, h_cross = self.local_encoder(
                    token_embeddings=token_embeddings,
                    num_patches=patch_lens[:, :seq_len].shape[1],
                    patch_ids=patch_ids,
                    cross_mask=cross_attn_mask_enc,
                    mask=self_attn_mask,
                    attn_impl=attn_impl
                ) # (bsz, seq_len, dim), (bsz, max_patches, seq_len)
                token_embeddings = torch.cat((cond_embeddings, h_cross), dim=1)
            else:
                token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
        else:
            raise NotImplementedError("Check this code block")
            if cond_idx is not None: # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)
            
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis

        if self.config.embedding_type == "dynamic_3d":
            freqs_cis = self.get_freqs_cis_3d(patch_lens)
            if self.training or input_pos is None:
                freqs_cis = freqs_cis[:, :h.size(1)]
            else:
                raise NotImplementedError
        elif self.config.embedding_type == "dynamic_4d":
            freqs_cis = self.get_freqs_cis_4d(patch_lens)
            if self.training or input_pos is None:
                freqs_cis = freqs_cis[:, :h.size(1)]
            else:
                raise NotImplementedError
        else:
            if self.training or input_pos is None:
                freqs_cis = self.freqs_cis[:h.shape[1]]
            else:
                freqs_cis = self.freqs_cis[input_pos]

        # transformer blocks
        for layer in self.layers:
            if self.config.gradient_checkpointing:
                h = checkpoint(layer, h, freqs_cis, input_pos, mask, attn_impl, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, input_pos, mask, attn_impl=attn_impl)

        # Step 3: Run local decoder
        if self.local_decoder:
            assert h_encoder is not None, "Embeddings must be provided"
            encoder_cond = self.decoder_start_embedding.weight.reshape(1, self.cls_token_num, self.config.dim).repeat(bsz,1,1)
            h_encoder =  torch.cat((encoder_cond, h_encoder), dim=1)
            decoder_patch_ids = patch_ids_from_lengths(
                patch_lengths=patch_lens, seq_len=seq_len+1
            ) # same ouput as patch_ids but with the last token added
            cross_attn_mask_dec = cross_attn_mask(
                decoder_patch_ids,
                h.shape[-2],
                seq_len+1,
                patches_as_queries=False,
                cross_attn_k=1,
                block_mask=False,
            )
            h = self.local_decoder(
                embeds=h_encoder,
                patch_embeds=h,
                cross_mask=cross_attn_mask_dec,
                attn_impl=attn_impl
            )

        # output layers
        h = self.norm(h)
        logits = self.output(h).float()
        
        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()

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
        cond_idx: torch.Tensor,
        patch_lens: torch.Tensor,
        targets: torch.Tensor,
        patch_lens_w_next: torch.Tensor,
        token_seqlens: torch.Tensor,
        patch_seqlens: torch.Tensor,
    ):
        bsz = cond_idx.shape[0]
        assert self.cls_token_num == 1, f"Only 1 class token number is supported"

        # condition embedding
        cond_embeddings = self.cls_embedding(cond_idx, train=self.training)
        assert cond_embeddings.shape[1] == self.cls_token_num, f"class embedding has incorrect token numbers"
        cond_embeddings = cond_embeddings.view(-1, self.config.dim)
        
        # token embedding
        assert not (idx == self.EOI_TOKEN).any(), f"EOI token should not be in the input"
        token_embeddings = self.tok_embeddings(idx)

        # local encoder logic
        if self.local_encoder:
            # local encoder
            h_encoder, h_cross = self.local_encoder(
                token_embeddings=token_embeddings,
                patch_lengths=patch_lens,
                token_seqlens=token_seqlens,
            )
        else:
            h_encoder = None
            h_cross = token_embeddings
            patch_seqlens = token_seqlens

        # add condition emebddings
        patch_seqlens_w_class_token = patch_seqlens + self.cls_token_num
        patch_w_class_start_pos = get_start_pos_from_seqlens(patch_seqlens_w_class_token)
        h = scatter_at_pos(h_cross, cond_embeddings, patch_w_class_start_pos)
        
        # token dropout
        h = self.tok_dropout(h)
        
        # position embedding
        if self.config.embedding_type == "dynamic_3d":
            _freqs_cis = self.get_freqs_cis_3d_packed(patch_lens, patch_seqlens, token_seqlens, patch_w_class_start_pos)
        elif self.config.embedding_type == "dynamic_4d":
            _freqs_cis = self.get_freqs_cis_4d_packed(patch_lens, patch_seqlens, token_seqlens, patch_w_class_start_pos)
        else:
            _freqs_cis = map_pos_to_freq(patch_seqlens_w_class_token, self.freqs_cis)

        # transformer blocks
        for layer in self.layers:
            if self.config.gradient_checkpointing:
                h = checkpoint(layer, h, _freqs_cis, patch_seqlens_w_class_token, use_reentrant=False)
            else:
                h = layer(h, _freqs_cis, seqlens=patch_seqlens_w_class_token)

        # Step 3: Run local decoder
        if self.local_decoder:
            assert h_encoder is not None, "Embeddings must be provided"
            
            # update token embedding with class token
            encoder_cond = self.decoder_start_embedding.weight.reshape(self.cls_token_num, self.config.dim).repeat(bsz,1)
            token_seqlens_w_class = token_seqlens + self.cls_token_num
            token_w_class_start_pos = get_start_pos_from_seqlens(token_seqlens_w_class)
            h_encoder_w_embed = scatter_at_pos(h_encoder, encoder_cond, token_w_class_start_pos)
            
            # local decoder
            h = self.local_decoder(
                embeds=h_encoder_w_embed,
                patch_embeds=h,
                patch_lens=patch_lens_w_next,
                patch_seqlens=patch_seqlens_w_class_token,
                token_seqlens=token_seqlens_w_class,
            )

        # output layers
        h = self.norm(h)
        logits = self.output(h).float()

        # if we are given some desired targets also calculate the loss
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)
    
    def move_buffers_to_device(self, device) -> None:
        if self.local_encoder:
            self.local_encoder.move_buffers_to_device(device)
        if self.local_decoder:
            self.local_decoder.move_buffers_to_device(device)
        if self.freqs_cis is not None:
            self.freqs_cis = self.freqs_cis.to(device)


    @torch.no_grad()
    @torch.compiler.disable
    def get_freqs_cis_3d(self, patch_lengths: torch.Tensor):
        # compute patch index
        grid_size = self.grid_size
        n_elem = self.config.dim // self.config.n_head 
        base = self.config.rope_base 
        cls_token_num = self.cls_token_num

        # patch start and end index
        cum_patch_length = torch.cumsum(patch_lengths, dim=-1)
        start_patch_index = torch.cat(
            (torch.zeros_like(cum_patch_length[:, :1]), cum_patch_length[:, :-1]),
            dim=-1
        )
        end_patch_index = cum_patch_length - 1

        # dim embeds
        half_dim = n_elem // 2
        half_half_dim = half_dim // 2
        freqs_half_dim = 1.0 / (
            base ** (torch.arange(0, half_dim, 2, device=patch_lengths.device)[: (half_dim // 2)].float() / half_dim)
        )
        freqs_half_half_dim = 1.0 / (
            base ** (torch.arange(0, half_half_dim, 2, device=patch_lengths.device)[: (half_half_dim // 2)].float() / half_half_dim)
        )

        # index embed
        cord_x = start_patch_index // grid_size
        cord_y_start = start_patch_index % grid_size
        cord_y_end = end_patch_index % grid_size
        freqs_x = cord_x[:, :, None] * freqs_half_dim[None, None, :] # (grid_size, head_dim // 2)
        freqs_y_start = cord_y_start[:, :, None] * freqs_half_half_dim[None, None, :]
        freqs_y_end = cord_y_end[:, :, None] * freqs_half_half_dim[None, None, :]

        # generate embeddings
        freqs_grid = torch.cat([freqs_x, freqs_y_start, freqs_y_end], dim=-1) # (num_patches, head_dim // 2)
        cache = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (num_patches, head_dim // 2, 2)
        cond_cache = torch.cat([
            torch.zeros(
                patch_lengths.size(0), cls_token_num, n_elem // 2, 2,
                device=patch_lengths.device
            ), 
            cache
        ], dim=1) # (bsz, 1+num_patches, head_dim // 2, 2)
        return cond_cache
    
    @torch.no_grad()
    @torch.compiler.disable
    def get_freqs_cis_4d(self, patch_lengths: torch.Tensor):
        # compute patch index
        grid_size = self.grid_size
        n_elem = self.config.dim // self.config.n_head 
        base = self.config.rope_base 
        cls_token_num = self.cls_token_num

        # patch start and end index
        cum_patch_length = torch.cumsum(patch_lengths, dim=-1)
        start_patch_index = torch.cat(
            (torch.zeros_like(cum_patch_length[:, :1]), cum_patch_length[:, :-1]),
            dim=-1
        )
        end_patch_index = cum_patch_length - 1

        # dim embeds
        half_dim = n_elem // 2
        half_half_dim = half_dim // 4
        freqs_half_dim = 1.0 / (
            base ** (torch.arange(0, half_dim, 2, device=patch_lengths.device)[: (half_dim // 2)].float() / half_dim)
        )
        freqs_half_half_dim = 1.0 / (
            base ** (torch.arange(0, half_half_dim, 2, device=patch_lengths.device)[: (half_half_dim // 2)].float() / half_half_dim)
        )

        # index embed
        cord_x = start_patch_index // grid_size
        cord_y_start = start_patch_index % grid_size
        cord_y_end = end_patch_index % grid_size
        freqs_x = cord_x[:, :, None] * freqs_half_dim[None, None, :] # (grid_size, head_dim // 2)
        freqs_y_start = cord_y_start[:, :, None] * freqs_half_half_dim[None, None, :]
        freqs_y_end = cord_y_end[:, :, None] * freqs_half_half_dim[None, None, :]

        # generate embeddings
        freqs_grid = torch.cat([freqs_x, freqs_y_start, freqs_y_end, freqs_y_end, freqs_y_start], dim=-1) # (num_patches, head_dim // 2)
        cache = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (num_patches, head_dim // 2, 2)
        cond_cache = torch.cat([
            torch.zeros(
                patch_lengths.size(0), cls_token_num, n_elem // 2, 2,
                device=patch_lengths.device
            ), 
            cache
        ], dim=1) # (bsz, 1+num_patches, head_dim // 2, 2)
        return cond_cache
    
    @torch.no_grad()
    def get_freqs_cis_3d_packed(
        self, 
        patch_lengths: torch.Tensor, 
        patch_seqlen: torch.Tensor, 
        token_seqlens: torch.Tensor, 
        patch_w_class_start_pos: torch.Tensor
    ):
        # add token start and end index for each patch. The token ids start at one since class token is at position 0
        batch_start_pos = torch.repeat_interleave(torch.cumsum(token_seqlens, dim=-1)-token_seqlens, patch_seqlen)
        patch_abs_token_start_pos = torch.cumsum(patch_lengths, dim=-1) - patch_lengths
        patch_start_token_index = patch_abs_token_start_pos - batch_start_pos
        patch_end_token_index = patch_start_token_index + patch_lengths - 1

        # compute patch index
        grid_size = self.grid_size
        n_elem = self.config.dim // self.config.n_head 
        base = self.config.rope_base 
        cls_token_num = self.cls_token_num

        # dim embeds
        half_dim = n_elem // 2
        half_half_dim = half_dim // 2
        freqs_half_dim = 1.0 / (
            base ** (torch.arange(0, half_dim, 2, device=patch_lengths.device)[: (half_dim // 2)].float() / half_dim)
        )
        freqs_half_half_dim = 1.0 / (
            base ** (torch.arange(0, half_half_dim, 2, device=patch_lengths.device)[: (half_half_dim // 2)].float() / half_half_dim)
        )

        # index embed
        cord_x = patch_start_token_index // grid_size
        cord_y_start = patch_start_token_index % grid_size
        cord_y_end = patch_end_token_index % grid_size
        freqs_x = torch.outer(cord_x, freqs_half_dim) # (grid_size, head_dim // 2)
        freqs_y_start = torch.outer(cord_y_start, freqs_half_half_dim)
        freqs_y_end = torch.outer(cord_y_end, freqs_half_half_dim)

        # generate embeddings
        freqs_grid = torch.cat([freqs_x, freqs_y_start, freqs_y_end], dim=-1) # (num_patches, head_dim // 2)
        cache = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (num_patches, head_dim // 2, 2)
        cond_cache = torch.zeros((cache.size(0) + patch_seqlen.size(0) * cls_token_num, n_elem // 2, 2), device=patch_lengths.device)
        freqs_grid_mask = torch.ones(cond_cache.size(0), dtype=torch.bool, device=cond_cache.device)
        freqs_grid_mask[patch_w_class_start_pos] = False
        cond_cache[freqs_grid_mask] = cache
        
        return cond_cache

    @torch.no_grad()
    def get_freqs_cis_4d_packed(
        self, 
        patch_lengths: torch.Tensor, 
        patch_seqlen: torch.Tensor, 
        token_seqlens: torch.Tensor, 
        patch_w_class_start_pos: torch.Tensor
    ):
        # add token start and end index for each patch. The token ids start at one since class token is at position 0
        batch_start_pos = torch.repeat_interleave(torch.cumsum(token_seqlens, dim=-1)-token_seqlens, patch_seqlen)
        patch_abs_token_start_pos = torch.cumsum(patch_lengths, dim=-1) - patch_lengths
        patch_start_token_index = patch_abs_token_start_pos - batch_start_pos
        patch_end_token_index = patch_start_token_index + patch_lengths - 1

        # compute patch index
        grid_size = self.grid_size
        n_elem = self.config.dim // self.config.n_head 
        base = self.config.rope_base 
        cls_token_num = self.cls_token_num

        # dim embeds
        half_dim = n_elem // 2
        half_half_dim = half_dim // 4
        freqs_half_dim = 1.0 / (
            base ** (torch.arange(0, half_dim, 2, device=patch_lengths.device)[: (half_dim // 2)].float() / half_dim)
        )
        freqs_half_half_dim = 1.0 / (
            base ** (torch.arange(0, half_half_dim, 2, device=patch_lengths.device)[: (half_half_dim // 2)].float() / half_half_dim)
        )

        # index embed
        cord_x = patch_start_token_index // grid_size
        cord_y_start = patch_start_token_index % grid_size
        cord_y_end = patch_end_token_index % grid_size
        freqs_x = torch.outer(cord_x, freqs_half_dim) # (grid_size, head_dim // 2)
        freqs_y_start = torch.outer(cord_y_start, freqs_half_half_dim)
        freqs_y_end = torch.outer(cord_y_end, freqs_half_half_dim)

        # generate embeddings
        freqs_grid = torch.cat([freqs_x, freqs_y_start, freqs_y_end, freqs_y_end, freqs_y_start], dim=-1) # (num_patches, head_dim // 2)
        cache = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (num_patches, head_dim // 2, 2)
        cond_cache = torch.zeros((cache.size(0) + patch_seqlen.size(0) * cls_token_num, n_elem // 2, 2), device=patch_lengths.device)
        freqs_grid_mask = torch.ones(cond_cache.size(0), dtype=torch.bool, device=cond_cache.device)
        freqs_grid_mask[patch_w_class_start_pos] = False
        cond_cache[freqs_grid_mask] = cache
        
        return cond_cache


### class-conditional
def DGPT_XXXL(**kwargs):
    return DynamicTransformer(ModelArgs(n_layer=41, n_local_encoder_layers=1, n_local_decoder_layers=6, n_head=40, dim=2560, **kwargs)) 

def DGPT_XXL(**kwargs):
    return DynamicTransformer(ModelArgs(n_layer=41, n_local_encoder_layers=1, n_local_decoder_layers=6, n_head=24, dim=1536, **kwargs)) 

def DGPT_XL(**kwargs):
    return DynamicTransformer(ModelArgs(n_layer=30, n_local_encoder_layers=1, n_local_decoder_layers=5, n_head=20, dim=1280, **kwargs)) # 853M

def DGPT_L(**kwargs):
    return DynamicTransformer(ModelArgs(n_layer=19, n_local_encoder_layers=1, n_local_decoder_layers=4, n_head=16, dim=1024, **kwargs)) # 376M

def DGPT_LE2D3(**kwargs):
    # Note: This model is used for ablation study in the paper
    return DynamicTransformer(ModelArgs(n_layer=19, n_local_encoder_layers=2, n_local_decoder_layers=3, n_head=16, dim=1024, **kwargs))

def DGPT_LE3D2(**kwargs):
    # Note: This model is used for ablation study in the paper
    return DynamicTransformer(ModelArgs(n_layer=19, n_local_encoder_layers=3, n_local_decoder_layers=2, n_head=16, dim=1024, **kwargs))

def DGPT_LE4D1(**kwargs):
    # Note: This model is used for ablation study in the paper
    return DynamicTransformer(ModelArgs(n_layer=19, n_local_encoder_layers=4, n_local_decoder_layers=1, n_head=16, dim=1024, **kwargs))

def DGPT_B(**kwargs):
    return DynamicTransformer(ModelArgs(n_layer=8, n_local_encoder_layers=1, n_local_decoder_layers=3, n_head=12, dim=768, **kwargs)) # 120M
        

DGPT_models = {
    'DGPT-B': DGPT_B, 'DGPT-L': DGPT_L, 'DGPT-XL': DGPT_XL, 'DGPT-XXL': DGPT_XXL, 'DGPT-XXXL': DGPT_XXXL,
    'DGPT-L-E2D3': DGPT_LE2D3, 'DGPT-L-E3D2': DGPT_LE3D2, 'DGPT-L-E4D1': DGPT_LE4D1,
}

