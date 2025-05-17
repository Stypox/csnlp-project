# code adapted from TinyLlama repo

import torch
from torch import nn
from typing import override, cast
from xformers.ops import SwiGLU
from causal_self_attention import CausalSelfAttention
from cache import build_kv_caches, build_mask_cache, build_rope_cache

# parameters taken from lil_gpt/config.py and lil_gpt/model.py in TinyLlama repo
n_embd = 2048
vocab_size = 32000
n_layer = 22
norm_eps = 1e-5 #Llama 2 use 1e-5. Llama 1 use 1e-6
intermediate_size = 5632
n_head = 32
n_query_groups = 4
head_size = n_embd // n_head
block_size = 2048
rotary_percentage = 1.0
condense_ratio = 1

class TinyLlama(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_tokens: nn.Embedding = nn.Embedding(vocab_size, n_embd)
        self.layers: nn.ModuleList = nn.ModuleList(
            Block() for _ in range(n_layer)
        )
        # FusedRmsNorm is not in torch, and this should be equivalent to it, albeit slower
        # https://github.com/pytorch/pytorch/issues/72643#issuecomment-1492583637
        self.norm: nn.RMSNorm = nn.RMSNorm(n_embd, eps=norm_eps)
        self.lm_head: nn.Linear = nn.Linear(n_embd, vocab_size, bias=False)

        self.rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self.mask_cache: torch.Tensor | None = None
        self.kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []

    @override
    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: int | None = None,
        input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        _B, T = idx.size()
        use_kv_cache = input_pos is not None

        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = build_rope_cache(
                seq_len=block_size,
                n_elem=int(rotary_percentage * head_size),
                dtype=self.embed_tokens.weight.dtype,
                device=idx.device,
                condense_ratio=condense_ratio,
            )
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = build_mask_cache(idx.device, block_size)

        cos, sin = self.rope_cache
        if use_kv_cache:
            cos = cos.index_select(0, input_pos)
            sin = sin.index_select(0, input_pos)
            mask = cast(torch.Tensor, self.mask_cache).index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            cos = cos[:T]
            sin = sin[:T]
            mask = None

        # forward the model itself
        x = self.embed_tokens(idx)  # token embeddings of shape (b, t, n_embd)

        if not use_kv_cache:
            for block in self.layers:
                x, *_ = block(x, (cos, sin), max_seq_length)
        else:
            self.kv_caches = self.kv_caches or build_kv_caches(
                x.device,
                x.size(0),
                max_seq_length,
                cos.size(-1) * 2,
                n_query_groups,
                head_size,
                rotary_percentage,
                n_layer,
            )
            for i, block in enumerate(self.layers):
                x, self.kv_caches[i] = block(x, (cos, sin), max_seq_length, mask, input_pos, self.kv_caches[i])

        x = self.norm(x)

        return self.lm_head(x)  # (b, t, vocab_size)

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layernorm: nn.RMSNorm = nn.RMSNorm(normalized_shape=n_embd, eps=norm_eps)
        self.self_attn: CausalSelfAttention = CausalSelfAttention(n_embd, n_head, n_query_groups, head_size)
        self.post_attention_layernorm: nn.RMSNorm = nn.RMSNorm(normalized_shape=n_embd, eps=norm_eps)
        self.mlp: SwiGLU = SwiGLU(n_embd, intermediate_size, bias=False, _pack_weights=False)

    @override
    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        max_seq_length: int,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        n_1 = self.input_layernorm(x)
        h, new_kv_cache = self.self_attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv_cache
