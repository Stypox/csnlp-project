# code taken from TinyLlama repo

import math
from typing import cast, override
import torch
from torch import nn
from .fused_rotary_embedding import apply_rotary_emb_func

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_query_groups: int, head_size: int) -> None:
        super().__init__()
        shape = (n_head + 2 * n_query_groups) * head_size
        # key, query, value projections for all heads, but in a batch
        self.attn: nn.Linear = nn.Linear(n_embd, shape, bias=False)
        # output projection
        self.proj: nn.Linear = nn.Linear(n_embd, n_embd, bias=False)

        self.n_head: int = n_head
        self.n_query_groups: int = n_query_groups
        self.head_size: int = head_size

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
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.n_query_groups, total_qkv, self.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B,  T, -1, self.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.head_size)
        v = v.reshape(B,  T, -1, self.head_size)

        cos, sin = rope

        # apply rope in fp32 significanly stabalize training
        # fused rope expect (batch_size, seqlen, nheads, headdim)
        q = cast(torch.Tensor, apply_rotary_emb_func(q, cos, sin, False, True))
        k = cast(torch.Tensor, apply_rotary_emb_func(k, cos, sin, False, True))

        # n_elem = int(self.config.rotary_percentage * self.config.head_size)

        # q_roped = apply_rope(q[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # k_roped = apply_rope(k[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # print( (q_roped - q).sum())
        # q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        # k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        if kv_cache is not None:
            assert input_pos is not None

            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ):
        scale = 1.0 / math.sqrt(self.head_size)

        # if (
        #     FlashAttention2Available
        #     and mask is None
        #     and q.device.type == "cuda"
        #     and q.dtype in (torch.float16, torch.bfloat16)
        # ):
        #     from flash_attn import flash_attn_func

        #     return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)