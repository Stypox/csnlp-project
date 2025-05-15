# code adapted from TinyLlama repo

import torch

def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin

def build_mask_cache(
    device: torch.device,
    block_size: int,
) -> torch.Tensor:
    ones = torch.ones((block_size, block_size), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)

def build_kv_caches(
    device: torch.device,
    batch_size: int,
    max_seq_length: int,
    rope_cache_length: int,
    heads: int,
    head_size: int,
    rotary_percentage: float,
    n_layer: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    k_cache_shape = (
        batch_size,
        max_seq_length,
        heads,
        rope_cache_length + head_size - int(rotary_percentage * head_size),
    )
    v_cache_shape = (
        batch_size,
        max_seq_length,
        heads,
        head_size,
    )
    return [
        (
            torch.zeros(k_cache_shape, device=device),
            torch.zeros(v_cache_shape, device=device),
        )
        for _ in range(n_layer)
    ]