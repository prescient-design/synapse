import math
from typing import Optional

import torch


def motif_search(
    solution: torch.LongTensor,
    motif: torch.LongTensor,
    spacing: Optional[torch.LongTensor] = None,
    mode: str = "present",
    quantization: Optional[int] = None,   # number of bins; steps are 1/quantization
    emphasis: float = 2.0,                # >=1 makes higher matches weigh more (convex)
):
    """
    Check if a spaced motif is present in a solution.
    The elements of the motif should be spaced according to `spacing`.
    If spacing is not provided, it is assumed to be 1.
    If `mode` is "count", the (quantized) motif evidence is summed over positions.
    """
    if emphasis < 1.0:
        raise ValueError("emphasis should be >= 1.0 to overweight higher matches")

    motif_size = motif.size(-1)
    if quantization is None:
        quantization = motif_size  # default: ~per-position resolution

    size = solution.size()[:-1]
    if spacing is None:
        spacing = torch.ones(*size, motif_size - 1, dtype=torch.long, device=solution.device)
    else:
        spacing = spacing.expand(*size, -1)

    # convert spacing into index tensor for gather
    base_index = torch.cat(
        [torch.zeros_like(spacing[..., 0]).unsqueeze(-1), spacing.cumsum(-1)],
        dim=-1
    )  # shape: (*size, motif_size)

    # slide indices out to maximum length
    max_base = int(base_index.max().item())
    num_steps = solution.size(-1) - max_base

    # Handle case where motif + spacing is longer than solution
    if num_steps <= 0:
        batch_shape = solution.shape[:-1]
        if mode in ("present", "count"):
            return torch.zeros(batch_shape, dtype=torch.float32, device=solution.device)
        raise ValueError(f"Unknown mode: {mode}")

    index_delta = torch.arange(num_steps, device=solution.device)
    index_delta = index_delta.view(num_steps, *([1] * base_index.dim()))
    index = base_index + index_delta  # (num_steps, *size, motif_size)

    # gather solution subsequences
    gathered = torch.stack([torch.gather(solution, -1, step) for step in index])  # (num_steps, *size, motif_size)

    # elementwise compare, then count matches per window
    is_equal = gathered.eq(motif)
    present_count = is_equal.sum(-1).to(torch.float32)  # (num_steps, *size)

    # --- Weighted quantization: bins are uniform in the transformed domain ---
    # frac in [0,1]; apply convex transform frac**emphasis; quantize into k bins
    k = int(quantization)
    if k <= 0:
        raise ValueError("quantization must be a positive integer")

    frac = (present_count / float(motif_size)).clamp_(0.0, 1.0)
    transformed = torch.pow(frac, emphasis)
    quantized = torch.floor(transformed * k) / float(k)   # ∈ {0, 1/k, ..., 1}

    if mode == "count":
        # Sum transformed, quantized evidence across positions
        return quantized.sum(0) 

    if mode == "present":
        # Best local (transformed, quantized) score
        return quantized.max(0).values 

    raise ValueError(f"Unknown mode: {mode}")