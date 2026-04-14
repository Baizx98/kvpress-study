from __future__ import annotations

import math

import torch


SUMMARY_MODES = {
    "mean_only",
    "norm_topk_mean_only",
    "mean_plus_norm_topk_mean",
    "multi_rep_max",
    "adaptive_fusion_v1",
}

REPRESENTATIVE_MODES = {
    "key_norm",
    "key_norm_diverse",
    "tail_query_relevance",
    "random_topk",
}

QUERY_AGG_MODES = {"mean", "max", "topr_mean", "adaptive_mean_max_v1"}
HEAD_AGG_MODES = {"uniform_mean", "strength_weighted", "top_head_only"}


def aggregate_query_scores(
    scores: torch.Tensor,
    mode: str,
    topr: int,
) -> torch.Tensor:
    if scores.shape[-2] == 0:
        return scores.new_zeros(scores.shape[:-2] + (scores.shape[-1],))
    if mode == "mean":
        return scores.mean(dim=-2)
    if mode == "max":
        return scores.max(dim=-2).values
    if mode == "topr_mean":
        actual_topr = max(1, min(topr, scores.shape[-2]))
        return scores.topk(actual_topr, dim=-2).values.mean(dim=-2)
    if mode == "adaptive_mean_max_v1":
        mean_scores = scores.mean(dim=-2)
        max_scores = scores.max(dim=-2).values
        score_range = max_scores - scores.min(dim=-2).values
        score_mean_abs = scores.abs().mean(dim=-2).clamp_min(1e-6)
        gate = (score_range / score_mean_abs).sigmoid()
        return gate * max_scores + (1.0 - gate) * mean_scores
    raise ValueError(f"Unsupported query aggregation mode: {mode}")


def aggregate_head_scores(
    scores: torch.Tensor,
    mode: str,
    eps: float,
    topk: int = 1,
) -> torch.Tensor:
    if scores.shape[1] == 0:
        return scores.new_zeros((scores.shape[0], scores.shape[-1]))
    if mode == "uniform_mean":
        return scores.mean(dim=1)
    if mode == "strength_weighted":
        strengths = scores.abs().amax(dim=-1).clamp_min(eps)
        weights = strengths / strengths.sum(dim=1, keepdim=True).clamp_min(eps)
        return (scores * weights[:, :, None]).sum(dim=1)
    if mode == "top_head_only":
        head_strengths = scores.abs().amax(dim=-1)
        actual_topk = max(1, min(topk, scores.shape[1]))
        top_heads = head_strengths.topk(actual_topk, dim=1).indices
        gather_index = top_heads[:, :, None].expand(-1, -1, scores.shape[-1])
        return scores.gather(1, gather_index).mean(dim=1)
    raise ValueError(f"Unsupported head aggregation mode: {mode}")


def deterministic_random_scores(
    shape: tuple[int, int, int],
    block_start: int,
    seed: int,
    layer_idx: int,
    device: torch.device,
) -> torch.Tensor:
    token_offsets = torch.arange(shape[-1], device=device, dtype=torch.float32)
    base = token_offsets + float(block_start + 1 + seed * 17 + layer_idx * 101)
    noise = torch.sin(base * 12.9898) * 43758.5453
    noise = torch.remainder(noise, 1.0)
    return noise.view(1, 1, -1).expand(shape[0], shape[1], -1)


def select_representative_indices(
    mode: str,
    block_keys: torch.Tensor,
    representative_k: int,
    block_start: int,
    layer_idx: int,
    seed: int,
    tail_query_states: torch.Tensor | None = None,
) -> torch.Tensor:
    block_len = block_keys.shape[2]
    actual_topk = max(1, min(representative_k, block_len))

    if mode == "key_norm":
        selector_scores = block_keys.norm(dim=-1)
        return selector_scores.topk(actual_topk, dim=-1).indices
    if mode == "key_norm_diverse":
        selector_scores = block_keys.norm(dim=-1)
        sorted_indices = selector_scores.argsort(dim=-1, descending=True)
        selected = []
        for rank in range(sorted_indices.shape[-1]):
            candidate = sorted_indices[..., rank]
            if not selected:
                selected.append(candidate)
            else:
                stacked = torch.stack(selected, dim=-1)
                min_distance = (candidate[..., None] - stacked).abs().amin(dim=-1)
                distance_threshold = max(1, math.ceil(block_len / max(actual_topk, 1)) - 1)
                accept = min_distance >= distance_threshold
                fallback_accept = len(selected) + (sorted_indices.shape[-1] - rank) <= actual_topk
                if bool(accept.all()) or fallback_accept:
                    selected.append(candidate)
            if len(selected) >= actual_topk:
                break
        while len(selected) < actual_topk:
            selected.append(sorted_indices[..., len(selected)])
        return torch.stack(selected, dim=-1)
    elif mode == "tail_query_relevance":
        if tail_query_states is None or tail_query_states.shape[-2] == 0:
            selector_scores = block_keys.norm(dim=-1)
        else:
            selector_scores = torch.einsum(
                "bhqd,bhkd->bhqk",
                tail_query_states,
                block_keys,
            ).mean(dim=-2)
    elif mode == "random_topk":
        selector_scores = deterministic_random_scores(
            shape=block_keys.shape[:3],
            block_start=block_start,
            seed=seed,
            layer_idx=layer_idx,
            device=block_keys.device,
        )
    else:
        raise ValueError(f"Unsupported representative selection mode: {mode}")

    return selector_scores.topk(actual_topk, dim=-1).indices


def resolve_query_topr(q_window: int, configured_topr: int | None) -> int:
    if configured_topr is not None:
        return max(1, min(configured_topr, q_window))
    return max(1, math.ceil(q_window / 4))
