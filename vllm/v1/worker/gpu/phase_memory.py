# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in single-request physical KV lifecycle experiment."""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def release(runner):
    cache = runner.extensible_kv_cache
    torch.cuda.synchronize(runner.device)
    cache.release_physical()
    cache.num_committed_blocks = 0
    torch.cuda.empty_cache()
    logger.info(
        "Phase memory: KV/GDN released; free=%d",
        torch.cuda.mem_get_info(runner.device)[0],
    )


def commit(runner):
    cache = runner.extensible_kv_cache
    blocks = runner._phase_required_blocks
    needed = cache.physical_bytes_for(blocks) - cache.physical_bytes
    free = torch.cuda.mem_get_info(runner.device)[0]
    reserve = 256 << 20
    if needed > max(0, free - reserve):
        raise RuntimeError(
            "Phase cache capacity exceeded: insufficient workspace headroom"
        )
    if blocks > runner.kv_cache_config.num_blocks:
        raise RuntimeError("Request references blocks beyond scheduler capacity")
    cache.commit(blocks)
    logger.info(
        "Phase memory: committed=%d bytes=%d free=%d",
        blocks,
        cache.physical_bytes,
        torch.cuda.mem_get_info(runner.device)[0],
    )


def prepare_phase_step(runner, output):
    if not getattr(runner, "_phase_runtime_ready", False):
        return
    if (
        runner.scheduler_config.max_num_seqs != 1
        or runner.cache_config.enable_prefix_caching
        or runner.speculator is not None
        or runner.parallel_config.world_size != 1
    ):
        raise RuntimeError(
            "Phase memory requires single request/rank, no prefix/speculation"
        )
    if runner.extensible_kv_cache is None:
        raise RuntimeError("Phase memory requires extensible KV cache")
    if output.preempted_req_ids:
        raise RuntimeError("Phase memory does not support preemption")
    admission = os.getenv("VLLM_EXPERIMENTAL_PHASE_ADMISSION") == "1"
    if admission:
        if (
            output.total_num_scheduled_tokens
            and getattr(runner, "_phase_budget_blocks", None) is None
        ):
            raise RuntimeError("Request lacks phase admission budget")
        return
    if output.finished_req_ids:
        release(runner)
        runner._phase_required_blocks = 1
    fresh = bool(output.scheduled_new_reqs)
    runner._phase_fresh = fresh
    if fresh:
        release(runner)
        runner._phase_required_blocks = 1
    blocks = getattr(runner, "_phase_required_blocks", 1)
    groups = [r.block_ids for r in output.scheduled_new_reqs]
    groups += [g for g in output.scheduled_cached_reqs.new_block_ids if g]
    for group in groups:
        for ids in group:
            if ids:
                blocks = max(blocks, max(ids) + 1)
    runner._phase_required_blocks = blocks
    encoder = getattr(runner.model_state, "encoder_runner", None)
    if encoder is not None:
        encoder.phase_before_encode = lambda: (
            release(runner) if runner._phase_fresh else None
        )
        encoder.phase_after_encode = lambda: commit(runner)
    if output.total_num_scheduled_tokens:
        commit(runner)
