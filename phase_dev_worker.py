import torch


class PhaseDevWorker:
    def enable_phase_runtime(self):
        self.model_runner._phase_runtime_ready = True
        return self.phase_memory_stats()

    def phase_memory_stats(self):
        r = self.model_runner
        torch.cuda.synchronize()
        c = r.extensible_kv_cache
        return dict(
            allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),
            free=torch.cuda.mem_get_info()[0],
            cache_bytes=c.physical_bytes,
            blocks=c.num_committed_blocks,
            capacity=r.kv_cache_config.num_blocks,
            last_budget=getattr(self, "_phase_last_budget", None),
            vision_devices=sorted({str(p.device) for p in r.model.visual.parameters()}),
            vision_parameter_bytes=sum(
                p.numel() * p.element_size() for p in r.model.visual.parameters()
            ),
        )

    def phase_admit(self, request):
        import copy

        from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
        from vllm.v1.core.kv_cache_planning import (
            _max_memory_usage_bytes_from_groups,
            _pool_bytes_per_block_from_builder,
        )
        from vllm.v1.worker.gpu.phase_memory import release

        r = self.model_runner
        r._phase_runtime_ready = True
        r._phase_budget_blocks = None
        release(r)
        r.model_state.encoder_runner.phase_before_encode = None
        r.model_state.encoder_runner.phase_after_encode = None
        cache = r.extensible_kv_cache
        outputs = r.encoder_cache.encoder_outputs
        outputs.clear()
        try:
            features = request.mm_features or []
            inputs = [(f.modality, f.data) for f in features if f.data is not None]
            if inputs:
                encoded = r.model_state.encoder_runner.execute_mm_encoder(inputs)
                for feature, value in zip(
                    [f for f in features if f.data is not None], encoded
                ):
                    outputs[feature.identifier] = value
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free = torch.cuda.mem_get_info()[0]
            reserve = 1024 << 20
            available = max(0, free - reserve)
            max_blocks = min(
                cache.blocks_within(available), r.kv_cache_config.num_blocks
            )
            cfg = copy.copy(r.vllm_config)
            cfg.model_config = copy.copy(cfg.model_config)
            builder = get_kv_cache_config_builder(cfg)
            groups = r.kv_cache_config.kv_cache_groups
            bytes_per_block = _pool_bytes_per_block_from_builder(builder, cfg, groups)

            def blocks_for(length):
                cfg.model_config.max_model_len = length
                required = _max_memory_usage_bytes_from_groups(builder, cfg, groups)
                return 1 + (required + bytes_per_block - 1) // bytes_per_block

            lo, hi = 0, r.max_model_len
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if blocks_for(mid) <= max_blocks:
                    lo = mid
                else:
                    hi = mid - 1
            prompt_tokens = (
                len(request.prompt_token_ids)
                if request.prompt_token_ids is not None
                else request.prompt_embeds.shape[0]
            )
            total = prompt_tokens + request.sampling_params.max_tokens
            budget = dict(
                accepted=total <= lo,
                prompt_tokens=prompt_tokens,
                output_tokens=request.sampling_params.max_tokens,
                capacity_tokens=lo,
                budget_blocks=blocks_for(total),
                free_bytes=free,
                reserve_bytes=reserve,
            )
            if not budget["accepted"]:
                budget["reason"] = (
                    f"Phase capacity {lo} tokens; input {prompt_tokens}, requested "
                    f"output {request.sampling_params.max_tokens}. Reduce input/output."
                )
                outputs.clear()
                return budget
            cache.commit(budget["budget_blocks"])
            r._phase_budget_blocks = budget["budget_blocks"]
            r._phase_required_blocks = budget["budget_blocks"]
            self._phase_last_budget = budget
            return budget
        except BaseException:
            outputs.clear()
            release(r)
            raise

    def phase_end_request(self):
        from vllm.v1.worker.gpu.phase_memory import release

        r = self.model_runner
        release(r)
        r._phase_budget_blocks = None
        r.encoder_cache.encoder_outputs.clear()
        return True

    def phase_test_pressure(self, num_bytes="0"):
        if getattr(self.model_runner, "_phase_budget_blocks", None) is not None:
            raise RuntimeError("Cannot alter pressure during an admitted request")
        self._phase_pressure = None
        torch.cuda.empty_cache()
        size = int(num_bytes)
        if size:
            self._phase_pressure = torch.empty(size, device="cuda", dtype=torch.uint8)
        torch.cuda.synchronize()
        return self.phase_memory_stats()
