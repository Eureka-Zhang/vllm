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

    def profiling_memory(self, reset="0"):
        torch.cuda.synchronize()
        if reset == "1":
            torch.cuda.reset_peak_memory_stats()
        return dict(
            allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
            free=torch.cuda.mem_get_info()[0],
            budget=getattr(self, "_phase_last_budget", None),
        )

    def install_profile_scopes(self):
        from functools import wraps

        def instrument(obj, name, label):
            original = getattr(obj, name)

            @wraps(original)
            def wrapped(*args, **kwargs):
                with torch.profiler.record_function(label):
                    return original(*args, **kwargs)

            setattr(obj, name, wrapped)

        instrument(self, "phase_admit", "phase/admission_encode_and_reserve")
        instrument(self, "phase_end_request", "phase/reclaim")
        instrument(self.model_runner.model.visual, "forward", "phase/vision_forward")
        instrument(self.model_runner.model, "embed_multimodal", "phase/vision_features")
        return True

    def gguf_layout_audit(self):
        rows = []
        for name, layer in self.model_runner.model.named_modules():
            weight = getattr(layer, "weight", None)
            kinds = getattr(
                getattr(layer, "weight_type", None), "shard_weight_type", {}
            )
            ids = getattr(weight, "shard_id", []) if weight is not None else []
            if (
                not ids
                or len(
                    {
                        kinds.get(i, getattr(layer.weight_type, "weight_type", 0))
                        for i in ids
                    }
                )
                <= 1
            ):
                continue
            shards = []
            names = getattr(layer, "_gguf_prepacked_names", {})
            for idx in ids:
                start, end, width = weight.shard_offset_map[idx]
                part = (
                    getattr(layer, names[idx]) if names else weight[start:end, :width]
                )
                shards.append(
                    dict(
                        id=idx,
                        shape=list(part.shape),
                        stride=list(part.stride()),
                        contiguous=part.is_contiguous(),
                        bytes=part.numel() * part.element_size(),
                        quant_type=int(kinds[idx]),
                    )
                )
            rows.append(
                dict(
                    name=name,
                    weight_shape=list(weight.shape),
                    weight_bytes=weight.numel() * weight.element_size(),
                    shards=shards,
                )
            )
        return rows

    def gguf_prepack_equivalence(self):
        import gguf

        tested = []
        torch.manual_seed(0)
        for name, layer in self.model_runner.model.named_modules():
            names = getattr(layer, "_gguf_prepacked_names", {})
            if not names:
                continue
            weight = layer.weight
            original_data = weight.data
            padded = torch.zeros(
                layer._gguf_original_padded_shape,
                dtype=weight.dtype,
                device=weight.device,
            )
            for idx, buffer_name in names.items():
                start, end, width = weight.shard_offset_map[idx]
                padded[start:end, :width].copy_(getattr(layer, buffer_name))
            idx = next(iter(names))
            block_size, type_size = gguf.GGML_QUANT_SIZES[
                layer.weight_type.shard_weight_type[idx]
            ]
            width = getattr(layer, names[idx]).shape[1] // type_size * block_size
            for batch in [1, 16]:
                x = torch.randn(
                    batch, width, dtype=torch.bfloat16, device=weight.device
                )
                bias = getattr(layer, "bias", None)
                try:
                    weight.data = padded
                    layer._gguf_prepacked_names = {}
                    expected = layer.quant_method.apply(layer, x, bias)
                finally:
                    weight.data = original_data
                    layer._gguf_prepacked_names = names
                actual = layer.quant_method.apply(layer, x, bias)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            tested.append(name)
            del padded, expected, actual, x
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return dict(
            layers=len(tested), cases=len(tested) * 2, bitwise_equal=True, names=tested
        )

    def gguf_dequant_nozero_check(self):
        import gguf
        from vllm_gguf_plugin import ops

        representatives = {}
        for name, layer in self.model_runner.model.named_modules():
            weight = getattr(layer, "weight", None)
            meta = getattr(layer, "weight_type", None)
            if weight is None or meta is None:
                continue
            packed = getattr(layer, "_gguf_prepacked_names", {})
            candidates = (
                [
                    (getattr(layer, key), int(meta.shard_weight_type[idx]))
                    for idx, key in packed.items()
                ]
                if packed
                else [(weight, int(getattr(meta, "weight_type", 0)))]
            )
            for w, kind in candidates:
                if (
                    kind in representatives
                    or kind not in ops._CUDA_GEMV_QUANT_TYPES
                    or w.ndim != 2
                ):
                    continue
                bs, ts = gguf.GGML_QUANT_SIZES[kind]
                n = w.shape[1] // ts * bs
                if w.shape[0] * n > 100_000_000:
                    continue
                representatives[kind] = (name, w, n)
        rows = []
        for kind, (name, w, n) in representatives.items():
            for dtype in [torch.bfloat16, torch.float16, torch.float32]:
                expected = torch.ops._C_gguf.ggml_dequantize(
                    w, kind, w.shape[0], n, dtype
                )
                poison = torch.full(
                    expected.shape, float("nan"), device=w.device, dtype=dtype
                )
                torch.cuda.synchronize()
                del poison
                actual = torch.ops._C_gguf.ggml_dequantize_empty(
                    w, kind, w.shape[0], n, dtype
                )
                assert torch.equal(
                    actual.view(torch.uint8), expected.view(torch.uint8)
                ), (kind, dtype)
                del actual, expected
            timings = {}
            for label, fn in [
                ("zeros", torch.ops._C_gguf.ggml_dequantize),
                ("empty", torch.ops._C_gguf.ggml_dequantize_empty),
            ]:
                for _ in range(2):
                    fn(w, kind, w.shape[0], n, torch.bfloat16)
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(10):
                    fn(w, kind, w.shape[0], n, torch.bfloat16)
                b.record()
                b.synchronize()
                timings[label] = a.elapsed_time(b) / 10
            rows.append(
                dict(
                    type=kind,
                    layer=name,
                    shape=[w.shape[0], n],
                    bitwise_equal=True,
                    old_ms=timings["zeros"],
                    new_ms=timings["empty"],
                    speedup=timings["zeros"] / timings["empty"],
                )
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return rows

    def matvec_set_tuning(self, config="{}"):
        import json

        from vllm_gguf_plugin import ops

        if getattr(self.model_runner, "_phase_budget_blocks", None) is not None:
            raise RuntimeError("Tuning can change only between requests")
        torch.cuda.synchronize()
        ops.set_matvec_tuning(json.loads(config).get("rows", []))
        return len(ops._MATVEC_TUNING)

    def matvec_sweep(self):
        import random
        import statistics

        import gguf
        from vllm_gguf_plugin import ops
        from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod

        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        if getattr(self.model_runner, "_phase_budget_blocks", None) is not None:
            raise RuntimeError("Cannot benchmark during generation")
        torch.manual_seed(123)
        samples = {}
        for name, layer in self.model_runner.model.named_modules():
            if not isinstance(
                getattr(layer, "quant_method", None), GGUFLinearMethod
            ) or (
                isinstance(layer, VocabParallelEmbedding)
                and not isinstance(layer, ParallelLMHead)
            ):
                continue
            weight = layer.weight
            meta = layer.weight_type
            packed = getattr(layer, "_gguf_prepacked_names", {})
            candidates = (
                [
                    (getattr(layer, key), int(meta.shard_weight_type[idx]))
                    for idx, key in packed.items()
                ]
                if packed
                else [(weight, int(meta.weight_type))]
            )
            for w, kind in candidates:
                if kind not in ops._CUDA_GEMV_QUANT_TYPES or w.ndim != 2:
                    continue
                bs, ts = gguf.GGML_QUANT_SIZES[kind]
                col = w.shape[1] // ts * bs
                key = (kind, w.shape[0], col)
                if key in samples:
                    samples[key][2] += 1
                else:
                    samples[key] = [name, w, 1]
        result = []
        checked = 0
        for (kind, row, col), (name, w, count) in samples.items():
            for batch in [1, 2, 4]:
                x = torch.randn(batch, col, device=w.device, dtype=torch.bfloat16)
                expected = torch.ops._C_gguf.ggml_mul_mat_vec_a8(w, x, kind, row)
                for launch in [1, 2, 4, 8, 16]:
                    actual = torch.ops._C_gguf.ggml_mul_mat_vec_a8_tuned(
                        w, x, kind, row, launch
                    )
                    assert torch.equal(
                        actual.view(torch.uint8), expected.view(torch.uint8)
                    ), (kind, row, col, batch, launch)
                    checked += 1
            # Exercise partial last blocks, absent from most aligned model shapes.
            tail = min(row, 67)
            x = torch.randn(1, col, device=w.device, dtype=torch.bfloat16)
            expected = torch.ops._C_gguf.ggml_mul_mat_vec_a8(w[:tail], x, kind, tail)
            for launch in [2, 4, 8, 16]:
                actual = torch.ops._C_gguf.ggml_mul_mat_vec_a8_tuned(
                    w[:tail], x, kind, tail, launch
                )
                assert torch.equal(
                    actual.view(torch.uint8), expected.view(torch.uint8)
                ), ("tail", kind, launch)
                checked += 1
            timings = {v: [] for v in [1, 2, 4, 8, 16]}

            def run(launch, w=w, x=x, kind=kind, row=row):
                if launch == 1:
                    return torch.ops._C_gguf.ggml_mul_mat_vec_a8(w, x, kind, row)
                return torch.ops._C_gguf.ggml_mul_mat_vec_a8_tuned(
                    w, x, kind, row, launch
                )

            for launch in timings:
                for _ in range(3):
                    run(launch)
            rng = random.Random(123)
            for _ in range(5):
                order = list(timings)
                rng.shuffle(order)
                for launch in order:
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(20):
                        run(launch)
                    b.record()
                    b.synchronize()
                    timings[launch].append(a.elapsed_time(b) / 20)
            med = {k: statistics.median(v) for k, v in timings.items()}
            best = min(med, key=med.get)
            # Independent confirmation pass; refuse noisy apparent wins.
            confirm = {k: [] for k in {1, best}}
            for _ in range(5):
                for launch in confirm:
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(20):
                        run(launch)
                    b.record()
                    b.synchronize()
                    confirm[launch].append(a.elapsed_time(b) / 20)
            verify = {k: statistics.median(v) for k, v in confirm.items()}
            chosen = (
                best
                if med[1] / med[best] >= 1.05 and verify[1] / verify[best] >= 1.05
                else 1
            )
            result.append(
                dict(
                    type=kind,
                    quant=gguf.GGMLQuantizationType(kind).name,
                    row=row,
                    col=col,
                    layer=name,
                    frequency=count,
                    weight_bytes=w.numel() * w.element_size(),
                    median_ms=med,
                    confirmation_ms=verify,
                    rows_per_block=chosen,
                    speedup=verify[1] / verify.get(chosen, verify[1]),
                )
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return dict(
            device=torch.cuda.get_device_name(),
            dtype="bfloat16",
            checked=checked,
            shapes=len(samples),
            rows=result,
        )

    def matvec_set_mode(self, mode="0"):
        from vllm_gguf_plugin import ops

        if getattr(self.model_runner, "_phase_budget_blocks", None) is not None:
            raise RuntimeError("Mode can change only between requests")
        value = int(mode)
        if value not in (0, 1, 2, 3):
            raise ValueError("Invalid matvec mode")
        torch.cuda.synchronize()
        ops.set_matvec_tuning([])
        ops._MATVEC_MODE = value
        if hasattr(torch.ops._C_gguf, "ggml_set_matvec_mode"):
            torch.ops._C_gguf.ggml_set_matvec_mode(value)
        return value

    def matvec_readopt_check(self):
        import statistics

        import gguf
        from vllm_gguf_plugin import ops
        from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod

        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        samples = {}
        checks = 0
        torch.manual_seed(123)
        for name, layer in self.model_runner.model.named_modules():
            if not isinstance(
                getattr(layer, "quant_method", None), GGUFLinearMethod
            ) or (
                isinstance(layer, VocabParallelEmbedding)
                and not isinstance(layer, ParallelLMHead)
            ):
                continue
            w = layer.weight
            meta = layer.weight_type
            packed = getattr(layer, "_gguf_prepacked_names", {})
            candidates = (
                [
                    (getattr(layer, key), int(meta.shard_weight_type[idx]))
                    for idx, key in packed.items()
                ]
                if packed
                else [(w, int(meta.weight_type))]
            )
            for w, kind in candidates:
                if kind not in ops._CUDA_GEMV_QUANT_TYPES or w.ndim != 2:
                    continue
                bs, ts = gguf.GGML_QUANT_SIZES[kind]
                col = w.shape[1] // ts * bs
                key = (kind, w.shape[0], col)
                if key in samples:
                    samples[key][2] += 1
                else:
                    samples[key] = [name, w, 1]
        rows = []
        for (kind, row, col), (name, w, freq) in samples.items():
            for batch in [1, 2, 4]:
                x = torch.randn(batch, col, device=w.device, dtype=torch.bfloat16)
                expected = torch.ops._C_gguf.ggml_mul_mat_vec_a8(w, x, kind, row)
                for launch in [1, 2, 4, 8, 16]:
                    actual = torch.ops._C_gguf.ggml_mul_mat_vec_a8_readopt(
                        w, x, kind, row, launch
                    )
                    assert torch.equal(
                        actual.view(torch.uint8), expected.view(torch.uint8)
                    ), (kind, row, col, batch, launch)
                    checks += 1
            x = torch.randn(1, col, device=w.device, dtype=torch.bfloat16)
            for mode in [1, 2]:
                expected = torch.ops._C_gguf.ggml_mul_mat_vec_a8(w, x, kind, row)
                actual = torch.ops._C_gguf.ggml_mul_mat_vec_a8_auto(
                    w, x, kind, row, mode
                )
                assert torch.equal(
                    actual.view(torch.uint8), expected.view(torch.uint8)
                ), ("auto", kind, row, col, mode)
                checks += 1
            tail = min(row, 67)
            for launch in [1, 2, 4, 8, 16]:
                expected = torch.ops._C_gguf.ggml_mul_mat_vec_a8(
                    w[:tail], x, kind, tail
                )
                actual = torch.ops._C_gguf.ggml_mul_mat_vec_a8_readopt(
                    w[:tail], x, kind, tail, launch
                )
                assert torch.equal(
                    actual.view(torch.uint8), expected.view(torch.uint8)
                ), ("tail", kind, launch)
                checks += 1
            times = {k: [] for k in [0, 1, 2]}
            for repeat in range(7):
                order = [0, 1, 2] if repeat % 2 == 0 else [2, 1, 0]
                for mode in order:
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(20):
                        if mode == 0:
                            torch.ops._C_gguf.ggml_mul_mat_vec_a8(w, x, kind, row)
                        else:
                            torch.ops._C_gguf.ggml_mul_mat_vec_a8_auto(
                                w, x, kind, row, mode
                            )
                    b.record()
                    b.synchronize()
                    times[mode].append(a.elapsed_time(b) / 20)
            rows.append(
                dict(
                    type=kind,
                    quant=gguf.GGMLQuantizationType(kind).name,
                    row=row,
                    col=col,
                    frequency=freq,
                    layer=name,
                    median_ms={k: statistics.median(v) for k, v in times.items()},
                )
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return dict(shapes=len(samples), checks=checks, rows=rows)

    def matvec_cold_cache_check(self):
        import statistics

        import gguf
        from vllm_gguf_plugin import ops
        from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod

        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        if getattr(self.model_runner, "_phase_budget_blocks", None) is not None:
            raise RuntimeError("Cannot benchmark during a request")
        samples = {}
        for name, layer in self.model_runner.model.named_modules():
            if not isinstance(
                getattr(layer, "quant_method", None), GGUFLinearMethod
            ) or (
                isinstance(layer, VocabParallelEmbedding)
                and not isinstance(layer, ParallelLMHead)
            ):
                continue
            meta = layer.weight_type
            packed = getattr(layer, "_gguf_prepacked_names", {})
            candidates = (
                [
                    (getattr(layer, key), int(meta.shard_weight_type[idx]))
                    for idx, key in packed.items()
                ]
                if packed
                else [(layer.weight, int(meta.weight_type))]
            )
            for w, kind in candidates:
                if kind not in ops._CUDA_GEMV_QUANT_TYPES or w.ndim != 2:
                    continue
                bs, ts = gguf.GGML_QUANT_SIZES[kind]
                col = w.shape[1] // ts * bs
                key = (kind, w.shape[0], col)
                if key in samples:
                    samples[key][1] += 1
                else:
                    samples[key] = [w, 1]
        original_mode = ops._MATVEC_MODE
        rows = []
        flush = torch.empty(256 << 20, device="cuda", dtype=torch.uint8)
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        b.record()
        b.synchronize()
        try:
            for (kind, row, col), (w, freq) in samples.items():
                x = torch.randn(1, col, device=w.device, dtype=torch.bfloat16)
                times = {0: [], 2: []}
                for repeat in range(7):
                    for mode in [0, 2] if repeat % 2 == 0 else [2, 0]:
                        torch.ops._C_gguf.ggml_set_matvec_mode(mode)
                        flush.zero_()
                        torch.cuda.synchronize()
                        a.record()
                        out = torch.ops._C_gguf.ggml_mul_mat_vec_a8(w, x, kind, row)
                        b.record()
                        b.synchronize()
                        times[mode].append(a.elapsed_time(b))
                rows.append(
                    dict(
                        type=kind,
                        quant=gguf.GGMLQuantizationType(kind).name,
                        row=row,
                        col=col,
                        frequency=freq,
                        median_ms={k: statistics.median(v) for k, v in times.items()},
                    )
                )
        finally:
            torch.ops._C_gguf.ggml_set_matvec_mode(original_mode)
            del flush, out, x
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return dict(flush_bytes=256 << 20, shapes=len(samples), rows=rows)

    def matvec_nofill_check(self):
        import gguf
        from vllm_gguf_plugin import ops
        from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod

        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        if getattr(self.model_runner, "_phase_budget_blocks", None) is not None:
            raise RuntimeError("Cannot validate during a request")
        samples = {}
        for name, layer in self.model_runner.model.named_modules():
            if not isinstance(
                getattr(layer, "quant_method", None), GGUFLinearMethod
            ) or (
                isinstance(layer, VocabParallelEmbedding)
                and not isinstance(layer, ParallelLMHead)
            ):
                continue
            meta = layer.weight_type
            packed = getattr(layer, "_gguf_prepacked_names", {})
            candidates = (
                [
                    (getattr(layer, key), int(meta.shard_weight_type[idx]))
                    for idx, key in packed.items()
                ]
                if packed
                else [(layer.weight, int(meta.weight_type))]
            )
            for w, kind in candidates:
                if kind not in ops._CUDA_GEMV_QUANT_TYPES or w.ndim != 2:
                    continue
                bs, ts = gguf.GGML_QUANT_SIZES[kind]
                col = w.shape[1] // ts * bs
                samples.setdefault((kind, w.shape[0], col), w)
        checked = 0
        extra = set()
        try:
            for (kind, row, col), w in samples.items():
                for dtype in [torch.bfloat16, torch.float16, torch.float32]:
                    if dtype != torch.bfloat16 and (kind, dtype) in extra:
                        continue
                    extra.add((kind, dtype))
                    for batch in [1, 2, 4] if dtype == torch.bfloat16 else [1]:
                        x = torch.randn(batch, col, device=w.device, dtype=dtype)
                        for count in [row, min(row, 67)]:
                            torch.ops._C_gguf.ggml_set_matvec_mode(0)
                            expected = torch.ops._C_gguf.ggml_mul_mat_vec_a8(
                                w[:count], x, kind, count
                            )
                            poison = torch.full(
                                expected.shape,
                                float("nan"),
                                device=w.device,
                                dtype=dtype,
                            )
                            torch.cuda.synchronize()
                            del poison
                            torch.ops._C_gguf.ggml_set_matvec_mode(3)
                            actual = torch.ops._C_gguf.ggml_mul_mat_vec_a8(
                                w[:count], x, kind, count
                            )
                            assert torch.equal(
                                actual.view(torch.uint8), expected.view(torch.uint8)
                            ), (kind, row, col, dtype, batch, count)
                            checked += 1
            del expected, actual, x
        finally:
            torch.ops._C_gguf.ggml_set_matvec_mode(0)
            ops._MATVEC_MODE = 0
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return dict(shapes=len(samples), cases=checked, bitwise_equal=True)
