# Single-request phase memory experiment

This checkout contains an opt-in Qwen3.5/Qwen3.8 multimodal experiment, not a
production serving feature. Baseline: e336407a3d. The GGUF plugin has a separate
editable development copy at /home/eureka/vllm-gguf-dev.

## Serving policy

- Hard context limit 16384 tokens, eager, one frontend/rank/active request, n=1.
- FP8 attention KV, FP32 GDN state, prefix caching and MM processor caching off.
- Incoming requests serialize through an AsyncLLM lock. Expanded prompt tokens
  include images, templates and history. Explicit output budgets are never
  silently truncated at the hard limit.
- Before language execution, worker encodes all images, retaining outputs, and
  moves vision parameters back to CPU. This is real EncoderRunner integration.
- A virtual KV reservation and block-table capacity are established at startup.
  Startup no longer shrinks that virtual capacity to measured physical capacity.
- Admission computes mixed-group page requirements with the existing cache
  planner, including the null block and physical allocation granularity. It
  measures free memory after encoding and empty_cache, leaving 1 GiB headroom.
- The entire input-plus-maximum-output physical budget is committed before
  admission. Core limits the free logical block queue to the same low-numbered
  prefix. The budget stays fixed until completion; execution does not grow it.
- Completion/cancellation waits for scheduler quiescence, releases physical KV
  and GDN backing, clears encoder outputs, and restores the virtual free pool.
  Virtual addresses remain stable. No prefix or live-session state survives.

Flags: VLLM_EXPERIMENTAL_PHASE_MEMORY=1 and
VLLM_EXPERIMENTAL_PHASE_ADMISSION=1. launch-dynamic-fp8.sh sets both and supplies
the PhaseDevWorker extension. dev-env.sh selects the Conda CUDA compiler.
Admission activation is automatic after warmup; the earlier manual activation
RPC is no longer needed. The internal phase_begin utility carries trusted
frontend requests, never external serialized payloads.

The original controller.py/gpu_smoke.py remain isolated lifecycle prototypes;
serving integration is in AsyncLLM, EngineCore, phase_memory.py, EncoderRunner,
ExtensibleKVCache and phase_dev_worker.py.

## Verified on RTX 5080, 2026-11-07

- 512, 8192 and 15872 input tokens, each with 128 output-token budget: correct
  requested answer; 5, 10 and 14 blocks reserved respectively. Physical KV is
  zero after completion. Near-16K test took about 22.5 s including prefill;
  this is not a sustained decode throughput benchmark.
- Input 16300 + output 128: HTTP 400, explicit hard-limit rejection.
- Holding 1 GiB extra CUDA allocation: measured capacity drops from 16384 to
  4800 tokens. Large budget rejected with HTTP 400; small requests still work.
  Releasing pressure restores acceptance of the same large budget.
- Text/blue-image/text/red-image alternation and repeated 768x768 green image:
  expected answers. Vision parameters on CPU after encoding: 921460192 bytes.
- Two snapshots during streaming show identical budget and committed pages;
  a concurrent arrival waits and then returns the correct answer.
- Client stream disconnect: physical cache reclaimed and later requests work.

Run acceptance_16k.py, acceptance_pressure.py, acceptance_lock.py and
acceptance.py using .venv/bin/python from the configured checkout. Pressure
checks require localhost development RPC (VLLM_SERVER_DEV_MODE=1); leave it
unexposed. Raw records also live in /home/eureka/phase16k-acceptance.json and
/home/eureka/phase-pressure-acceptance.json.

## Limits

The 1 GiB reserve is provisional for the tested single-image/256-token chunk
configuration. No long-context quality benchmark, MTP, PD, multi-GPU, streaming
input, multiple frontend workers or recovery from CUDA faults is supported.
An external process can consume memory after admission; a budget is not GPU
isolation. Unexpected cleanup failure is an error requiring worker recovery.
Images are reprocessed on each request because MM processor caching is off.
