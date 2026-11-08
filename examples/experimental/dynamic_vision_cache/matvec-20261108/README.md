# GGUF plugin reproducibility

Upstream: https://github.com/vllm-project/vllm-gguf-plugin.git

Base commit: `e454affe0ca23c9dc8754ec3033270cb30622ed5`. Apply `gguf-plugin.patch` at that commit using `git apply --unidiff-zero`, then build/install the editable plugin in the same vLLM Torch/CUDA environment. This is the complete active plugin diff, including prior prepack/dequant and Qwen compatibility changes as well as matvec experiments.

The launcher enables mode 3 only; modes 1/2 are experimental and have no stable service gain. Set VLLM_GGUF_MATVEC_MODE=0 to revert this matvec optimization.

See ../matvec-report-zh.md and ../matvec_tuning.md for measurements and validation. JSON files are recorded test results, not live configuration. Worker validation entry points are in phase_dev_worker.py (matvec_nofill_check, matvec_sweep, matvec_cold_cache_check).

Commit validation: Ruff check/format, shell syntax and staged whitespace checks passed; prior GPU build, bitwise validation and service regressions passed. Full pre-commit initialization stalled while fetching crate-ci/typos, so external hooks were not completed.
