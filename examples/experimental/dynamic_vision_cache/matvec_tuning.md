# RTX 5080 matvec experiments

The separate editable GGUF plugin at /home/eureka/vllm-gguf-matvec-nofill
contains the experiment. The vLLM source itself does not contain this CUDA
extension. Preserve the plugin source patch and build it against the same
Torch/CUDA environment when reproducing.

VLLM_GGUF_MATVEC_MODE controls the original C++ matvec operator:
- 0: reference, default one output row per block and zero-initialized output.
- 1: exploratory shape-specific C++ launch configuration.
- 2: exploratory launch configuration plus packed IQ3/IQ4 loads.
- 3: reference math and launch geometry, but uninitialized output allocation;
  the existing kernels fully overwrite valid output elements.

Only mode 3 demonstrated stable service throughput gains in alternating
A/B tests. Modes 1/2 remain experimental and are not selected by the launcher.
Python per-op lookup was rejected because it erased microbenchmark gains.
Startup selection uses the environment; the localhost worker test setter changes
it only while no request is admitted. No tensors or extra weight copies are
retained by mode 3. The prepack and dequant-nozero switches are independent.

Coverage: 14 quantization types, 54 actual linear shapes including LM head,
excluding token embedding. Mode 3 passed 380 bitwise checks covering BF16,
FP16/FP32 representatives, vector batches 1/2/4, 67-row tails and NaN-poisoned
allocation reuse. The baseline block/reading exploration and cold-cache check
are diagnostic measurements, not proof of DRAM bandwidth saturation.

Raw experiment records: /home/eureka/matvec-20261108. End-to-end tests use the
same short text, 15020-token synthetic long input and 768px image as earlier
experiments. Do not use profiler-enabled throughput as normal service speed.
