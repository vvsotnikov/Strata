# Experimental native Hy4-flash text engine

Upstream Strata at `3142b66b96e9c0e775e30f620b11b191a1427c2a` has no Hy4 engine.
This opt-in target reads the AngelSlim Hy4-flash GGUF directly. It does not start
llama.cpp or link libllama. The default Qwen engine and its model catalog stay as before.

The first port implements text inference: iHC residual streams, latent MoE with
all 16 selected experts, gated MLA with sinks, sliding-window attention, pooled
sparse indexing, and the FP8 Engram table. The HTTP server uses the GGUF's own
namespaced chat template, reasoning markers and tool arguments.

This is a correctness baseline, not the optimized Qwen execution engine applied to
Hy4. Experts, per-head attention and KV state run on the CPU. Optional CUDA offload
uses Strata's existing MMVQ kernels for dense quantized projections. Dense weights
are admitted up to explicit device budgets; remaining matrices stay on the CPU.
There is no expert streaming, CUDA graph capture or tensor parallelism in this target.

## Build and run

From the repository root, on Linux:

```sh
cmake -S engine -B build-hy4 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_NATIVE_EXPERTS=OFF \
  -DSTRATA_BUILD_HY4=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-hy4 --target strata-hy4 hy4_math_test hy4_reader_test -j4
ctest --test-dir build-hy4 -R '^hy4_' --output-on-failure
```

The independent full-forward fixture needs NumPy in CMake's selected Python.
Without NumPy, CMake states that this test was not registered. Select an environment
with `-DPython3_EXECUTABLE=/path/to/python` to run it.

For CUDA dense offload, add `-DSTRATA_HY4_CUDA=ON` and explicitly choose the card's
architecture, for example `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX A6000. Keep
`STRATA_ENABLE_CUDA=OFF` to avoid building the separate Qwen engine. The Hy4 target
builds the existing native MMVQ/IQ kernels it needs.

Download both Q4_K_M shards from
[AngelSlim/Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF).
Keep their original names in the same directory. The two files total
168,347,133,632 bytes (156.79 GiB). They are mapped read-only; mapped bytes are
not a claim that all pages are resident or locked in RAM. Engram alone occupies
about 59.01 GiB on disk and stays host mapped, including with CUDA enabled.

```sh
python terminal/tools/hy4/prepare.py \
  --model /models/Hy4-flash-Q4_K_M-00001-of-00002.gguf \
  --engine build-hy4/src/hy4/strata-hy4 --output run-hy4 \
  --context 8192 --threads 4
PYTHONPATH=terminal:terminal/tools python terminal/serve/server.py \
  --engine strata --config run-hy4/serve.json --host 127.0.0.1 --port 8095
```

For CUDA, pass `--cuda-devices` and `--cuda-cache-mib` to the prepare command,
for example `--cuda-devices 0 --cuda-cache-mib 4096`. The latter is a dense weight
budget in MiB per selected device, excluding scratch and the CUDA context. The
engine checks for at least 256 MiB of free memory beyond each budget at startup.
Budgets do not reserve VRAM against other processes; do not overcommit a shared card.

Prefer GPU UUIDs in `CUDA_VISIBLE_DEVICES`. Numeric ordinals in that variable use
CUDA's ordering, which can differ from `nvidia-smi`. `--cuda-devices 0` means the
first device visible to the engine, not necessarily physical GPU 0.

The config defaults to `max` reasoning, the highest level accepted by this GGUF's
template. Requests can select `low`, `high`, `max` or `none` (`no_think`). The model
card recommends temperature 0.9 and top_p 1.0; no top-k limit is imposed by default.

## Validation and limits

The implementation was checked against the independent Hy4 llama.cpp port at
`b7c3f5da30fe1c44f7d015dc051ef0f373c708ec`, linked from
[upstream PR 32517](https://github.com/ggml-org/llama.cpp/pull/32517).
That separate process is a test oracle, not a runtime dependency.

Measured on Linux x86-64 with GCC 13.3 and RTX A6000 (sm_86):

- The independent NumPy fixture crosses its sliding-window wrap and sparse top-k
  threshold. Its complete hidden-state and logit comparisons pass at 1e-4 tolerance.
- Actual Hy4 token IDs match the oracle for 17 strings, including multilingual text,
  special tokens and rendered reasoning modes.
- On 35 forced tokens from a real chat prompt, 33 next-token choices match the CPU
  oracle, including the final answer token. The logits are **not identical**:
  mean cosine similarity is about 0.979 and the minimum is about 0.834. This is not
  an exact numerical parity claim or a broad quality evaluation. Replaying the first
  dense FFN mismatch through the oracle isolated activation quantization sensitivity:
  identical activations give bit-identical native/oracle Q8_0 dot outputs, while tiny
  earlier floating-point differences change activation quantization. Later drift
  remains unqualified.
- The real Q4 model answered `4` to `What is 2 + 2? Answer briefly.` over HTTP with
  reasoning disabled, and with a separate reasoning trace at the default `max` effort.
- A 1 GiB dense-offload test on one A6000 ran native CUDA Q8 kernels, confirmed by
  dispatch counters. Its two forced-token next-token choices matched the CPU oracle.
  This test does not qualify a full two-GPU placement or establish a speedup.
- A real HTTP tool-choice request completed on that CUDA configuration as
  `weather({"city":"Paris","days":2})`, with `days` preserved as an integer.

Current limits:

- One serialized text context. No batching, shared KV pool, prompt checkpoints or
  conversation cache; reported cached tokens are zero.
- Default context 8192. The loader accepts limits up to the model's declared 1M,
  but near-capacity operation and long-context quality have not been qualified.
- No vision, MTP or DFly2. The GGUF's one converted next-token block is not the
  five-layer DFly2 drafter; neither executes in this port.
- No logprobs, logit bias or Qwen tuning keys. Unsupported requests are refused.
- Linux CPU/CUDA builds were exercised. HIP, SYCL, Metal and Windows are untested.
- Q4_K_M is the actual-weight artifact tested. Other quantizations are unqualified.

The model uses the [Tencent Hy License](https://huggingface.co/tencent/Hy4-flash/blob/main/LICENSE),
which includes additional terms for large MaaS businesses. Model weights are not
included in this repository.
