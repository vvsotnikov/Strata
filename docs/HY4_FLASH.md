# Native Hy4-flash in this source fork

Upstream Strata at `3142b66b96e9c0e775e30f620b11b191a1427c2a` has no Hy4 engine.
This fork adds a separate native `hy_v4_flash` target, selected as `hy4` in the
ordinary model catalog. It reads AngelSlim's GGUF shards directly and does not
start llama.cpp or link libllama. Released Qwen defaults remain unchanged.

The port implements Hy4's own iHC residual streams, latent MoE with all 16
selected experts, gated MLA with sinks, sliding-window attention, pooled sparse
indexing and host-mapped FP8 Engram. The HTTP server uses the GGUF's namespaced
chat template, reasoning, tool arguments and image markers. Source setup selects
this engine rather than applying a Qwen pack, projector or tuning profile.

## Ordinary setup and launch

Use this fork's source checkout, not an installed release bundle. From its root:

```sh
./setup.sh --setup --family hy4 --model Q4_K_M \
  --gguf-dir /models/AngelSlim-Hy4-flash-GGUF \
  --gpus 2,3 --context 1048576 --context-pool 1048576 --parallel 4 \
  --vision cpu --no-start --yes
```

The card numbers are examples: choose the intended physical cards. Setup reads
their architecture, builds `strata-hy4` and `strata-hy4-vision` in
`build-hy4-cuda`, verifies both shards and the projector, exports the tokenizer,
and writes the normal `terminal/strata-hy4-q4_k_m.json` and start script. It needs
a C++20 compiler, CMake and a CUDA toolkit for the selected architecture. There
is no published Hy4 engine binary to download. Omitting `--gguf-dir` downloads
the pinned files. A local folder must contain every required file; setup does
not silently replace an incomplete local model with a download.

`--context-pool` counts **shared** logical KV tokens. Four slots sharing 1,048,576
tokens do not each receive a separate million-token allocation. `--context` is
the per-request maximum. Native memory allocation and live capability checks
still determine whether a requested configuration can start.

The normal Terminal source command also uses this own-family installer:

```sh
PYTHONPATH=terminal:terminal/tools python terminal/strata.py launch \
  --family hy4 --model Q4_K_M --model-path /models/AngelSlim-Hy4-flash-GGUF \
  --gpu 2,3 --context 128k --vision cpu --yes --no-tui
```

To make a read-only plan, add `--plan`. Giving only `--model-path` discovers Hy4
from `general.architecture = hy_v4_flash` and validates tensor encodings and
both shard headers. File names alone are not evidence of compatibility.

For the serial CPU reference build, use `--backend cpu` in source setup. It uses
F32 KV and one context; shared-pool, speculation and image ingestion require the
selected CUDA device path. A CUDA-built executable without `--cuda-devices` also
stays on this CPU reference path, so it never silently chooses an unselected GPU. For an already
built pair of executables set `STRATA_HY4_ENGINE=/path/to/strata-hy4`; its vision
reader must be alongside it. Ordinary starts never compile or download.

`./setup.sh --update` rebuilds the installed Hy4 source target for its saved
selected cards and checks the resulting capabilities. It does not fetch a Qwen
engine component for a Hy4-only installation. An engine still serving is left
untouched with a pending-update message; stop it and rerun the update. A custom
`STRATA_HY4_ENGINE` executable stays externally managed.

## Memory and native options

The pinned [AngelSlim/Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF)
revision is `bcbc7db5dcf4162cc1e1b8ed6c65567e843ebe46`:

| File | Bytes |
| --- | ---: |
| Hy4-flash-Q4_K_M-00001-of-00002.gguf | 104,986,517,504 |
| Hy4-flash-Q4_K_M-00002-of-00002.gguf | 63,360,616,128 |
| mmproj-Hy4-flash-bf16.gguf | 1,044,397,440 |

The model shards total 168,347,133,632 bytes (156.79 GiB). Mapped bytes are not a
claim that all pages are resident or locked. Engram occupies about 59.01 GiB and
stays host mapped. The pinned trunk's expert tensors occupy 83.35 GiB and its
other tensors 7.45 GiB, excluding MTP and runtime caches. The converted MTP
block uses BF16 experts, unlike the trunk's Q4_K/Q6_K experts.

Dense and expert CUDA budgets are automatic when omitted. The native planner
must account for actual tensors, scratch, KV and the free memory of the selected
cards. An explicit `--cuda-cache-mib` or `--expert-cache-mib` is a per-device
budget in MiB; explicit zero requests streaming rather than automatic residency.
`--vram-reserve-mib` leaves room for other processes; it cannot reserve VRAM
against them. These native arguments can be set in the written config.

Use GPU UUIDs in an explicit `CUDA_VISIBLE_DEVICES`. `--cuda-devices 0,1` means
visible ordinals inside that process. The server maps source setup's physical
card selection to the child process; do not confuse those numbering schemes.

On the CUDA target, KV defaults to host F16. GPU KV requires `kv_gpu` in the executable's reported
features and a CUDA build. Vision defaults to CPU to avoid assuming a GPU memory
budget for the encoder. `--vision gpu` is explicit opt-in. Native speculation
uses the converted MTP block when the executable reports that feature;
`--mtp off` disables it. No external draft model is fetched for Hy4.

Reasoning defaults to `max`, this template's highest level. Requests can select
`low`, `high`, `max` or `none` (`no_think`). Sampling defaults are temperature 0.9
and top_p 1.0, without a top-k cap. Declared string tool arguments remain strings
when the model emits an accepted alias of a namespaced tool.

## Direct development build

The installer owns the same source commands below. They are optional for
contributors, not an extra preparation step required for normal setup:

```sh
cmake -S engine -B build-hy4-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_NATIVE_EXPERTS=OFF \
  -DSTRATA_BUILD_HY4=ON -DSTRATA_HY4_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=ON
cmake --build build-hy4-cuda --target strata-hy4 strata-hy4-vision -j4
build-hy4-cuda/src/hy4/strata-hy4 --capabilities
```

Set `STRATA_HY4_CUDA=OFF` for a CPU-only reference build. `STRATA_ENABLE_CUDA=OFF`
disables the separate Qwen target; it does not disable Hy4's own CUDA option.
The independent forward/vision fixtures need NumPy in CMake's selected Python.
Use `-DPython3_EXECUTABLE=/path/to/python` when necessary.

## Validation and limits

The port is native throughout. CUDA executes the target attention/indexer,
quantized projections and all selected routed experts. Host memory holds Engram
and, by default, F16 KV. Expert misses stream weights to CUDA; they do not skip
experts. Whole resident layers use routed kernels without copying routing IDs
to the CPU. Separate cards exchange activations and accepted MTP features directly.

Serving uses native slots for both solo and concurrent requests. The shared KV
allocator supplies capacity across those slots; it does not reserve a full context
per slot. Continuous batching alternates bounded prefill tiles with decode windows.
The current native tile is at most eight rows, with up to eight slots. MTP windows
are 0..5 and shrink under concurrency to fit verification into the same tile.
Rejected candidates require target truncation, canonical draft replay and restoration
of the accepted target feature. Cancellation is checked between GPU windows.

Checked on Linux x86-64, GCC 13.3, CUDA and RTX A6000 (sm_86):

- Nine CPU CTests pass, including the scalar memory planner, sampler distribution,
  native scheduler, cancellation, image positions and independent NumPy forwards.
  The separate NumPy image/MTP fixture has maximum logit errors below 6e-7.
- Python launcher/frontend/protocol regression: 436 passed, one skipped because
  PowerShell was unavailable, and 4,113 subtests passed.
- One- and two-GPU target/image/MTP tests pass against the CPU reference, with
  maximum absolute error 6.97e-7. They exercise four slots, eight-row batches,
  recursive draft features, rejection across pooling/SWA boundaries and slot reuse.
- CUDA expert tests compare streaming, LRU caching, local/remote residency and
  Q4_K/Q6_K/F32/F16/BF16 execution. Tested outputs are bit-identical, including
  1536-wide BF16 expert batches. These are generated weights at the model's shapes.
- The SHA-verified real projector was checked through all 27 layers against an
  independent NumPy calculation on a 64x64 image. GPU F32 output has maximum
  error 2.86e-6; default GPU F16 has minimum cosine similarity 0.99999976.
- The actual Q4_K_M model starts with a 1,048,576-token shared RAM KV pool,
  four slots and MTP. Bounded text requests return `4`, max-effort reasoning
  produces a separate thought stream, four concurrent requests complete with
  correct prompt counts, and image embeddings produce a coherent description.
  The normal OpenAI-compatible image request identifies the real image as
  `Red and blue.` through the CPU vision encoder and native GPU target.
  Multiple speculative rounds include both accepted and rejected candidates.
- The ordinary Python transport is tested against the actual C++ scheduler,
  including four concurrent requests. It uses BGEN/BGENI throughout, avoiding
  Qwen-specific solo handoffs and their prompt/decode accounting errors.

The real-model serving checks above deliberately use only a 1 GiB dense cache
and 512 MiB expert cache on one shared A6000 beside the existing service. They
qualify execution and integration, not resident-model speed. A 1M allocation and
short requests do not establish near-capacity correctness or long-context quality.
Full-model two-GPU residency, sustained throughput and comparative model quality
remain unqualified.

The indexer scoring kernel was measured separately at 262,143 pooled groups,
128 dimensions and 16 heads, with nonidentity physical pages. Two paired samples
on the shared A6000 measured the previous kernel at 39.05/79.41 ms and the warp
kernel at 0.81/1.82 ms. Maximum score difference was 8.94e-8. These are isolated
kernel results under changing load, not end-to-end inference speedups. Long
prefill still needs wider-query kernels; full radix sorting and direct host-KV
reads also remain tuning opportunities.

Earlier CPU-reference qualification used the independent Hy4 llama.cpp port at
`b7c3f5da30fe1c44f7d015dc051ef0f373c708ec`, linked from
[upstream PR 32517](https://github.com/ggml-org/llama.cpp/pull/32517).
It is a test oracle, not a runtime dependency. Tokenizer IDs matched on 17 cases.
Full real-model logits were not identical: 33/35 forced next-token choices matched,
with mean cosine 0.979 and minimum 0.834. Identical-activation Q8_0 dot replay
isolated activation-quantization sensitivity at the first mismatch; this does not
explain or qualify all later drift. This is not an exact model-parity claim.

`--capabilities` reports compiled features without loading weights. Setup and the
server inspect the executable before enabling features; saved JSON cannot add
support to an older binary. The supported catalog artifact is the pinned Q4_K_M.
Other quantizations and five-layer DFly2 are unqualified; no Hy4 DFlash2 pack is
provided. Logprobs, logit bias and Qwen tuning keys are refused. Linux CPU/CUDA is
the tested target; HIP, SYCL, Metal and Windows remain unqualified.

The model uses the [Tencent Hy License](https://huggingface.co/tencent/Hy4-flash/blob/main/LICENSE),
which includes additional terms for large MaaS businesses. Model weights are not
included in this repository.
