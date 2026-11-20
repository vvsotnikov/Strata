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

On the CUDA target, bulk sparse KV defaults to host F16, while the bounded
sliding-window rings use GPU memory. `--swa-kv-location ram` moves those rings
back to RAM. Moving the bulk pool to GPU requires `kv_gpu` in the executable's
reported features and a CUDA build. Vision defaults to CPU to avoid assuming a GPU memory
budget for the encoder. `--vision gpu` is explicit opt-in and requires the native
`vision_gpu_reserve` capability. Setup then reserves 2048 MiB on the first
selected card for the external encoder; CPU vision adds no GPU vision reserve.
Native speculation
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

The port is native throughout. CUDA executes target attention/indexing,
quantized projections and all 16 selected routed experts. Engram remains host
mapped. Bulk sparse KV defaults to host F16; the CUDA target gathers selected host cells into
a device tile before attention rather than making every head reread host memory.
Expert misses stream weights to CUDA without dropping experts. Whole resident
layers use routed kernels without copying routing IDs to the CPU. Dense matrices
and resident experts occupy bounded arenas, avoiding the physical allocation
inflation observed with thousands of separate CUDA allocations.

Serving uses native slots for solo and concurrent requests. A shared KV allocator
and admission reservations account for each live request's remaining prompt and
output budget. A request waits for capacity instead of overcommitting another
active conversation. Prompt processing supports tiles up to 512 rows; decode and
MTP verification remain bounded to eight rows. With MTP enabled, the number of
proposals shrinks under concurrency to fit that verification capacity. Rejection
requires target truncation, canonical draft replay and restoration of accepted
features. Cancellation is checked between bounded GPU windows.

The source installer selects a 256-row prompt tile on the current CUDA target,
bounded by the executable's advertised limit (at most 512). CPU reference builds
and older executables without that limit retain eight rows. Explicit choices
remain unchanged. MTP-first residency, host-KV staging and GPU sliding-window KV
are model-specific defaults; `--trunk-first` and `--swa-kv-location ram` provide
controlled opt-outs. These changes were measured together and do not each have
an isolated end-to-end speedup claim. Prompt checkpoints retain
cached logits with the target/draft state, so an identical prompt can resume at
its full length. The CPU scheduler tests check this without recomputing a prompt
token, including MTP state, restore failure and changed image embeddings.
`ckpt=0` bypasses reuse and snapshot saving.

### Numerical and integration checks

Checks use Linux x86-64, GCC 13.3, CUDA and RTX A6000 (sm_86).

- CPU tests cover quantized arithmetic, memory planning, sampler distribution,
  scheduler admission/cancellation, checkpoint restoration, image positions and
  independent NumPy forwards. The standalone benchmark-evidence suite has nine
  passing tests, including truncated streams, natural EOS, deferred grading and
  bounded concurrent collection.
- All eleven CPU CTests pass. The final Python regression covering frontend,
  reasoning, image counting, launch and native protocol has 83 passes and 143
  subtests. Overlapping earlier targeted counts are not added to these totals.
- Generated-weight one-/two-GPU tests exercise four slots, recursive draft
  features, rejection across pooling/SWA boundaries and slot reuse. The original
  bounded target/image/MTP tests had maximum absolute error 6.97e-7 against the
  CPU reference. Expert tests cover streaming, LRU caching, local/remote residency
  and Q4_K/Q6_K/F32/F16/BF16 execution, including 1536-wide BF16 expert batches.
- The final host-KV staging run passed all twelve bit-identical gathered-cell
  and attention cases: F16/F32, sparse/sliding attention, one/eight/sixteen query
  rows, wraparound and nonidentity physical mappings.
- The actual SHA-verified 27-layer projector was checked on a 64x64 image against
  an independent NumPy calculation. GPU F32 maximum error was 2.86e-6; GPU F16
  minimum cosine similarity was 0.99999976. The ordinary OpenAI image path also
  produced `Red and blue.` through the CPU encoder and native CUDA target.
  Runtime admission was also tested with both the actual projector and a tiny
  fixture: GPU encoding, a forced low-memory observation after READY, CPU
  fallback, then GPU recovery. Recovery was bit-identical to the first GPU
  result. The real-image F16/CPU maximum errors against NumPy were 0.001299 and
  3.10e-6. This used a test-only observation override, not actual VRAM exhaustion.
- The actual Q4_K_M model starts with automatic budgets on both selected A6000s,
  a 1,048,576-token shared RAM KV pool, four slots and MTP2. This supersedes the
  earlier integration run with only a 1 GiB dense/512 MiB expert cache. Allocated
  VRAM does not establish that every model tensor is resident: Engram stays in
  RAM and cache misses remain possible.
- The normal Python transport uses BGEN/BGENI for Hy4, avoiding Qwen-specific
  solo handoffs. Native EOS metadata is used to drain final completion counters.

The independent llama.cpp reference is the Hy4 port at
`b7c3f5da30fe1c44f7d015dc051ef0f373c708ec`, linked from
[upstream PR 32517](https://github.com/ggml-org/llama.cpp/pull/32517).
It is a test oracle, not a runtime dependency. A separate FP32 PyTorch calculation
implements the published model equations over the same dequantized GGUF tensors;
it is not an evaluation of the original BF16 model's quality.

The earlier 35-position native/llama comparison matched 33 next-token choices.
Comparing both implementations against the independent FP32 calculation gives
34/35 matches for each. Identical-input arithmetic checks over a complete
92-token real code prefix covered 648,232 quantized matrix calls and 22,549,962
reference dot rows, with zero differing activation bytes or dot-product bits.
This establishes the tested quantization contract; it does not claim that every
intermediate activation in the two full executions is identical.

Complete highest-effort HumanEval/0 and MMLU college-physics/0 prompts were also
teacher-forced through all three CPU implementations. At the final assistant
boundary, all three choose the same top token:

| Prompt | Tokens | Native vs FP32 KL / TV | llama.cpp vs FP32 KL / TV |
| --- | ---: | ---: | ---: |
| HumanEval/0 | 153 | 0.02969 / 0.07464 | 0.11810 / 0.12115 |
| College physics/0 | 144 | 0.0000417 / 0.000113 | 0.0000546 / 0.000142 |

The actual CUDA target was also compared with the independent FP32 reference
on canonical retokenizations of completed assistant responses. All 166 coding
continuation positions agreed on top-1 (mean KL 3.02e-5), including all 32 final
algorithm-body positions (mean KL 2.28e-6). The science continuation agreed on
31/32 positions (mean KL 1.17e-4); the differing choice was near-tied whitespace,
with reference probability 0.50024 and GPU probability 0.49845. These reconstructed
continuations preserve the exact rendered prompts but are not claimed to be the
original sampled token IDs.

User-prefix logit statistics are retained separately from the assistant-boundary
and continuation results above. Completed-task quality is measured separately:
the actual maximum-effort CUDA mini suite completed all eight requests
naturally: HumanEval/0–3 passed all public execution tests in isolated CPU
containers, and college-physics/0–3 returned all four correct answers. None hit
the 3072-token output cap or ended with an error. The suite used up to four
concurrent requests; coding responses used 585–917 tokens and physics
responses 246–1026 tokens, including reasoning. This is four coding and four
science cases, not full HumanEval pass@1 or a full MMLU benchmark.

The final real-model API smoke checks pass in both OpenAI and Anthropic image
dialects. Anthropic image count returns 43 tokens, matching the same generation
request's total input including cached tokens. Counting shares the actual image
pad expansion, stores no prompt checkpoint and creates no combined SVE request
file or language-model generation. Disabled Anthropic thinking maps to Hy4's
`no_think` mode. A forced `get_weather` call carries Paris, repeated checkpoint
opt-outs reuse zero tokens, and all four concurrent greedy fresh/cached
completions agree with complete prefix reuse. A canceled request drains its own
slots, after which `2+2` returns `4`.

Earlier reasoning-disabled code/physics smoke tests also passed. The profiled
code sample passed, while its aborted science sample remains recorded as a
request error; it was not silently converted into a successful answer.

### Measured performance

Exact-size real-source request fixtures are generated with the live CPU token
counter and checked against actual OpenAI usage. They contain no repeated padding
and preserve natural EOS. C1 and C4 use disjoint prompts. Initial/repeated are
ordering labels; only reported cache hits determine whether a sample is cold.
The reusable harnesses are in `engine/tests/hy4/qualification.py`,
`prepare_qualification.py` and `quality.py`. Generated-code execution is deferred
outside timing windows and uses a pinned local Docker image with no network,
host mounts or GPU access.

The headless canonical-ingestion candidate completed an exact 8K-to-1K C1 pair
on the selected two A6000s. Both generated 1,024 tokens with identical output
SHA256 `773562bd43cb1ebc44af367b79129c73d45b6b1ad908888ab56b88f0f7163898`:

| C1 request | Cached tokens | TTFT | Fresh prefill tok/s | Observed decode tok/s | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold | 0 | 160.213s | 51.1435 | 17.5483 | 218.509s |
| Full prompt cached | 8,192 | 0.0124s | — | 17.6536 | 57.961s |

Four parallel, disjoint 8K-to-1K requests also complete with the same configuration:

| C4 wave | Cached tokens per request | TTFT range | Aggregate observed decode tok/s | Wave wall time |
| --- | ---: | ---: | ---: | ---: |
| Cold | 0 | 699.432–716.512s | 17.6837 | 930.832s |
| Full prompts cached | 8,192 | 0.0200–0.5928s | 19.7770 | 206.928s |

Every C4 request reads 8,192 prompt tokens and generates 1,024 tokens. The cold
wave's observed decode window includes remaining prefill interference; these are
serving measurements, not isolated GPU kernel rates. Unlike the C1 pair, three
of four C4 cold/cached output hashes differ; the exact cause is not established
by these measurements, and no deterministic stochastic-wave claim is made.
The separate final four-slot greedy fresh/cached check does match all messages
with complete prefix reuse; that is the deterministic cache-equivalence check.

Canonical MTP ingestion now omits unused vocabulary projections
and logits readback. Its paired GPU test preserves future features/logits
bit-for-bit across slots, images, rejection and slot reuse. The bounded real-model
screen also retained the exact 64-token output SHA. Combined residency, staging,
SWA placement and ingestion changes are not isolated per-optimization speedups.

The timed executable SHA256 is
`b84eea5377a3465f705754e5544d3c27c85673937c2c614ed20f030e2d1be9f0`.
Two subsequent defensive low-level snapshot fixes are not in that executable:
valid feature carry with an empty draft history, and forced-zero draft rewind
invalidation. Both pass the subsequent two-GPU wide/peer regression run,
including all four F16/F32 and RAM/GPU-SWA checkpoint modes. The final candidate
text executable is SHA256
`a148bdb5462efa4d7f86466e844e730ba3d6d9008ee76fece02f4c0cac534a8f`.
Its final real-API checks pass: both OpenAI and Anthropic image requests, native
image count, forced tool call, checkpoint opt-out, four concurrent greedy
fresh/cached equivalents, and owned-request cancellation followed by reuse.
The long benchmark retains the earlier binary identity above rather than
claiming the defensive fixes were timed.

For context, the same-day existing DeepSeek deployment completed its cold
8K-to-1K C1 request in 32.99s (16.00s TTFT, 60.19 observed decode tok/s).
Its C4 aggregate observed decode was 79.70 tok/s cold and 89.15 with cached KV.
The captured native build was `619a92fa3b2c-sm86-ragged`, using `--tp 2`,
`--experts 6`, `--dspark on --dspark-conf 0.65`, `--prefill-chunk 8192`,
`--max-context 262144` and `--parallel 4`. This is DSpark with a different
model, quantization, expert count and context configuration; Hy4 uses all 16
routed experts and MTP2. It compares serving choices on this machine, not the
isolated speed of the same model across engines or equivalent speculative
algorithms. DeepSeek remains faster in these C1/C4 measurements. Hy4 is not
being cut over to production. The original DeepSeek native and adapter
containers are restored with the same IDs, images and commands, both healthy.
The public service is loaded and a public `2+2` request returns `4`; no backend
configuration was changed for a Hy4 cutover.

The final extended CUDA tests pass prompt widths 8, 9, 15, 16, 31, 128, 256 and
511, cross-chunk MTP features, exact sparse selections and odd shared-pool
capacity. A 64-token checkpoint/replay is bit-identical in all four F16/F32 and
RAM/GPU-SWA combinations. Mixed-format arena coalescing, two-card streaming and
remote residency tests also pass. A CLI tracing fixture writes all 48 expected
records for 16 tokens and three layers.

The former strict cross-chunk comparison exposed the precision difference
between FP32 arithmetic and rounded FP16 tensor-core inputs. Its final shifted
case has normalized target error 1.64e-4 under an explicitly documented 2^-11
FP16-scale empirical tolerance; this is not a formal error bound. A separate
independent test holding the rounded projection inputs identical has maximum
absolute error 7.15e-7. Original failed logs are retained. Neither this tolerance
nor the small-shape tests imply bit-identical full-model execution; actual
assistant-distribution and task checks are reported separately above.

A successful 1M-token pool allocation and the measured shorter requests do not
establish near-capacity correctness or long-context quality. Full-context stress
and larger task suites remain separate validation work.

`--capabilities` reports compiled features without loading weights. Setup and the
server inspect the executable before enabling features; saved JSON cannot add
support to an older binary. The catalog artifact is the pinned Q4_K_M. Other
quantizations and five-layer DFly2 have no completed qualification here; no Hy4
DFlash2 pack is provided. Logprobs, logit bias and Qwen tuning keys are refused.
Linux CPU/CUDA is the tested target; HIP, SYCL, Metal and Windows have not been
runtime-qualified for this target.

The model uses the [Tencent Hy License](https://huggingface.co/tencent/Hy4-flash/blob/main/LICENSE),
which includes additional terms for large MaaS businesses. Model weights are not
included in this repository.
