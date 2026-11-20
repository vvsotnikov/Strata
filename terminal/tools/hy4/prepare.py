"""Prepare Hy4's native tokenizer and serving configuration (also used by setup)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from gguf_reader import GGUFFile
from strata_control import hy4
from strata_tokenizer import DEEPSEEK3_PATTERNS, extract


def prepare(
    model: Path,
    engine: Path,
    output: Path,
    context=8192,
    threads=4,
    cuda_devices: str | None = None,
    cuda_cache_mib: str | None = None,
    *, expert_cache_mib: str | None = None, slots: int = 1,
    context_pool: int | None = None, prefill_chunk: int | None = None,
    kv_location: str = "ram", kv_type: str | None = None, mmproj: Path | None = None,
    mtp: int = 0, native_caps: dict | None = None, vision_gpu: bool = False,
    vram_reserve_mib: int | None = None,
):
    if not engine.is_file():
        raise ValueError(f"native engine not found: {engine}")
    caps = native_caps if native_caps is not None else hy4.capabilities(engine)
    features = caps.get("features") or {}
    if cuda_devices is None:
        # This executable's CUDA features require a selected device path. A
        # no-device start is the serial CPU reference, including in CUDA builds.
        features = {key: False for key in features} | {"kv_ram": True}
    if kv_type is None:
        kv_type = "f16" if cuda_devices is not None else "f32"
    if cuda_devices is None and kv_type != "f32":
        raise ValueError("the Hy4 CPU reference stores KV in F32")
    if cuda_devices is None and (cuda_cache_mib is not None or expert_cache_mib is not None):
        raise ValueError("CUDA weight budgets require --cuda-devices")
    if cuda_devices is not None:
        devices = [int(x) for x in cuda_devices.split(",")]
        if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
            raise ValueError("CUDA devices must be unique nonnegative ordinals")
        for budget in (cuda_cache_mib, expert_cache_mib):
            if budget is not None:
                values = [int(x) for x in budget.split(",")]
                if len(values) != len(devices) or min(values) < 0:
                    raise ValueError("provide one nonnegative MiB weight budget per CUDA device")
        if caps.get("backend") != "cuda":
            raise ValueError("this Hy4 executable was built without CUDA")
    if type(slots) is not int or not 1 <= slots <= 8:
        raise ValueError("slots must be 1..8")
    if slots > 1 and not features.get("batching"):
        raise ValueError("this Hy4 engine does not implement batched serving")
    limits = caps.get("limits", {})
    if not isinstance(limits, dict):
        raise ValueError("the Hy4 engine returned malformed capability limits")  # noqa: TRY004 - invalid executable JSON is a configuration error
    native_prefill = limits.get("prefill_rows", 8)
    if type(native_prefill) is not int or native_prefill < 1:
        raise ValueError("the Hy4 engine returned an invalid prefill limit")
    max_prefill = min(512 if cuda_devices is not None else 8, native_prefill)
    if prefill_chunk is None:
        prefill_chunk = min(256, max_prefill)
    if type(prefill_chunk) is not int or not 1 <= prefill_chunk <= max_prefill:
        raise ValueError(f"this Hy4 engine supports prefill chunks of 1..{max_prefill}")
    if kv_location not in ("ram", "gpu") or kv_type not in ("f16", "f32"):
        raise ValueError("unsupported Hy4 KV setting")
    if kv_location == "gpu" and (cuda_devices is None or not features.get("kv_gpu")):
        raise ValueError("this Hy4 engine does not support a GPU KV pool")
    if type(mtp) is not int or not 0 <= mtp <= 5 or (mtp and not features.get("mtp")):
        raise ValueError("this Hy4 engine does not support the requested MTP window")
    if mmproj is not None and (not mmproj.is_file() or not features.get("vision")):
        raise ValueError("vision requires the Hy4 projector and a vision-capable Hy4 engine")
    if vision_gpu and (mmproj is None or cuda_devices is None):
        raise ValueError("GPU vision requires a CUDA model configuration and its projector")
    if vision_gpu and not features.get("vision_gpu_reserve"):
        raise ValueError("GPU vision requires an updated Hy4 engine with a vision memory reserve")
    if vram_reserve_mib is not None and (type(vram_reserve_mib) is not int or vram_reserve_mib < 0):
        raise ValueError("VRAM reserve must be nonnegative MiB")
    g = GGUFFile(model)
    if g.metadata.get("general.architecture") != "hy_v4_flash":
        raise ValueError("expected a hy_v4_flash GGUF")
    if g.metadata.get("tokenizer.ggml.pre") != "hy_v4":
        raise ValueError("unsupported Hy4 pre-tokenizer")
    if (
        not 1 <= context <= g.metadata["hy_v4_flash.context_length"]
        or not 1 <= threads <= 256
    ):
        raise ValueError("context or CPU thread count out of range")
    pool = context if context_pool is None else context_pool
    if type(pool) is not int or not context <= pool <= slots * g.metadata["hy_v4_flash.context_length"]:
        raise ValueError("shared context pool must cover one request and fit the configured slots")
    if cuda_devices is None and pool != context:
        raise ValueError("the Hy4 CPU reference has one context, not a shared KV pool")
    output.mkdir(parents=True, exist_ok=True)
    tk = extract(model, output)
    tk.pop("pre_pattern", None)
    tk["pre_patterns"] = list(DEEPSEEK3_PATTERNS)
    tk["pre_pattern_source"] = "llama.cpp b7c3f5da src/llama-vocab.cpp:328-336 (HY_V4)"
    tk["add_bos_token"] = bool(g.metadata.get("tokenizer.ggml.add_bos_token", False))
    (output / "tokenizer/tokenizer.json").write_text(
        json.dumps(tk, ensure_ascii=False, indent=2) + "\n"
    )
    cfg = {
        "exe": str(engine.resolve()),
        "args": [
            "--model",
            str(model.resolve()),
            "--threads",
            str(threads),
            "--max-context",
            str(context),
            "--context-pool", str(pool), "--slots", str(slots),
            "--prefill-chunk", str(prefill_chunk), "--kv-location", kv_location,
            "--kv-type", kv_type, "--mtp", str(mtp),
        ],
        "arch": "hy_v4_flash",
        "model_name": "hy4-flash",
        "backend": "cpu",
        "parallel": slots,
        "context_pool": pool,
        "native_capabilities": caps,
        "tokenizer": str((output / "tokenizer").resolve()),
        "log": str((output / "engine.log").resolve()),
        "reasoning_effort": "max",
        "reasoning_budget_tokens": 0,
        "answer_room": False,
        "sampling": {"temperature": 0.9, "top_p": 1.0},
        "fit_max_tokens": False,
        "drafting": "mtp" if mtp else "off",
        "mtp": bool(mtp),
        "engine_ready_s": 900,
    }
    if cuda_devices is not None:
        cfg["backend"] = "cuda"
        cfg["args"] += ["--cuda-devices", cuda_devices]
        if cuda_cache_mib is not None:
            cfg["args"] += ["--cuda-cache-mib", cuda_cache_mib]
        if expert_cache_mib is not None:
            cfg["args"] += ["--expert-cache-mib", expert_cache_mib]
    if vram_reserve_mib is not None:
        cfg["args"] += ["--vram-reserve-mib", str(vram_reserve_mib)]
    if vision_gpu:
        cfg["args"] += ["--vision-gpu-reserve-mib", "2048"]
    if mmproj is not None:
        reader = engine.with_name("strata-hy4-vision" + (".exe" if engine.suffix == ".exe" else ""))
        if not reader.is_file():
            raise ValueError("the Hy4 vision encoder executable is missing; build strata-hy4-vision")
        cfg["args"] += ["--mmproj", str(mmproj.resolve())]
        cfg["vision"] = {"exe": str(reader.resolve()), "mmproj": str(mmproj.resolve()),
                         "model": str(model.resolve()), "gpu": bool(vision_gpu), "threads": threads,
                         "max_tokens": 1024 if vision_gpu else 300}
    path = output / "serve.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--engine", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--cuda-devices", help="ordinals within CUDA_VISIBLE_DEVICES, e.g. 0,1"
    )
    p.add_argument(
        "--cuda-cache-mib",
        help="dense weight budget per selected device, e.g. 8192,8192",
    )
    p.add_argument("--expert-cache-mib")
    p.add_argument("--vram-reserve-mib", type=int)
    p.add_argument("--slots", type=int, default=1)
    p.add_argument("--context-pool", type=int)
    p.add_argument("--prefill-chunk", type=int)
    p.add_argument("--kv-location", choices=("ram", "gpu"), default="ram")
    p.add_argument("--kv-type", choices=("f16", "f32"))
    p.add_argument("--mtp", type=int, default=0)
    p.add_argument("--mmproj", type=Path)
    p.add_argument("--vision-gpu", action="store_true")
    a = p.parse_args()
    print(
        prepare(
            a.model,
            a.engine,
            a.output,
            a.context,
            a.threads,
            a.cuda_devices,
            a.cuda_cache_mib,
            expert_cache_mib=a.expert_cache_mib, slots=a.slots, context_pool=a.context_pool,
            prefill_chunk=a.prefill_chunk, kv_location=a.kv_location, kv_type=a.kv_type,
            mmproj=a.mmproj, mtp=a.mtp, vision_gpu=a.vision_gpu, vram_reserve_mib=a.vram_reserve_mib,
        )
    )
