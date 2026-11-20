"""Export Hy4's tokenizer and a manual native-engine serving configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from gguf_reader import GGUFFile
from strata_tokenizer import DEEPSEEK3_PATTERNS, extract


def prepare(
    model: Path,
    engine: Path,
    output: Path,
    context=8192,
    threads=4,
    cuda_devices: str | None = None,
    cuda_cache_mib: str | None = None,
):
    if not engine.is_file():
        raise ValueError(f"native engine not found: {engine}")
    if (cuda_devices is None) != (cuda_cache_mib is None):
        raise ValueError("provide both --cuda-devices and --cuda-cache-mib")
    if cuda_devices is not None:
        devices = [int(x) for x in cuda_devices.split(",")]
        budgets = [int(x) for x in cuda_cache_mib.split(",")]
        if (
            len(devices) != len(budgets)
            or len(set(devices)) != len(devices)
            or min(devices + budgets) < 0
        ):
            raise ValueError(
                "CUDA devices must be unique and have one nonnegative MiB budget each"
            )
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
        ],
        "arch": "hy_v4_flash",
        "model_name": "hy4-flash",
        "backend": "cpu",
        "parallel": 1,
        "tokenizer": str((output / "tokenizer").resolve()),
        "log": str((output / "engine.log").resolve()),
        "reasoning_effort": "max",
        "reasoning_budget_tokens": 0,
        "answer_room": False,
        "sampling": {"temperature": 0.9, "top_p": 1.0},
        "fit_max_tokens": False,
        "drafting": "off",
        "engine_ready_s": 900,
    }
    if cuda_devices is not None:
        cfg["backend"] = "cuda"
        cfg["args"] += [
            "--cuda-devices",
            cuda_devices,
            "--cuda-cache-mib",
            cuda_cache_mib,
        ]
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
        )
    )
