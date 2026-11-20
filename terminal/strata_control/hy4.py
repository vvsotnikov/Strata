"""Hy4 artifact identity and the native engine's capability contract.

The ordinary installer and model importer share these pins. No weights or runtime
are fetched while inspecting a model or making a launch plan.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

FAMILY = "hy4"
ARCH = "hy_v4_flash"
NAME = "hy4-flash"
REPO = "AngelSlim/Hy4-flash-GGUF"
REVISION = "bcbc7db5dcf4162cc1e1b8ed6c65567e843ebe46"
MODEL = "Q4_K_M"
FILES = (
    "Hy4-flash-Q4_K_M-00001-of-00002.gguf",
    "Hy4-flash-Q4_K_M-00002-of-00002.gguf",
)
PROJECTOR = "mmproj-Hy4-flash-bf16.gguf"
PINS = {
    FILES[0]: (104986517504, "d6f76e5620f949bb50b23d0104786a4d091c487b6bee331facbeb996445ebe14"),
    FILES[1]: (63360616128, "ecd6a0ecd7f258b1ee9211bf566261ae33efc1743ad40f58cb5004cf7b051839"),
    PROJECTOR: (1044397440, "8c33a48ef57480abf4854126964058a2015b5da3b773b14c847c995cf8899bda"),
}
CONTEXTS = (8192, 32768, 65536, 131072, 262144, 524288, 1048576)


def family_definition(base_url: str) -> dict:
    return {
        "title": "Hy4-flash", "name": NAME, "by": "Tencent; GGUF by AngelSlim", "arch": ARCH,
        "about": "Hybrid sparse/sliding-window MLA, latent MoE with 16 of 512 experts, and a host-mapped Engram table",
        "hf": base_url, "file": "Hy4-flash-{q}-00001-of-00002.gguf", "tag": "hy4-",
        "files": {MODEL: list(FILES)}, "mmproj_hf": base_url, "mmproj": PROJECTOR,
        "sha256": dict(PINS), "source_only": True, "experimental": True,
        "license": "Tencent Hy License: https://huggingface.co/tencent/Hy4-flash/blob/main/LICENSE",
        "models": {MODEL: {"about": "4-bit K-quant, AngelSlim's recommended default", "ram_gb": 180,
                           "download_gb": sum(PINS[n][0] for n in FILES) / 1e9,
                           # Actual trunk tensor bytes in the pinned first shard:
                           # 89,492,815,872 expert + 8,002,029,876 dense bytes.
                           "arena_gb": 89.492815872, "memory_gb": 97.494845748,
                           "experimental": True}},
        # Sparse layers retain one 576-wide MLA cell. SWA layers keep only 512
        # cells per slot; shared-context allocation is sized by the native engine.
        "kv_bytes_per_token": 12 * 576 * 2,
        "context_max": CONTEXTS[-1], "contexts": list(CONTEXTS), "mtp": 0,
        "vision": True, "state_gb": 0.0,
    }


def model_files(family: dict, model: str) -> list[str]:
    """All shards of an own-path model, including older single-file families."""
    return list((family.get("files") or {}).get(model) or [family["file"].format(q=model)])


def capabilities(executable: Path, *, env: dict | None = None) -> dict:
    """Read native capabilities before configuring features, without loading weights."""
    result = subprocess.run([str(executable), "--capabilities"], env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=30, check=False)
    if result.returncode:
        raise ValueError("this Hy4 engine predates capability discovery or cannot start; rebuild strata-hy4")
    try:
        value = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise ValueError("the Hy4 engine returned invalid capability information") from None
    if not isinstance(value, dict) or value.get("architecture") != ARCH or value.get("backend") not in ("cpu", "cuda") or not isinstance(value.get("features"), dict):
        raise ValueError("the executable does not declare Hy4-flash capabilities")
    if any(type(enabled) is not bool for enabled in value["features"].values()):
        raise ValueError("the Hy4 engine returned malformed feature flags")
    return value


def source_tree(terminal: Path) -> Path:
    root = Path(terminal).resolve().parent
    if not (root / "engine/CMakeLists.txt").is_file():
        raise ValueError("Hy4 has no published engine package yet. Use this fork's source checkout or set STRATA_HY4_ENGINE to a built strata-hy4 executable")
    return root


def engine_path(terminal: Path, backend: str = "cuda") -> Path:
    if os.environ.get("STRATA_HY4_ENGINE"):
        path = Path(os.environ["STRATA_HY4_ENGINE"]).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"STRATA_HY4_ENGINE does not name a file: {path}")
        return path
    return source_tree(terminal) / f"build-hy4-{backend}/src/hy4" / ("strata-hy4.exe" if os.name == "nt" else "strata-hy4")


def build_commands(terminal: Path, backend: str, archs: list[str] | None = None) -> list[list[str]]:
    if backend not in ("cpu", "cuda"):
        raise ValueError("Hy4 currently has native CPU and NVIDIA CUDA backends")
    root = source_tree(terminal)
    build = root / f"build-hy4-{backend}"
    args = ["cmake", "-S", str(root / "engine"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
            "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_NATIVE_EXPERTS=OFF", "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_BUILD_HY4=ON",
            "-DSTRATA_HY4_CUDA=" + ("ON" if backend == "cuda" else "OFF")]
    if backend == "cuda":
        codes = sorted({str(a).removeprefix("sm_").removeprefix("compute_") for a in archs or []})
        if not codes or any(not a.isdecimal() or int(a) < 75 for a in codes):
            raise ValueError("choose NVIDIA cards with known compute capability 7.5 or newer before building Hy4")
        args.append("-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(codes))
    return [args, ["cmake", "--build", str(build), "--config", "Release", "--target", "strata-hy4", "strata-hy4-vision", "-j4"]]


def cuda_visible_devices(indexes: list[int]) -> str:
    """Resolve physical nvidia-smi indices to UUIDs before CUDA renumbers them."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("cannot resolve the selected Hy4 GPUs to UUIDs; check nvidia-smi or explicitly set CUDA_VISIBLE_DEVICES") from exc
    found = {}
    for line in result.stdout.splitlines():
        cells = [s.strip() for s in line.split(",")]
        if len(cells) == 2 and cells[0].isdigit() and cells[1].startswith("GPU-"):
            found[int(cells[0])] = cells[1]
    if result.returncode or any(i not in found for i in indexes):
        raise ValueError("a selected physical Hy4 GPU is not available; check nvidia-smi and the saved card selection")
    return ",".join(found[i] for i in indexes)
