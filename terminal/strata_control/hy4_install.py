"""The ordinary install path for Hy4's native source-built engine.

Unlike the Qwen engine component, this target has no published binary package.
It consumes the pinned GGUF shards in place, with the FP8 Engram table mapped on
host memory. All GPU admission and resident weight budgets are native decisions.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from strata_control import hy4


def previous(S) -> dict:
    try:
        value = json.loads((S.home() / "strata-hy4-q4_k_m.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) and value.get("arch") == hy4.ARCH else {}


def saved_option(cfg, name, default):
    args = cfg.get("args") or []
    for i in range(len(args)-2, -1, -1):
        if args[i] == name:
            return args[i+1]
    return default

def build(S, backend: str, archs: list[str]):
    exe = hy4.engine_path(S.ROOT, backend)
    # The source-only target is built by an installer-owned bundled helper,
    # permitted by the same Control boundary as other preparation tools.
    if not os.environ.get("STRATA_HY4_ENGINE"):
        hy4.build_commands(S.ROOT, backend, archs)
        if exe.is_file() and running(exe):
            raise ValueError("the Hy4 engine is still running; stop it before rebuilding from source")
        build_requirements = getattr(S, "BUILD_REQUIREMENTS", None)
        packages = S.requirement_lines(build_requirements) if build_requirements is not None and build_requirements.is_file() else getattr(S, "BUILD_PACKAGES", ["cmake", "ninja"])
        if not getattr(S, "CONTROL", False):
            S.pip_install(packages, "CMake and Ninja for the native Hy4 source build")
        else:
            S.say("  Hy4's source build uses the installed CMake/compiler; no ready-made engine is fetched.")
        command = [sys.executable, str(S.ROOT / "tools/hy4_build.py"), "--backend", backend]
        for arch in archs if backend == "cuda" else []:
            command += ["--arch", arch]
        S.run(command)
    caps = hy4.capabilities(exe)
    if caps.get("backend") != backend:
        raise ValueError(f"the selected Hy4 executable does not provide the {backend} backend")
    return exe, caps


def running(exe: Path) -> bool:
    """Do not rebuild an executable still used by a serving process."""
    import psutil
    target = exe.resolve()
    for process in psutil.process_iter(["exe", "name"]):
        try:
            if process.info.get("exe") and Path(process.info["exe"]).resolve() == target:
                return True
        except (OSError, psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return False


def update(S, paths: list[Path]) -> set[Path]:
    """Rebuild only this source family's own targets; never fetch a Qwen package.

    Custom binaries stay externally managed. Running engines stay untouched and
    are explicitly reported as pending, rather than called up to date.
    """
    grouped = {}
    pending = set()
    for path in paths:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        if cfg.get("arch") == hy4.ARCH:
            grouped.setdefault(str(Path(cfg["exe"]).resolve()), []).append((path, cfg))
    for raw, entries in grouped.items():
        exe = Path(raw)
        backend = entries[0][1].get("backend", "cpu")
        expected = hy4.source_tree(S.ROOT) / f"build-hy4-{backend}/src/hy4" / ("strata-hy4.exe" if os.name == "nt" else "strata-hy4")
        if exe != expected.resolve() or os.environ.get("STRATA_HY4_ENGINE"):
            S.warn(f"Hy4 executable {exe} is externally managed; rebuild it from this source checkout, then rerun setup")
            pending.update(path for path, _ in entries)
            continue
        if running(exe):
            S.warn(f"Hy4 engine {exe} is running: its update is deferred; stop it and run --update again")
            pending.update(path for path, _ in entries)
            continue
        selected = set()
        for _path, cfg in entries:
            gpu = cfg.get("gpu", [])
            selected.update(gpu if isinstance(gpu, list) else [gpu])
        archs = sorted({S.gpu_info(i)["arch"] for i in selected}) if backend == "cuda" else []
        S.say(f"  Updating the native Hy4 {backend} source targets ...")
        _exe, caps = build(S, backend, archs)
        for path, cfg in entries:
            cfg["native_capabilities"] = caps
            S.write_config(path, cfg)
    return pending

def install(S, a, *, gpu=None, multi=None, hip=False, port=8080):
    if hip or getattr(a, "backend", None) not in (None, "cpu", "cuda"):
        raise ValueError("Hy4 currently supports NVIDIA CUDA and the CPU reference backend")
    if getattr(a, "drafter", None) == "dflash2":
        raise ValueError("there is no qualified Hy4 DFlash2 drafter; choose MTP or --mtp off")
    if a.model not in (None, hy4.MODEL):
        raise ValueError(f"Hy4 currently supports the pinned {hy4.MODEL} artifact")
    if getattr(a, "kv", None) not in (None, "fp16"):
        raise ValueError("Hy4 currently stores its KV pool in F16; use --kv fp16")
    old = previous(S)
    ctx = int(saved_option(old, "--max-context", 131072)) if a.context is None else a.context
    slots = int(old.get("parallel", saved_option(old, "--slots", 1))) if getattr(a, "parallel", None) is None else a.parallel
    pool = max(ctx, int(old.get("context_pool", saved_option(old, "--context-pool", ctx)))) if getattr(a, "context_pool", None) is None else a.context_pool
    if getattr(a, "parallel", None) is not None and getattr(a, "context_pool", None) is None:
        pool = min(pool, slots * 1048576)
    if not 1 <= ctx <= 1048576 or not 1 <= slots <= 8 or not ctx <= pool <= slots * 1048576:
        raise ValueError("Hy4 requires 1..1M context, 1..8 slots and a shared pool covering at least one request")
    backend = "cpu" if getattr(a, "backend", None) == "cpu" else "cuda"
    if backend == "cpu" and (getattr(a, "kv", None) is not None or pool != ctx or slots != 1):
        raise ValueError("the CPU reference uses F32 KV and one context; shared/F16 KV requires CUDA")
    if backend == "cuda" and (not gpu or "index" not in gpu):
        raise ValueError("select the physical NVIDIA cards before setting up the Hy4 CUDA target")
    devices = list(multi or [int(gpu["index"])]) if backend == "cuda" else []
    S.step(3, "preparing the Hy4 Python runtime")
    S.pip_install(S.runtime_packages(), "Hy4 tokenizer, HTTP server and image preprocessing")
    S.step(4, "checking the supplied native Hy4 engine" if os.environ.get("STRATA_HY4_ENGINE") else "building the native Hy4 engine from this source checkout")
    exe, caps = build(S, backend, (gpu or {}).get("archs") or [(gpu or {}).get("arch", "")])
    features = caps["features"]
    if slots > 1 and not features.get("batching"):
        raise ValueError("this Hy4 engine does not implement native batch slots")
    choice = getattr(a, "mtp", None)
    old_mtp = int(saved_option(old, "--mtp", 2))
    kept_off = choice is None and (old.get("mtp") is False or old.get("drafting") == "off" or old_mtp == 0)
    mtp = 0 if choice == "off" or kept_off or not features.get("mtp") else (old_mtp if choice is None else old_mtp or 2)
    if getattr(a, "mtp", None) == "on" and not features.get("mtp"):
        raise ValueError("this Hy4 engine does not yet implement MTP")
    vision_word = getattr(a, "vision", None)
    if vision_word in ("gpu", "cpu", "yes") and not features.get("vision"):
        raise ValueError("this Hy4 engine does not yet implement vision")
    vision = features.get("vision") and vision_word not in ("none", "no")
    folder = Path(a.gguf_dir).expanduser().resolve() if a.gguf_dir else Path(a.models_dir) / ("hy4-" + hy4.MODEL)
    folder.mkdir(parents=True, exist_ok=True)
    S.step(5, "checking the pinned Hy4 GGUF shards" if a.gguf_dir else "downloading the pinned Hy4 GGUF shards")
    fam = S.OWN_PATH_FAMILIES[hy4.FAMILY]
    for name in [*hy4.FILES, *([hy4.PROJECTOR] if vision else [])]:
        path = folder / name
        if not path.is_file():
            if a.gguf_dir:
                raise ValueError(f"{path} is missing; --gguf-dir must contain both shards" + (" and the projector" if vision else ""))
            S.download(fam["hf"] + name, path)
        S.verify_sha256(path, *hy4.PINS[name], own=bool(a.gguf_dir))
    data, _ = S.data_folder_for(a)
    pack = data / "packs" / ("hy4-" + hy4.MODEL.lower())
    S.step(6, "preparing Hy4's native tokenizer and serving configuration")
    # This is the same preparation routine exposed as a tool, not an extra
    # manual step the person has to run or a Qwen pack conversion.
    tools = str(S.ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from hy4.prepare import prepare
    prepared = prepare(folder / hy4.FILES[0], exe, pack, ctx,
                       4 if getattr(a, "threads", None) is None else a.threads,
                       ",".join(str(i) for i in range(len(devices))) if devices else None,
                       slots=slots, context_pool=pool,
                       prefill_chunk=getattr(a, "prefill_chunk", None),
                       mmproj=folder / hy4.PROJECTOR if vision else None,
                       mtp=mtp, native_caps=caps, vision_gpu=vision_word == "gpu",
                       vram_reserve_mib=getattr(a, "vram_reserve_mib", None))
    cfg = json.loads(prepared.read_text())
    cfg.update(cwd=str(S.ROOT), port=port, model_name="hy4-flash-q4_k_m",
               log=str(S.home() / "strata-hy4-q4_k_m.log"))
    if devices:
        cfg.update(gpu=devices, gpus_asked=True)
    for key in ("host", "api_key"):
        if getattr(a, key, None):
            cfg[key] = getattr(a, key)
    if getattr(a, "browser", None) is not None:
        cfg["open_browser"] = a.browser
    if vision and getattr(a, "vision_tokens", None):
        cfg["vision"]["max_tokens"] = a.vision_tokens
    S.step(7, "writing the Hy4 start script")
    cfg_path = S.home() / "strata-hy4-q4_k_m.json"
    S.write_setup_config(cfg_path, cfg, None, S.explicit_args(a))
    script = S.write_run_script("hy4-q4_k_m", cfg_path, port, cfg.get("open_browser") is not False)
    S.ok(f"start script: {script.name}")
    S.ok(f"Hy4: {slots} slots share {pool:,} KV tokens, request limit {ctx:,}; Engram remains in host memory")
    return 0 if a.no_start else S.start(cfg_path, port)
