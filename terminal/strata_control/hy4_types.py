"""The native Hy4 loader's artifact checks for ordinary 'use my model files' import."""
from __future__ import annotations

from strata_control.hy4 import ARCH, FAMILY, MODEL

MATRIX = frozenset({0, 1, 30, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 42})
EXPERT = frozenset({12, 14})


def problems(types: dict[str, int]) -> list[str]:
    """Refuse unknown encodings or missing defining tensors, before choosing an engine.

    The first catalog artifact is Q4_K_M: Q4_K gate/up, Q4_K/Q6_K down, Q8_0 dense,
    F32 scalars and an FP8 E4M3 table. Other artifact encodings stay unqualified.
    Native loading subsequently validates every shape and required layer tensor.
    """
    errors = []
    for name in ("token_embd.weight", "output.weight", "output_norm.weight", "output_hc_fn.weight",
                 "output_hc_base.weight", "output_hc_scale.weight", "per_layer_token_embd.weight"):
        if name not in types:
            errors.append(f"{name}: missing required Hy4 tensor")
    for name, kind in types.items():
        if name == "per_layer_token_embd.weight":
            if kind != 43:
                errors.append(f"{name}: requires FP8 E4M3 in the supported artifact")
        elif name.startswith("blk.42.") and name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")):
            if kind != 30:
                errors.append(f"{name}: requires BF16 experts in the pinned MTP block")
        elif name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight")):
            if kind != 12:
                errors.append(f"{name}: requires Q4_K gate/up in Q4_K_M")
        elif name.endswith("ffn_down_exps.weight"):
            if kind not in EXPERT:
                errors.append(f"{name}: requires Q4_K or Q6_K down in Q4_K_M")
        elif kind not in MATRIX:
            errors.append(f"{name}: unsupported native Hy4 tensor type {kind}")
    if not any(n.endswith("ffn_gate_exps.weight") for n in types):
        errors.append("missing Hy4 routed experts")
    return errors


def candidate(shards, issues: list[str], notes: list[str]) -> dict | None:
    from strata_control import gguf_preflight
    from strata_control.profiles import ProfileError
    types = {}
    try:
        for path in shards:
            types.update(gguf_preflight.tensor_types(path))
    except ProfileError as exc:
        issues.append(f"{shards[0].name}: {exc}")
        return None
    errors = problems(types)
    if errors:
        issues.extend(errors[:8])
        return None
    return {"family": FAMILY, "model": MODEL, "basis": ARCH + "_engine_types",
            "expected_expert_types": sorted({"Q4_K", *(["Q6_K"] if 14 in types.values() else []), *(["BF16"] if any(n.startswith("blk.42.") and "_exps.weight" in n for n in types) else [])}),
            "expected_tensor_count": len(types),
            "setup_args_hint": ["--family", FAMILY, "--model", MODEL, "--gguf-dir", str(shards[0].parent)]}
