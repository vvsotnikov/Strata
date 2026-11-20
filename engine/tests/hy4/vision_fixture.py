"""Independent NumPy Hy4 vision oracle: non-square positions, separate QKV, conv merger and row newlines."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np


def run(engine: Path, output: Path, gpu: bool = False, resize: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4216)
    d, ff, heads, patch, merge, out = 8, 12, 2, 2, 2, 6
    height, width, grid = 16 if resize else 12, 8, 4
    hidden = d * merge * merge
    meta = {
        "general.architecture": "clip",
        "clip.projector_type": "hy_v4_flash",
        "clip.vision.embedding_length": d,
        "clip.vision.attention.head_count": heads,
        "clip.vision.feed_forward_length": ff,
        "clip.vision.block_count": 1,
        "clip.vision.patch_size": patch,
        "clip.vision.spatial_merge_size": merge,
        "clip.vision.projection_dim": out,
        "clip.vision.image_min_pixels": 16,
        "clip.vision.image_max_pixels": 1024,
        "clip.vision.attention.layer_norm_epsilon": 1e-5,
        "clip.vision.image_mean": [0.5, 0.5, 0.5],
        "clip.vision.image_std": [0.5, 0.5, 0.5],
    }
    weights, shapes = {}, {}

    def weight(name, shape, norm=False):
        a = (rng.normal(0, 0.1, shape[::-1]) + (1 if norm else 0)).astype("<f4")
        weights[name] = a
        shapes[name] = shape
        return a

    weight("v.patch_embd.weight", [patch, patch, 3, d])
    weight("v.patch_embd.bias", [d])
    weight("v.position_embd.weight", [d, grid * grid])
    for norm in ("ln1", "ln2"):
        weight(f"v.blk.0.{norm}.weight", [d], True)
        weight(f"v.blk.0.{norm}.bias", [d])
    for name in ("attn_q", "attn_k", "attn_v", "attn_out"):
        weight(f"v.blk.0.{name}.weight", [d, d])
        weight(f"v.blk.0.{name}.bias", [d])
    weight("v.blk.0.ffn_up.weight", [d, ff])
    weight("v.blk.0.ffn_up.bias", [ff])
    weight("v.blk.0.ffn_down.weight", [ff, d])
    weight("v.blk.0.ffn_down.bias", [d])
    weight("mm.pre_norm.weight", [d], True)
    weight("mm.0.weight", [merge, merge, d, hidden])
    weight("mm.0.bias", [hidden])
    weight("mm.2.weight", [1, 1, hidden, out])
    weight("mm.2.bias", [out])
    weight("v.image_newline", [out])

    def text(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    def value(v):
        if isinstance(v, str):
            return 8, text(v)
        if isinstance(v, int):
            return 4, struct.pack("<I", v)
        if isinstance(v, float):
            return 6, struct.pack("<f", v)
        t = value(v[0])[0]
        return 9, struct.pack("<IQ", t, len(v)) + b"".join(value(x)[1] for x in v)

    header = struct.pack("<IIQQ", 0x46554747, 3, len(weights), len(meta))
    for name, v in meta.items():
        ty, raw = value(v)
        header += text(name) + struct.pack("<I", ty) + raw
    payload = b""
    for name, a in weights.items():
        dims = shapes[name]
        header += (
            text(name)
            + struct.pack("<I", len(dims))
            + struct.pack("<" + "Q" * len(dims), *dims)
            + struct.pack("<IQ", 0, len(payload))
        )
        payload += a.tobytes()
        payload += b"\0" * (-len(payload) % 32)
    header += b"\0" * (-len(header) % 32)
    model = output / "vision-tiny.gguf"
    model.write_bytes(header + payload)
    input_h, input_w = (14, 10) if resize else (height, width)
    rgb = rng.integers(0, 256, (input_h, input_w, 3), dtype="u1")
    image = output / "image.ppm"
    image.write_bytes(f"P6\n{input_w} {input_h}\n255\n".encode() + rgb.tobytes())
    if resize:
        from PIL import Image

        # Python rounds 10/4=2.5 down to even 2 and 14/4=3.5 up to even 4.
        # Pillow is an independent oracle for the native bicubic resampler.
        assert (width, height) == (round(input_w / 4) * 4, round(input_h / 4) * 4)
        rgb = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BICUBIC))
    pixels = (rgb.transpose(2, 0, 1).astype("f4") / np.float32(255) - 0.5) / 0.5
    pw, ph = width // patch, height // patch
    patches = pixels.reshape(3, ph, patch, pw, patch).transpose(1, 3, 0, 2, 4).reshape(ph * pw, -1)
    x = patches @ weights["v.patch_embd.weight"].reshape(d, -1).T + weights["v.patch_embd.bias"]
    table = weights["v.position_embd.weight"].reshape(grid, grid, d)
    for y in range(ph):
        for xx in range(pw):
            fy = (y + 0.5) / ((ph + 0.1) / grid) - 0.5
            fx = (xx + 0.5) / ((pw + 0.1) / grid) - 0.5
            y0, y1 = np.clip([np.floor(fy), np.floor(fy) + 1], 0, grid - 1).astype(int)
            x0, x1 = np.clip([np.floor(fx), np.floor(fx) + 1], 0, grid - 1).astype(int)
            wy, wx = np.clip(fy - y0, 0, 1), np.clip(fx - x0, 0, 1)
            x[y * pw + xx] += (
                (1 - wy) * (1 - wx) * table[y0, x0]
                + (1 - wy) * wx * table[y0, x1]
                + wy * (1 - wx) * table[y1, x0]
                + wy * wx * table[y1, x1]
            )
    traces = {"vision_patch--1": x.copy()}

    def norm(a, w, b=None, rms=False):
        mean = 0 if rms else a.astype("f8").mean(-1, keepdims=True).astype("f4")
        a = a - mean
        z = a / np.sqrt((a.astype("f8") ** 2).mean(-1, keepdims=True) + 1e-5).astype("f4") * w
        return z if b is None else z + b

    def linear(a, name):
        return a @ weights[name + ".weight"].reshape(-1, a.shape[-1]).T + weights[name + ".bias"]

    def gelu(a):
        import math

        return (0.5 * a * (1 + np.vectorize(math.erf)(a * np.float32(2**-0.5)))).astype("f4")

    n = norm(x, weights["v.blk.0.ln1.weight"], weights["v.blk.0.ln1.bias"])
    q, k, v = [
        linear(n, "v.blk.0." + p).reshape(ph * pw, heads, d // heads).transpose(1, 0, 2)
        for p in ("attn_q", "attn_k", "attn_v")
    ]
    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(d // heads)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    attention = (probs @ v).transpose(1, 0, 2).reshape(ph * pw, d)
    x += linear(attention, "v.blk.0.attn_out")
    n = norm(x, weights["v.blk.0.ln2.weight"], weights["v.blk.0.ln2.bias"])
    x += linear(gelu(linear(n, "v.blk.0.ffn_up")), "v.blk.0.ffn_down")
    traces["vision_layer-0"] = x.copy()
    x = norm(x, weights["mm.pre_norm.weight"], rms=True).reshape(ph, pw, d)
    # Convolution weight flattening is channel, y-tap, x-tap, unlike a Qwen merger.
    packed = (
        x.reshape(ph // merge, merge, pw // merge, merge, d).transpose(0, 2, 4, 1, 3).reshape(-1, d * merge * merge)
    )
    projected = linear(gelu(linear(packed, "mm.0")), "mm.2")
    traces["vision_projected--1"] = projected.copy()
    expected = np.concatenate(
        [
            projected.reshape(ph // merge, pw // merge, out),
            np.broadcast_to(weights["v.image_newline"], (ph // merge, 1, out)),
        ],
        axis=1,
    )
    sve = output / "actual.sve"
    trace_dir = output / "trace"
    command = [
        str(engine.resolve()),
        "--mmproj",
        str(model.resolve()),
        "--image",
        str(image.resolve()),
        "--output",
        str(sve.resolve()),
        "--dump-dir",
        str(trace_dir.resolve()),
        "--precision",
        "f32",
    ]
    if gpu:
        command += ["--gpu"]
    completed = subprocess.run(command, text=True, capture_output=True, check=True, timeout=120)
    data = sve.read_bytes()
    magic, count, nx, ny, dim = struct.unpack("<5i", data[:20])
    assert (magic, count, nx, ny, dim) == (
        0x31455653,
        (pw // merge + 1) * (ph // merge),
        pw // merge + 1,
        ph // merge,
        out,
    )
    actual = np.frombuffer(data[20:], "<f4").reshape(expected.shape)
    error = float(abs(actual - expected).max())
    assert error < 2e-4, error
    errors = {}
    for name, wanted in traces.items():
        got = np.fromfile(trace_dir / (name + ".f32"), "<f4").reshape(wanted.shape)
        errors[name] = float(abs(got - wanted).max())
        assert errors[name] < 2e-4, (name, errors[name])
    report = {
        "backend": "cuda-f32" if gpu else "cpu",
        "bicubic_resize": resize,
        "embedding_max_abs": error,
        "stages": errors,
        "width_with_newline": nx,
        "height": ny,
        "dim": dim,
        "stdout": completed.stdout,
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--resize", action="store_true")
    a = p.parse_args()
    run(a.engine, a.output, a.gpu, a.resize)
