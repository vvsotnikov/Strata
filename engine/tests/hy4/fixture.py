"""Independent NumPy forward for a small Hy4 GGUF, including FP8 Engram, SWA wrap and sparse top-k."""

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--engine", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument(
    "--reference",
    action="store_true",
    help="Run image MRoPE and independent MTP reference driver",
)
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(5315)
D = 32
HC = 4
N = 3
H = 2
V = 128
R = 16
K = 32
F = 32
L = 32
IH = 2
ID = 32
P = 2
TOP = 8
WIN = 4
E = 4
USE = 2
meta = {
    "general.architecture": "hy_v4_flash",
    "hy_v4_flash.block_count": 4 if args.reference else 3,
    "hy_v4_flash.nextn_predict_layers": int(args.reference),
    "hy_v4_flash.context_length": 64,
    "hy_v4_flash.embedding_length": D,
    "hy_v4_flash.vocab_size": V,
    "hy_v4_flash.hyper_connection.count": HC,
    "hy_v4_flash.attention.q_lora_rank": 32,
    "hy_v4_flash.attention.kv_lora_rank": K,
    "hy_v4_flash.attention.key_length_mla": 32,
    "hy_v4_flash.attention.value_length_mla": 32,
    "hy_v4_flash.rope.dimension_count": R,
    "hy_v4_flash.feed_forward_length": 64,
    "hy_v4_flash.expert_feed_forward_length": F,
    "hy_v4_flash.expert_latent_length": L,
    "hy_v4_flash.expert_count": E,
    "hy_v4_flash.expert_used_count": USE,
    "hy_v4_flash.leading_dense_block_count": 1,
    "hy_v4_flash.expert_shared_count": 1,
    "hy_v4_flash.expert_gating_func": 2,
    "hy_v4_flash.expert_weights_norm": True,
    "hy_v4_flash.attention.head_count_kv": 1,
    "hy_v4_flash.attention.sliding_window": WIN,
    "hy_v4_flash.attention.indexer.key_length": ID,
    "hy_v4_flash.attention.indexer.head_count": IH,
    "hy_v4_flash.attention.indexer.top_k": TOP,
    "hy_v4_flash.attention.indexer.kpool": P,
    "hy_v4_flash.attention.indexer.kpool_select_tail": True,
    "hy_v4_flash.attention.layer_norm_rms_epsilon": 1e-5,
    "hy_v4_flash.hyper_connection.epsilon": 1e-6,
    "hy_v4_flash.hyper_connection.magnitude": 2.0,
    "hy_v4_flash.expert_weights_scale": 3.998,
    "hy_v4_flash.rope.freq_base": 10000000.0,
    "hy_v4_flash.rope.freq_base_swa": 10000.0,
    "hy_v4_flash.attention.head_count": [2] * (N + int(args.reference)),
    "hy_v4_flash.attention.sliding_window_pattern": [False, True, False]
    + ([False] if args.reference else []),
    "hy_v4_flash.attention.indexer.types": [True, False, True],
    "hy_v4_flash.swiglu_clamp_exp": [0.1] * (N + int(args.reference)),
    "hy_v4_flash.ple.ngram_size": 3,
    "hy_v4_flash.ple.heads_per_ngram": 2,
    "hy_v4_flash.embedding_length_per_layer_input": 32,
    "hy_v4_flash.engram.pad_token_id": 0,
    "hy_v4_flash.ple.layers": [1],
    "hy_v4_flash.ple.layer_multipliers": [
        81853575799317,
        39690085502883,
        24229915018165,
    ],
    "hy_v4_flash.ple.head_offsets": [0, 5, 12, 23],
    "hy_v4_flash.ple.head_vocab_sizes": [5, 7, 11, 13],
    "hy_v4_flash.engram.token_map": list(range(V)),
}
w = {}
raw = {}
shapes = {}


def add(name, shape, norm=False):
    a = (rng.normal(0, 0.04, size=shape[::-1]) + (1 if norm else 0)).astype("<f4")
    w[name] = a
    raw[name] = (0, a.tobytes())
    shapes[name] = shape


for name, shape in [
    ("token_embd.weight", [D, V]),
    ("output.weight", [D, V]),
    ("output_hc_fn.weight", [D * HC, HC]),
    ("output_hc_base.weight", [HC]),
    ("output_hc_scale.weight", [1]),
]:
    add(name, shape)
add("output_norm.weight", [D], True)
for l in range(N + int(args.reference)):

    def put(n, s, norm=False, l=l):
        add(f"blk.{l}.{n}", s, norm)

    for n in (
        "attn_norm.weight",
        "ffn_norm.weight",
        "post_attention_norm.weight",
        "post_ffw_norm.weight",
    ):
        put(n, [D], True)
    for n, s in [
        ("attn_q_a.weight", [D, 32]),
        ("attn_q_b.weight", [32, H * 32]),
        ("attn_kv_a_mqa.weight", [D, K + R]),
        ("attn_k_b.weight", [16, K, H]),
        ("attn_v_b.weight", [K, 32, H]),
        ("attn_gate.weight", [D, H * 32]),
        ("attn_output.weight", [H * 32, D]),
        ("attn_sinks.weight", [H]),
    ]:
        put(n, s)
    put("attn_q_a_norm.weight", [32], True)
    put("attn_kv_a_norm.weight", [K], True)
    for kind in ("attn", "ffn") if l < N else ():
        for n, s in [("fn", [D * HC, HC * 2]), ("base", [HC * 2]), ("scale", [2])]:
            put(f"hc_{kind}_{n}.weight", s)
    if l == N:
        for n in ("enorm", "hnorm", "shared_head_norm"):
            put("nextn." + n + ".weight", [D], True)
        put("nextn.eh_proj.weight", [D * 2, D])
    if l != 1:
        for n, s in [
            ("indexer.attn_k.weight", [D, ID]),
            ("indexer.attn_q_b.weight", [32, IH * ID]),
            ("indexer.proj.weight", [D, IH]),
            ("indexer_compressor_value.weight", [ID, ID]),
            ("indexer_compressor_gate.weight", [ID, ID]),
            ("indexer_compressor_ape.weight", [ID, P]),
        ]:
            put(n, s)
        for n in ("indexer.k_norm.weight", "indexer_compressor_norm.weight"):
            put(n, [ID], True)
    if l == 1:
        for n, s in [
            ("engram_kv.weight", [128, D * (HC + 1)]),
            ("engram_kv_b.weight", [D, HC + 1]),
            ("engram_gate.weight", [D, HC]),
            ("engram_hc_scale.weight", [D, HC]),
        ]:
            put(n, s)
        put("engram_lookup_norm.weight", [32, 4], True)
        put("engram_pre_gamma.weight", [D], True)
    if l == 0:
        for n, s in [
            ("ffn_up.weight", [D, 64]),
            ("ffn_gate.weight", [D, 64]),
            ("ffn_down.weight", [64, D]),
        ]:
            put(n, s)
    else:
        for n, s in [
            ("ffn_gate_inp.weight", [D, E]),
            ("exp_probs_b.bias", [E]),
            ("ffn_routed_down.weight", [D, L]),
            ("ffn_routed_up.weight", [L, D]),
            ("ffn_gate_exps.weight", [L, F, E]),
            ("ffn_up_exps.weight", [L, F, E]),
            ("ffn_down_exps.weight", [F, L, E]),
            ("ffn_gate_shexp.weight", [D, F]),
            ("ffn_up_shexp.weight", [D, F]),
            ("ffn_down_shexp.weight", [F, D]),
        ]:
            put(n, s)
b = rng.integers(0x20, 0x50, size=(36, 32), dtype=np.uint8)
e = (b >> 3) & 15
m = b & 7
table = np.where(e == 0, m * 2.0**-9, (1 + m / 8) * np.exp2(e.astype(int) - 7)).astype(
    "f4"
)
w["per_layer_token_embd.weight"] = table
raw["per_layer_token_embd.weight"] = (43, b.tobytes())
shapes["per_layer_token_embd.weight"] = [32, 36]


def st(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def val(v):
    if isinstance(v, str):
        return 8, st(v)
    if isinstance(v, bool):
        return 7, struct.pack("<?", v)
    if isinstance(v, float):
        return 6, struct.pack("<f", v)
    if isinstance(v, int):
        return 10, struct.pack("<Q", v)
    t, _ = val(v[0])
    return 9, struct.pack("<IQ", t, len(v)) + b"".join(val(x)[1] for x in v)


header = struct.pack("<IIQQ", 0x46554747, 3, len(raw), len(meta))
for k, v in meta.items():
    t, b = val(v)
    header += st(k) + struct.pack("<I", t) + b
payload = b""
for name, (ty, b) in raw.items():
    shape = shapes[name]
    header += (
        st(name)
        + struct.pack("<I", len(shape))
        + struct.pack("<" + "Q" * len(shape), *shape)
        + struct.pack("<IQ", ty, len(payload))
    )
    payload += b
    payload += b"\0" * ((-len(payload)) % 32)
header += b"\0" * ((-len(header)) % 32)
model = out / "hy4-tiny.gguf"
model.write_bytes(header + payload)


def norm(x, weight=None, eps=1e-5):
    z = x / np.sqrt(np.mean(x.astype("f8") ** 2, axis=-1, keepdims=True) + eps).astype(
        "f4"
    )
    return z if weight is None else z * weight


def sig(x):
    return 1 / (1 + np.exp(-x))


def rot(x, pos, base):
    x = x.copy()
    if not np.isscalar(pos):
        pos = np.asarray(pos)[
            [
                1 if i < 30 and i % 3 == 1 else 2 if i < 30 and i % 3 == 2 else 0
                for i in range(R // 2)
            ]
        ]
    angle = pos * base ** (-np.arange(R // 2, dtype="f4") * 2 / R)
    a = x[..., -R::2].copy()
    b = x[..., -R + 1 :: 2].copy()
    x[..., -R::2] = a * np.cos(angle) - b * np.sin(angle)
    x[..., -R + 1 :: 2] = a * np.sin(angle) + b * np.cos(angle)
    return x


def linear(name, x, plane=None):
    return (w[name] if plane is None else w[name][plane]) @ x


cache = [
    {"kv": [], "v": [], "g": [], "pooled": []} for _ in range(N + int(args.reference))
]
hist, dhist, rhist, drhist = [], [], [], []
last_reference_hidden = None
traces = {}


def forward(token, embedding=None, position=None, previous_hidden=None):
    global last_reference_hidden
    draft = previous_hidden is not None
    active_hist, active_positions = (dhist, drhist) if draft else (hist, rhist)
    pos = len(active_hist)
    position = pos if position is None else position
    embedding = w["token_embd.weight"][token] if embedding is None else embedding
    if draft:
        pre = f"blk.{N}.nextn."
        joined = np.concatenate(
            [
                norm(embedding, w[pre + "enorm.weight"]),
                norm(previous_hidden, w[pre + "hnorm.weight"]),
            ]
        )
        x = linear(pre + "eh_proj.weight", joined)[None, :]
    else:
        x = np.tile(embedding, (HC, 1))
    for l in [N] if draft else range(N):
        pre = f"blk.{l}."
        at = lambda s, pre=pre: w[pre + s]
        mv = lambda s, z, pre=pre: linear(pre + s, z)
        nm = lambda z, s, at=at: norm(z, at(s))
        if l == 1 and token >= 0:
            mult = meta["hy_v4_flash.ple.layer_multipliers"]
            tokens = [token] + [
                max(0, hist[-i]) if len(hist) >= i else 0 for i in (1, 2)
            ]
            rolling = (tokens[0] * mult[0]) & ((1 << 64) - 1)
            ix = []
            for i in (1, 2):
                rolling ^= (tokens[i] * mult[i]) & ((1 << 64) - 1)
                for h in range(2):
                    col = (i - 1) * 2 + h
                    ix.append(
                        rolling % meta["hy_v4_flash.ple.head_vocab_sizes"][col]
                        + meta["hy_v4_flash.ple.head_offsets"][col]
                    )
            rows = norm(table[ix], at("engram_lookup_norm.weight"), 2**-23).flatten()
            kv = mv("engram_kv.weight", rows).reshape(HC + 1, D) + at(
                "engram_kv_b.weight"
            )
            score = np.sum(
                norm(x, eps=2**-23)
                * norm(kv[:HC], eps=2**-23)
                * at("engram_gate.weight"),
                axis=1,
            ) / np.sqrt(D)
            gate = sig(np.copysign(np.sqrt(np.maximum(abs(score), 1e-6)), score))
            x += norm(kv[HC], at("engram_pre_gamma.weight"), 2**-23)[None, :] * (
                gate[:, None] + at("engram_hc_scale.weight")
            )

        def mix(kind, mv=mv, at=at, x=x):
            if draft:
                return x[0].copy(), np.ones(1, "f4")
            y = mv("hc_" + kind + "_fn.weight", norm(x.flatten()))
            base = at("hc_" + kind + "_base.weight")
            scale = at("hc_" + kind + "_scale.weight")
            a = sig(y[:HC] * scale[0] + base[:HC]) + 1e-6
            b = sig(y[HC:] * scale[1] + base[HC:]) * 2 + 1e-6
            return (a[:, None] * x).sum(axis=0), b

        z, post = mix("attn")
        z = nm(z, "attn_norm.weight")
        q = nm(mv("attn_q_a.weight", z), "attn_q_a_norm.weight")
        qr = mv("attn_q_b.weight", q).reshape(H, 32)
        kv = mv("attn_kv_a_mqa.weight", z)
        kv[:K] = nm(kv[:K], "attn_kv_a_norm.weight")
        kv = rot(kv, position, 10000 if l == 1 else 10000000)
        c = cache[l]
        c["kv"].append(kv)
        if l != 1:
            ik = nm(mv("indexer.attn_k.weight", z), "indexer.k_norm.weight")
            c["v"].append(mv("indexer_compressor_value.weight", ik))
            c["g"].append(mv("indexer_compressor_gate.weight", ik))
            if pos % P == P - 1:
                gate = np.stack(c["g"][-P:]) + at("indexer_compressor_ape.weight")
                probs = np.exp(gate - gate.max(axis=0))
                probs /= probs.sum(axis=0)
                pooled = (probs * np.stack(c["v"][-P:])).sum(axis=0)
                c["pooled"].append(
                    rot(
                        nm(pooled, "indexer_compressor_norm.weight"),
                        active_positions[pos - P + 1],
                        10000000,
                    )
                )
            iq = rot(
                mv("indexer.attn_q_b.weight", q).reshape(IH, ID), position, 10000000
            )
            weights = mv("indexer.proj.weight", z) / np.sqrt(ID * IH)
            groups = pos // P
            scores = [
                (np.maximum(iq @ k, 0) * weights).sum() for k in c["pooled"][:groups]
            ]
            chosen = sorted(range(groups), key=lambda i: (-scores[i], i))[
                : TOP // P - 1
            ]
            selected = sorted(
                [g * P + t for g in chosen for t in range(P)]
                + list(range(pos - pos % P, pos + 1))
            )
        else:
            selected = list(range(max(0, pos - WIN + 1), pos + 1))
        out = []
        for h in range(H):
            a = rot(
                np.concatenate(
                    [linear(pre + "attn_k_b.weight", qr[h, :16], h), qr[h, 16:]]
                ),
                position,
                10000 if l == 1 else 10000000,
            )
            keys = np.stack(c["kv"])[selected]
            scores = keys @ a / np.sqrt(32)
            sink = at("attn_sinks.weight")[h]
            mx = max(scores.max(), sink)
            pr = np.exp(scores - mx)
            den = pr.sum() + np.exp(sink - mx)
            latent = (pr[:, None] * keys[:, :K]).sum(axis=0) / den
            out.extend(linear(pre + "attn_v_b.weight", latent, h))
        att = nm(
            mv("attn_output.weight", np.array(out) * sig(mv("attn_gate.weight", z))),
            "post_attention_norm.weight",
        )
        traces[f"{pos}-{l}-q_resid"] = q
        traces[f"{pos}-{l}-attn_out"] = att
        x += post[:, None] * att[None, :]
        z, post = mix("ffn")
        z = nm(z, "ffn_norm.weight")

        def glu(a, b, limit=0):
            if limit:
                a = np.minimum(a, limit)
                b = np.clip(b, -limit, limit)
            return a * sig(a) * b

        if l == 0:
            f = mv(
                "ffn_down.weight", glu(mv("ffn_gate.weight", z), mv("ffn_up.weight", z))
            )
        else:
            probs = sig(mv("ffn_gate_inp.weight", z))
            scores = probs + at("exp_probs_b.bias")
            pick = np.argsort(-scores, kind="stable")[:USE]
            weights = probs[pick] / probs[pick].sum() * 3.998
            latent = mv("ffn_routed_down.weight", z)
            summ = np.zeros(L, "f4")
            for e, weight in zip(pick, weights):
                summ += (
                    linear(
                        pre + "ffn_down_exps.weight",
                        glu(
                            linear(pre + "ffn_gate_exps.weight", latent, e),
                            linear(pre + "ffn_up_exps.weight", latent, e),
                            0.1,
                        ),
                        e,
                    )
                    * weight
                )
            f = mv("ffn_routed_up.weight", summ) + mv(
                "ffn_down_shexp.weight",
                glu(mv("ffn_gate_shexp.weight", z), mv("ffn_up_shexp.weight", z)),
            )
        f = nm(f, "post_ffw_norm.weight")
        x += post[:, None] * f[None, :]
        traces[f"{pos}-{l}-ffn_out"] = f
        traces[f"{pos}-{l}-l_out"] = x.copy()
    if draft:
        hidden = norm(x[0], w[f"blk.{N}.nextn.shared_head_norm.weight"])
    else:
        mixes = linear("output_hc_fn.weight", norm(x.flatten()))
        gates = (
            sig(mixes * w["output_hc_scale.weight"] + w["output_hc_base.weight"]) + 1e-6
        )
        hidden = norm((x * gates[:, None]).sum(axis=0), w["output_norm.weight"])
    traces[f"{pos}--1-result_norm"] = hidden
    active_hist.append(token)
    active_positions.append(position)
    last_reference_hidden = hidden.copy()
    return linear("output.weight", hidden)


if args.reference:
    records = bytearray(struct.pack("<II", 12, D))
    target_logits, target_hidden, draft_logits, draft_hidden = [], [], [], []
    for i in range(12):
        image = 2 <= i < 6
        token = -1 if image else i + 1
        if image:
            position = (2, 2 + (i - 2) // 2, 2 + (i - 2) % 2)
            embedding = w["token_embd.weight"][30 + i] * 1.7
        else:
            p = i if i < 2 else i - 2
            position = (p, p, p)
            embedding = None
        next_position = (2 if i < 5 else 4) if image else position[0] + 1
        target_logits.append(forward(token, embedding, position))
        hidden = last_reference_hidden.copy()
        target_hidden.append(hidden)
        candidate = -1 if image else i + 20
        draft_position = (i + 1, i + 3, i + 5)
        draft_logits.append(
            forward(candidate, embedding if image else None, draft_position, hidden)
        )
        draft_hidden.append(last_reference_hidden.copy())
        records += struct.pack("<i4q", token, *position, next_position)
        records += (
            np.zeros(D, "f4") if embedding is None else embedding.astype("f4")
        ).tobytes()
        records += struct.pack("<i3q", candidate, *draft_position)
    (out / "reference-input.bin").write_bytes(records)
    wanted = {
        "target_logits": target_logits,
        "target_hidden": target_hidden,
        "draft_logits": draft_logits,
        "draft_hidden": draft_hidden,
    }
    subprocess.run(
        [
            str(args.engine.resolve()),
            str(model),
            str(out / "reference-input.bin"),
            str(out),
        ],
        check=True,
    )
    errors = {}
    for name, values in wanted.items():
        expected = np.stack(values).astype("f4")
        expected.tofile(out / (name + "-expected.f32"))
        actual = np.fromfile(out / (name + ".f32"), "f4").reshape(expected.shape)
        errors[name] = float(np.max(abs(actual - expected)))
        assert errors[name] < 2e-4, (name, errors[name])
    report = {
        "rows": 12,
        "image_rows": 4,
        "mtp_rows": 12,
        "crossed_mtp_sparse_selection": TOP,
        "maximum_errors": errors,
    }
    (out / "reference-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0)

tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
expected = np.stack([forward(t) for t in tokens])
expected.astype("f4").tofile(out / "expected.f32")
exe = args.engine.resolve()
subprocess.run(
    [
        str(exe),
        "--model",
        str(model),
        "--max-context",
        "64",
        "--tokens",
        ",".join(map(str, tokens)),
        "--dump-logits",
        str(out / "actual.f32"),
        "--dump-dir",
        str(out / "trace"),
    ],
    check=True,
)
actual = np.fromfile(out / "actual.f32", "f4").reshape(expected.shape)
errors = {}
for name, e in traces.items():
    a = np.fromfile(out / "trace" / f"{name}.f32", "f4").reshape(e.shape)
    err = float(np.max(abs(a - e)) / (1 + np.max(abs(e))))
    errors[name] = err
    if err > 1e-4:
        print("DIVERGENCE", name, err, float(np.max(abs(a - e))))
        raise SystemExit(1)
err = float(np.max(abs(actual - expected)))
assert err < 1e-4, err
report = {
    "tokens": len(tokens),
    "layers": 3,
    "crossed_swa_window": WIN,
    "crossed_sparse_selection": TOP,
    "all_stage_normalized_max_error": max(errors.values()),
    "logit_absolute_max_error": err,
    "top1_equal": bool(np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))),
}
(out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

# QUIT must exit even with stdin still open, and errors must not poison the next request.
p = subprocess.Popen(
    [str(exe), "--model", str(model), "--max-context", "64", "--serve"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)
ready = p.stdout.readline()
while ready.startswith("INFO "):
    ready = p.stdout.readline()
assert ready.startswith("READY 64 stop"), ready
p.stdin.write(
    "GEN 1 top_k=1.5 1\nGEN 1 temperature=nan 1\n"
    "GEN 1 temperature=1 penalty_repeat=1e-42 penalty_last_n=64 pen_from=0 seed=1 1,2,3,4,5,6,7,8,9,10\n"
    "GEN 2 temperature=0 1,2\nQUIT\n"
)
p.stdin.flush()
p.wait(timeout=15)
lines = p.stdout.read().splitlines()
p.stdin.close()
assert p.returncode == 0, lines
assert sum(s.startswith("ERR ") for s in lines) == 3, lines
assert any("sampling penalties overflowed logits" in s for s in lines), lines
assert sum(s.startswith("T ") for s in lines) == 2, lines
assert any(s.startswith("DONE 2 2 ") for s in lines), lines
