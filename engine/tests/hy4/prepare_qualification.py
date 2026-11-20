"""Build exact-size prompts from real source excerpts using the server's CPU token counter.

No inference requests, padding, repeated source, or fake generation are used.
Run after the serving frontend is available, before performance measurements.
"""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--base-url", help="CPU-only live count endpoint (authoritative)")
p.add_argument(
    "--tokenizer-dir",
    type=pathlib.Path,
    help="prepare Hy4 inputs offline; verify live usage before accepting results",
)
p.add_argument("--model", required=True)
p.add_argument(
    "--source-root",
    type=pathlib.Path,
    default=pathlib.Path(__file__).resolve().parents[3],
)
p.add_argument("--output", type=pathlib.Path, required=True)
p.add_argument("--prompt-tokens", type=int, default=8192)
a = p.parse_args()
U = a.source_root
TARGET = a.prompt_tokens
if not a.base_url and not a.tokenizer_dir:
    p.error("provide --base-url or --tokenizer-dir")
if a.base_url and a.tokenizer_dir:
    p.error("choose either the live counter or the offline tokenizer")
if a.tokenizer_dir:
    sys.path[:0] = [str(U / "terminal"), str(U / "terminal/tools")]
    from serve.frontend import ChatTemplate, use_family
    from strata_tokenizer import Tokenizer

    config = json.loads((a.tokenizer_dir / "tokenizer.json").read_text())
    if config.get("pre") != "hy_v4":
        p.error("offline mode currently supports only the Hy4 tokenizer")
    vocab = json.loads((a.tokenizer_dir / "vocab.json").read_text())
    tokens = [t for t, _ in sorted(vocab.items(), key=lambda item: item[1])]
    # Match the serving loader exactly: lines beginning '#' can be real merges
    # (for example C++ preprocessor/comment prefixes), not just a file header.
    merges = (a.tokenizer_dir / "merges.txt").read_text().split("\n")
    tokenizer = Tokenizer(
        tokens,
        merges,
        json.loads((a.tokenizer_dir / "token_type.json").read_text()),
        "hy_v4",
        config.get("special_ids"),
    )
    template = ChatTemplate(a.tokenizer_dir / "chat_template.jinja")
    use_family("hy4")
specs = [
    ("scheduler", ["server.cpp"]),
    ("device-runtime", ["device.cu"]),
    ("expert-cache", ["device_experts.cu"]),
    ("kernel-correctness", ["device_kernels.cuh", "device_experts.cu"]),
    ("cpu-oracle", ["engine.cpp", "math.cpp"]),
]


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def count(text):
    if a.tokenizer_dir:
        rendered = template.render(
            [{"role": "user", "content": text}], reasoning_effort="max"
        )
        return len(tokenizer.encode(rendered, parse_special=True))
    b = {
        "model": a.model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 16000},
        "output_config": {"effort": "max"},
    }
    req = urllib.request.Request(
        a.base_url.rstrip("/") + "/v1/messages/count_tokens",
        data=json.dumps(b).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)["input_tokens"]


rows = []
for case, files in specs:
    parts = []
    provenance = []
    for name in files:
        p = U / "engine/src/hy4" / name
        s = p.read_text()
        parts.append(f"// Source file: engine/src/hy4/{name}\n" + s)
        provenance.append(
            {
                "path": str(p.relative_to(U)),
                "sha256": digest(s),
                "bytes": len(s.encode()),
            }
        )
    source = "\n".join(parts)
    intro = f"Review task {case}. Audit this real inference-engine implementation for correctness, concurrency, numerical stability, memory lifetime, and exposed performance bottlenecks. Explain concrete execution paths and counterexamples; do not invent missing code. Propose prioritized fixes and specific regression tests. Analyze the entire excerpt carefully and provide a detailed engineering review.\n\n```cpp\n"
    tail = "\n```\n\nReview the implementation above. Distinguish confirmed defects from questions requiring measurement. Include specific triggering conditions, severity, repair options, and validation for each finding."
    lo, hi = 0, len(source)
    best = None
    while lo <= hi:
        n = (lo + hi) // 2
        text = intro + source[:n] + tail
        c = count(text)
        if c == TARGET:
            best = (n, text, c)
            break
        if c < TARGET:
            lo = n + 1
        else:
            hi = n - 1
    if best is None:
        for n in range(max(0, hi - 100), min(len(source), lo + 100) + 1):
            text = intro + source[:n] + tail
            c = count(text)
            if c == TARGET:
                best = (n, text, c)
                break
    if best is None:
        # Keep meaningful task text; changing a word avoids BPE boundaries that skip the exact count.
        for variant in (
            " Include reproducible tests.",
            " Explain all assumptions.",
            " Discuss edge cases.",
            " Prioritize risks.",
        ):
            tail2 = tail + variant
            lo, hi = 0, len(source)
            while lo <= hi:
                n = (lo + hi) // 2
                text = intro + source[:n] + tail2
                c = count(text)
                if c == TARGET:
                    best = (n, text, c)
                    break
                if c < TARGET:
                    lo = n + 1
                else:
                    hi = n - 1
            if best is not None:
                break
    if best is None:
        raise RuntimeError(
            (case, "unable to fit source prefix", lo, hi, count(intro + source + tail))
        )
    n, text, c = best
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": text}],
        "reasoning_effort": "max",
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 64,
        "min_p": 0.0,
        "seed": 42,
        "stream": True,
    }
    rows.append(
        {
            "case": case,
            "body": body,
            "prompt_tokens": c,
            "counter": "offline Hy4 tokenizer; requires actual generation usage verification"
            if a.tokenizer_dir
            else "live POST /v1/messages/count_tokens; same messages and max effort; no generation",
            "source_files": provenance,
            "source_prefix_characters": n,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=U, text=True
            ).strip(),
            "content_sha256": digest(text),
            "padding": False,
        }
    )
    print(case, n, c, flush=True)
    p = a.output
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2) + "\n")
print(p, flush=True)
