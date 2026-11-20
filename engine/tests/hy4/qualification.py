"""Controlled OpenAI-API qualification; never starts/stops an engine.

Outputs preserve early EOS, raw timings, payload hashes and streamed reasoning.
A capped request is a speed sample, not automatically a passed quality answer.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha(value):
    raw = (
        value
        if isinstance(value, bytes)
        else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    )
    return hashlib.sha256(raw).hexdigest()


def get_json(url, headers):
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=15
        ) as response:
            return json.load(response)
    except (OSError, ValueError) as error:
        return {"error": str(error)}


def sample(base_url, body, name, headers, timeout, gate=None):
    if gate is not None:
        gate.wait(timeout=30)
    begin = time.monotonic()
    first = None
    result = {
        "name": name,
        "request": body,
        "payload_sha256": sha(body),
        "start_monotonic": begin,
        "content": "",
        "reasoning": "",
        "finish_reason": None,
        "usage": None,
        "timings": None,
        "events": [],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                event = json.loads(line[5:].strip())
                now = time.monotonic()
                result["events"].append({"seconds": now - begin, "event": event})
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage") is not None:
                    result["usage"] = event["usage"]
                if event.get("timings") is not None:
                    result["timings"] = event["timings"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    reasoning = (
                        delta.get("reasoning_content") or delta.get("reasoning") or ""
                    )
                    if first is None and (content or reasoning):
                        first = now
                    result["content"] += content
                    result["reasoning"] += reasoning
                    result["finish_reason"] = (
                        choice.get("finish_reason") or result["finish_reason"]
                    )
        if result["usage"] is None or result["finish_reason"] is None:
            raise RuntimeError("stream ended without final usage and finish reason")
        result["status"] = "completed"
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
    end = time.monotonic()
    result.update(
        end_monotonic=end,
        wall_seconds=end - begin,
        ttft_seconds=None if first is None else first - begin,
        first_monotonic=first,
        output_sha256=sha((result["content"] + result["reasoning"]).encode()),
    )
    usage = result["usage"] or {}
    n = usage.get("completion_tokens")
    result["natural_eos_before_cap"] = (
        result["finish_reason"] == "stop"
        and isinstance(n, int)
        and n < body["max_tokens"]
    )
    result["requested_output_tokens"] = body["max_tokens"]
    result["sustained_output_target_met"] = (
        isinstance(n, int) and n >= body["max_tokens"]
    )
    result["observed_decode_tok_s"] = (
        (n - 1) / (end - first)
        if isinstance(n, int) and n > 1 and first and end > first
        else None
    )
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    result["cache_class"] = (
        "cold"
        if cached == 0
        else "cached"
        if isinstance(cached, int) and cached > 0
        else "unknown"
    )
    timings = result["timings"] or {}
    fresh = (usage.get("prompt_tokens") or 0) - (cached or 0)
    result["fresh_prompt_tok_s"] = (
        fresh / (timings["prompt_ms"] / 1000) if timings.get("prompt_ms") else None
    )
    return result


def burst(args, fixtures, stage, repeat, count, headers):
    chosen = fixtures[args.fixture_offset : args.fixture_offset + count]
    gate = threading.Barrier(count)
    bodies = []
    for fixture in chosen:
        body = copy.deepcopy(fixture["body"])
        body.pop("strata_tune", None)
        body.update(
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=args.tokens,
            temperature=1.0,
            top_p=1.0,
            top_k=64,
            min_p=0.0,
            seed=42,
            reasoning_effort="max",
            strata_checkpoint=True,
        )
        bodies.append(body)
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [
            pool.submit(
                sample,
                args.base_url,
                body,
                f.get("case", str(i)),
                headers,
                args.timeout,
                gate,
            )
            for i, (f, body) in enumerate(zip(chosen, bodies))
        ]
        rows = [future.result() for future in futures]
    started = min(row["start_monotonic"] for row in rows)
    ended = max(row["end_monotonic"] for row in rows)
    firsts = [
        row["first_monotonic"] for row in rows if row["first_monotonic"] is not None
    ]
    tokens = [(row["usage"] or {}).get("completion_tokens") or 0 for row in rows]
    decode_window = ended - min(firsts) if firsts else None
    result = {
        "schema": "hy4.qualification.burst/1",
        "label": args.label,
        "stage": stage,
        "repeat": repeat,
        "parallel": count,
        "utc": dt.datetime.now(dt.UTC).isoformat(),
        "base_url": args.base_url,
        "fixture_sha256": sha(Path(args.fixtures).read_bytes()),
        "runtime_manifest": args.runtime_manifest,
        "wall_seconds": ended - started,
        "decode_window_seconds": decode_window,
        "aggregate_e2e_tok_s": sum(tokens) / (ended - started),
        "aggregate_observed_decode_tok_s": sum(max(n - 1, 0) for n in tokens)
        / decode_window
        if decode_window
        else None,
        "all_output_targets_met": all(
            row["sustained_output_target_met"] for row in rows
        ),
        "all_prompt_targets_met": all(
            (row["usage"] or {}).get("prompt_tokens") == args.prompt_tokens
            for row in rows
        ),
        "all_cold": all(row["cache_class"] == "cold" for row in rows),
        "requests": rows,
    }
    if args.runtime_manifest:
        manifest = Path(args.runtime_manifest)
        result["runtime_manifest_sha256"] = sha(manifest.read_bytes())
        result["runtime"] = json.loads(manifest.read_text())
    path = Path(args.output) / f"{args.label}-{stage}-c{count}-r{repeat}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "requests"}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--fixtures", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--runtime-manifest",
        help="path to independently captured binary/config/hardware provenance",
    )
    p.add_argument("--parallel", default="1,4")
    p.add_argument(
        "--fixture-offset",
        type=int,
        default=0,
        help="use disjoint fixtures for separately measured concurrency levels",
    )
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--prompt-tokens", type=int, default=8192)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--api-key-env", default="STRATA_BENCH_API_KEY")
    p.add_argument("--cached-repeat", action="store_true")
    a = p.parse_args()
    Path(a.output).mkdir(parents=True, exist_ok=True)
    fixtures = json.loads(Path(a.fixtures).read_text())
    headers = {"Content-Type": "application/json"}
    if os.environ.get(a.api_key_env):
        headers["Authorization"] = "Bearer " + os.environ[a.api_key_env]
    snapshot = {
        name: get_json(a.base_url.rstrip("/") + name, headers)
        for name in ("/health", "/props", "/v1/status")
    }
    (Path(a.output) / (a.label + "-service-snapshot.json")).write_text(
        json.dumps(snapshot, indent=2) + "\n"
    )
    results = []
    for count in map(int, a.parallel.split(",")):
        if not 1 <= count <= len(fixtures) - a.fixture_offset or a.fixture_offset < 0:
            raise ValueError("parallel count exceeds fixture count")
        for repeat in range(a.repeats):
            results.append(burst(a, fixtures, "initial", repeat, count, headers))
            if a.cached_repeat:
                results.append(burst(a, fixtures, "repeated", repeat, count, headers))
    summary = {
        "label": a.label,
        "reports": len(results),
        "note": "Initial/repeated are order labels; cache_class is observed, never assumed. Early EOS is retained, never masked.",
        "median_aggregate_e2e_tok_s": statistics.median(
            r["aggregate_e2e_tok_s"] for r in results
        ),
    }
    (Path(a.output) / (a.label + "-summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return (
        0
        if all(
            row["status"] == "completed"
            for result in results
            for row in result["requests"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
