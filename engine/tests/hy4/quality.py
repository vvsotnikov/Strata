"""Small pinned public quality suites; preserve raw output and isolate generated code.

HumanEval execution requires an explicitly selected local Docker image. The
container has no network, model mount, GPU, writable root, or host bind mounts.
This is a mini regression suite, not a full HumanEval or MMLU benchmark.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from qualification import sample, sha


def cases(dataset_dir, count):
    root = Path(dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    human_raw = (root / "HumanEval.jsonl.gz").read_bytes()
    science_raw = (root / "mmlu-college-physics-rows.json").read_bytes()
    if (
        sha(human_raw) != manifest["humaneval"]["sha256"]
        or sha(science_raw) != manifest["mmlu"]["sha256"]
    ):
        raise ValueError("dataset content does not match pinned manifest")
    human = [json.loads(line) for line in gzip.decompress(human_raw).splitlines()]
    science = json.loads(science_raw)["rows"]
    if not 1 <= count <= min(len(human), len(science)):
        raise ValueError("count exceeds the available pinned subset")
    output = []
    for row in human[:count]:
        output.append(
            {
                "case": row["task_id"],
                "kind": "humaneval",
                "reference": row,
                "prompt": "Implement this Python function correctly. Return only the completed function, with any needed imports.\n"
                + row["prompt"],
            }
        )
    for item in science[:count]:
        row = item["row"]
        text = (
            row["question"]
            + "\n"
            + "\n".join(f"{chr(65 + i)}. {x}" for i, x in enumerate(row["choices"]))
        )
        output.append(
            {
                "case": f"mmlu-college_physics/{item['row_idx']}",
                "kind": "mmlu",
                "reference": row,
                "prompt": text
                + "\nSolve carefully. Finish your final answer with exactly Answer: A, Answer: B, Answer: C or Answer: D.",
            }
        )
    return output


def code_candidate(content, reference):
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content, flags=re.DOTALL)
    code = max(fenced, key=len) if fenced else content.rstrip()
    if re.search(
        r"^\s*(?:async\s+)?def\s+" + re.escape(reference["entry_point"]) + r"\s*\(",
        code,
        re.MULTILINE,
    ):
        header = reference["prompt"].split("def " + reference["entry_point"], 1)[0]
        return header + "\n" + code
    return reference["prompt"] + code


def grade(row, response, image):
    text = response.get("content") or ""
    if response.get("status") != "completed":
        return {"status": "request_error", "pass": False}
    if row["kind"] == "mmlu":
        matches = re.findall(r"\bAnswer:\s*([ABCD])\b", text, re.IGNORECASE)
        if not matches:
            return {
                "status": "incomplete"
                if response.get("finish_reason") == "length"
                else "unparseable",
                "pass": False,
            }
        answer = matches[-1].upper()
        return {
            "status": "scored",
            "answer": answer,
            "expected": chr(65 + row["reference"]["answer"]),
            "pass": ord(answer) - 65 == row["reference"]["answer"],
        }
    if not image:
        return {
            "status": "not_executed",
            "pass": False,
            "reason": "an isolated local Docker image is required",
        }
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("the sandbox image must be pinned by immutable sha256 ID")
    source = code_candidate(text, row["reference"])
    program = (
        source
        + "\n"
        + row["reference"]["test"]
        + "\ncheck("
        + row["reference"]["entry_point"]
        + ')\nprint("HUMANEVAL_PASS")\n'
    )
    name = "hy4-quality-" + uuid.uuid4().hex
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "-i",
        "--name=" + name,
        "--runtime=runc",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=16",
        "--memory=256m",
        "--cpus=1",
        "--user=65534:65534",
        "--ulimit=cpu=2:2",
        "--ulimit=fsize=1048576:1048576",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--entrypoint=/opt/venv/bin/python3",
        image,
        "-I",
        "-S",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            input=program,
            text=True,
            capture_output=True,
            timeout=8,
            check=False,
        )
        return {
            "status": "scored",
            "pass": result.returncode == 0 and "HUMANEVAL_PASS" in result.stdout,
            "returncode": result.returncode,
            "stdout": result.stdout[-8192:],
            "stderr": result.stderr[-8192:],
            "candidate_sha256": sha(source.encode()),
            "candidate": source,
            "sandbox_image": image,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "status": "execution_error",
            "pass": False,
            "error": str(error),
            "candidate_sha256": sha(source.encode()),
        }
    finally:
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base-url", help="omit to prepare fixtures without executing inference"
    )
    parser.add_argument("--model", default="hy4-flash")
    parser.add_argument("--effort", choices=("none", "max"), default="max")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument(
        "--parallel",
        type=int,
        choices=range(1, 5),
        default=1,
        help="concurrent quality requests; code execution waits until all requests finish",
    )
    parser.add_argument(
        "--grade-only",
        action="store_true",
        help="score saved results without sending inference requests",
    )
    parser.add_argument(
        "--sandbox-image",
        help="pinned sha256 image with /opt/venv/bin/python3; never pulled automatically",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.grade_only:
        fixtures = {
            row["case"]: row
            for row in json.loads((output / "fixtures.json").read_text())
        }
        rows = json.loads((output / "results.json").read_text())
        for row in rows:
            row["grade"] = grade(
                fixtures[row["case"]], row["response"], args.sandbox_image
            )
        (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        return write_summary(output, rows)
    fixtures = cases(args.dataset_dir, args.count)
    for row in fixtures:
        row["body"] = {
            "model": args.model,
            "messages": [{"role": "user", "content": row.pop("prompt")}],
            "reasoning_effort": args.effort,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 64,
            "min_p": 0.0,
            "seed": 42,
            "max_tokens": args.tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    (output / "fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")
    if not args.base_url:
        return 0
    rows = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        pending = {
            pool.submit(
                sample,
                args.base_url,
                fixture["body"],
                fixture["case"],
                {"Content-Type": "application/json"},
                args.timeout,
            ): fixture
            for fixture in fixtures
        }
        for future in as_completed(pending):
            fixture = pending[future]
            response = future.result()
            row = {
                "case": fixture["case"],
                "kind": fixture["kind"],
                "response": response,
            }
            # No Docker/CPU grading while any inference request is active.
            row["grade"] = grade(fixture, response, None)
            rows.append(row)
            (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "case": row["case"],
                        "grade": row["grade"],
                        "usage": response.get("usage"),
                    },
                    default=str,
                ),
                flush=True,
            )
    if args.sandbox_image:
        fixtures_by_case = {fixture["case"]: fixture for fixture in fixtures}
        for row in rows:
            row["grade"] = grade(
                fixtures_by_case[row["case"]], row["response"], args.sandbox_image
            )
        (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    return write_summary(output, rows)


def write_summary(output, rows):
    summary = {
        kind: {
            "cases": sum(r["kind"] == kind for r in rows),
            "passed": sum(r["kind"] == kind and r["grade"]["pass"] for r in rows),
            "scored": sum(
                r["kind"] == kind and r["grade"]["status"] == "scored" for r in rows
            ),
            "errors": sum(
                r["kind"] == kind
                and r["grade"]["status"] in ("request_error", "execution_error")
                for r in rows
            ),
            "incomplete": sum(
                r["kind"] == kind
                and r["grade"]["status"]
                in ("incomplete", "unparseable", "not_executed")
                for r in rows
            ),
        }
        for kind in ("humaneval", "mmlu")
    }
    summary["note"] = (
        "Pinned mini subsets; not full HumanEval pass@1 or MMLU accuracy. Natural EOS and truncation retained."
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if all(r["response"]["status"] == "completed" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
