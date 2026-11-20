"""CPU-only tests for benchmark evidence and grading (no inference/GPU/Docker)."""

import ast
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import qualification as q
import quality


class EvidenceTests(unittest.TestCase):
    def stream(self, events):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in events:
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            return q.sample(
                f"http://127.0.0.1:{server.server_port}",
                {"max_tokens": 64},
                "case",
                {"Content-Type": "application/json"},
                5,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_early_eos_kept_and_cache_measured(self):
        row = self.stream(
            [
                {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
                {
                    "choices": [
                        {"delta": {"content": "answer"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 256,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                    "timings": {"prompt_ms": 100},
                },
            ]
        )
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["content"], "answer")
        self.assertEqual(row["reasoning"], "thinking")
        self.assertTrue(row["natural_eos_before_cap"])
        self.assertFalse(row["sustained_output_target_met"])
        self.assertEqual(row["cache_class"], "cold")
        self.assertEqual(row["fresh_prompt_tok_s"], 2560)
        self.assertEqual(len(row["events"]), 2)

    def test_truncated_stream_is_not_success(self):
        row = self.stream([{"choices": [{"delta": {"content": "partial"}}]}])
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["content"], "partial")
        self.assertIn("without final usage", row["error"])

    def test_stream_error_preserves_partial_output(self):
        row = self.stream(
            [
                {"choices": [{"delta": {"content": "partial"}}]},
                {"error": {"message": "capacity"}},
            ]
        )
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["content"], "partial")
        self.assertIn("capacity", row["error"])

    def test_null_accounting_does_not_invent_cold_or_speed(self):
        row = self.stream(
            [
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": None, "completion_tokens": None},
                }
            ]
        )
        self.assertEqual(row["cache_class"], "unknown")
        self.assertIsNone(row["observed_decode_tok_s"])
        self.assertFalse(row["sustained_output_target_met"])

    def test_partial_function_indentation_is_preserved(self):
        reference = {"entry_point": "f", "prompt": 'def f():\n    """Return one."""\n'}
        candidate = quality.code_candidate("    return 1\n", reference)
        function = ast.parse(candidate).body[0]
        self.assertEqual(function.name, "f")
        self.assertEqual(function.body[-1].value.value, 1)

    def test_mmlu_requires_final_content_answer(self):
        fixture = {"kind": "mmlu", "reference": {"answer": 2}}
        good = quality.grade(
            fixture,
            {"status": "completed", "content": "Answer: C", "finish_reason": "stop"},
            None,
        )
        self.assertTrue(good["pass"])
        incomplete = quality.grade(
            fixture,
            {
                "status": "completed",
                "content": "",
                "reasoning": "Answer: C",
                "finish_reason": "length",
            },
            None,
        )
        self.assertEqual(incomplete["status"], "incomplete")
        self.assertFalse(incomplete["pass"])

    def test_code_never_executes_without_explicit_sandbox(self):
        fixture = {"kind": "humaneval", "reference": {}}
        with patch.object(quality.subprocess, "run") as run:
            result = quality.grade(
                fixture, {"status": "completed", "content": "anything"}, None
            )
        run.assert_not_called()
        self.assertEqual(result["status"], "not_executed")

    def test_deferred_grading_does_not_send_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = {"case": "science", "kind": "mmlu", "reference": {"answer": 2}}
            response = {
                "status": "completed",
                "content": "Answer: C",
                "finish_reason": "stop",
            }
            (root / "fixtures.json").write_text(json.dumps([fixture]))
            (root / "results.json").write_text(
                json.dumps([{"case": "science", "kind": "mmlu", "response": response}])
            )
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "quality",
                        "--dataset-dir",
                        folder,
                        "--output",
                        folder,
                        "--grade-only",
                    ],
                ),
                patch.object(quality, "sample") as infer,
            ):
                self.assertEqual(quality.main(), 0)
            infer.assert_not_called()
            self.assertEqual(
                json.loads((root / "summary.json").read_text())["mmlu"]["passed"], 1
            )

    def test_parallel_collection_preserves_every_case(self):
        fixtures = [
            {
                "case": f"science-{i}",
                "kind": "mmlu",
                "reference": {"answer": 2},
                "prompt": "Question",
            }
            for i in range(4)
        ]
        lock = threading.Lock()
        active = peak = 0

        def infer(*args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(active, peak)
            time.sleep(0.02)
            with lock:
                active -= 1
            return {
                "status": "completed",
                "content": "Answer: C",
                "finish_reason": "stop",
            }

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(quality, "cases", return_value=fixtures),
            patch.object(quality, "sample", side_effect=infer),
            patch.object(
                sys,
                "argv",
                [
                    "quality",
                    "--dataset-dir",
                    folder,
                    "--output",
                    folder,
                    "--base-url",
                    "unused",
                    "--parallel",
                    "2",
                ],
            ),
        ):
            self.assertEqual(quality.main(), 0)
            rows = json.loads((Path(folder) / "results.json").read_text())
        self.assertEqual(peak, 2)
        self.assertEqual(
            {row["case"] for row in rows}, {fixture["case"] for fixture in fixtures}
        )
        self.assertTrue(all(row["grade"]["pass"] for row in rows))


if __name__ == "__main__":
    unittest.main()
