"""Python transport against the real C++ scheduler with a CPU synthetic model.

Set STRATA_HY4_TEST_SERVER to a built hy4_server_test; no GPU or weights are used.
"""
from __future__ import annotations

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from serve.server import Hy4Engine

EXE = os.environ.get("STRATA_HY4_TEST_SERVER")


@unittest.skipUnless(EXE, "build hy4_server_test and set STRATA_HY4_TEST_SERVER")
class NativeProtocol(unittest.TestCase):
    def test_frontend_closing_on_native_eos_drains_complete_stats(self):
        with tempfile.TemporaryDirectory(prefix="hy4-native-eos-") as folder:
            engine = Hy4Engine(EXE, ["--stdio", "--mtp", "2"], log=str(Path(folder) / "engine.log"))
            try:
                self.assertEqual(engine.eos_ids, {6})
                stream = engine.generate([0, 1], 8, {"temperature": 0}, threading.Event())
                tokens = []
                for token in stream:
                    if token is None:
                        continue
                    tokens.append(token)
                    if token in engine.eos_ids:
                        break  # Service.run closes the native iterator at the model's EOS
                stream.close()
                self.assertEqual(tokens, [2, 3, 4, 5, 6])
                self.assertEqual(engine.last["generated"], 5)
                self.assertEqual(engine.last["finish"], "stop")
                self.assertIsNotNone(engine.last["decode_ms"])
                self.assertGreater(engine.last["decode_ms"], 0)
                self.assertGreater(engine.last["drafts_accepted"], 0)
                self.assertFalse(any(engine.slot_busy))
                # A new READY on a restart refreshes from native metadata too.
                engine.restart()
                self.assertEqual(engine.eos_ids, {6})
            finally:
                engine.close()

    def test_four_parallel_requests_use_native_slots_and_release_them(self):
        with tempfile.TemporaryDirectory(prefix="hy4-native-protocol-") as folder:
            engine = Hy4Engine(EXE, ["--stdio", "--mtp", "2"], log=str(Path(folder) / "engine.log"))
            try:
                commands = []
                send = engine._send
                def tracked(command):
                    commands.append(command)
                    return send(command)
                engine._send = tracked
                self.assertEqual(engine.batch, 4)
                self.assertTrue(engine.slot_read)
                self.assertEqual(engine.info["kv_unified"], 1)
                solo = [t for t in engine.generate([1, 2], 3, {"temperature": 0}, threading.Event()) if t is not None]
                self.assertEqual(solo, [3, 4, 5])
                gate = threading.Barrier(4)
                def generate(i):
                    gate.wait(timeout=5)
                    prompt = [0, 1] * (i + 4)
                    tokens = [t for t in engine.generate(prompt, 3, {"temperature": 0}, threading.Event()) if t is not None]
                    return tokens, dict(engine.last), len(prompt)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    result = list(pool.map(generate, range(4)))
                for tokens, stats, prompt_tokens in result:
                    self.assertEqual(tokens, [2, 3, 4])
                    self.assertEqual(stats["generated"], 3)
                    self.assertEqual(stats["prompt_tokens"], prompt_tokens)
                    self.assertGreaterEqual(stats["prompt_read"], 0)
                    self.assertLessEqual(stats["prompt_read"], prompt_tokens)
                self.assertGreater(sum(s.get("drafts_accepted", 0) for _, s, _ in result), 0)
                self.assertFalse(any(engine.slot_busy))
                self.assertEqual(engine.info["drafting"], "mtp")
                self.assertEqual(sum(command.startswith("BGEN ") for command in commands), 5)
                self.assertFalse(any(command.startswith(("GEN ", "GENI ", "BCO ")) for command in commands))
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
