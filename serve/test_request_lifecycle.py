"""The queued/executing contract, independently of a native wire protocol."""
import sys
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve import server
from serve.request_lifecycle import ExecutionStalled, QueueWaitExpired, RequestLifecycle


class Policy(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.cancel = threading.Event()
        self.transitions = []
        self.request = RequestLifecycle(self.cancel, clock=lambda: self.now, changed=self.transitions.append)

    def test_waiting_has_no_execution_clock(self):
        self.now = 10000
        self.assertEqual(self.request.phase, 'queued')
        self.assertIsNone(self.request.heard)
        self.assertGreater(self.request.queue_wait(), 0)
        self.assertGreater(self.request.execution_wait(), 0)
        self.request.executing(0, 2)
        self.now += 2.1
        with self.assertRaises(ExecutionStalled):
            self.request.execution_wait()

    def test_queue_deadline_is_absolute_and_stops_at_execution(self):
        self.cancel._strata_queue_deadline = 3
        self.now = 2
        self.assertGreater(self.request.queue_wait(), 0)
        self.request.executing(0, 10)
        self.now = 4
        self.assertIsNone(self.cancel._strata_queue_deadline)
        self.assertGreater(self.request.execution_wait(), 0)

    def test_expired_queue_deadline_does_not_become_an_execution_stall(self):
        self.cancel._strata_queue_deadline = 3
        self.now = 4
        with self.assertRaises(QueueWaitExpired):
            self.request.queue_wait()
        self.assertEqual(self.transitions, [])
        self.assertIsNone(self.request.heard)

    def test_prompt_allowance_then_token_silence_window(self):
        self.request.executing(100, 2)
        self.now = 3
        self.assertGreater(self.request.execution_wait(), 0)
        self.request.progress('PP 100 200 1000 100')
        self.now = 5.5  # 100-token chunk at 100 tokens/s, with the existing 3x allowance
        self.assertGreater(self.request.execution_wait(), 0)
        self.request.progress('T 65')
        self.now += 2.1
        with self.assertRaises(ExecutionStalled):
            self.request.execution_wait()

    def test_batched_tokens_use_the_same_execution_policy(self):
        self.request.executing(0, 2)
        self.now = 1
        self.request.progress('BT 1 65')
        self.now = 2.5
        self.assertGreater(self.request.execution_wait(), 0)
        self.now = 3.1
        with self.assertRaises(ExecutionStalled):
            self.request.execution_wait()

    def test_resumed_prompt_only_charges_the_new_chunk(self):
        self.request.executing(1000, 2)
        self.request.progress('RESUME 900')
        self.request.progress('PP 1000 1000 1000 100')
        self.assertEqual(self.request.allow_s, 3)

    def test_old_request_cannot_read_or_kill_a_replacement_engine(self):
        old, replacement = mock.Mock(), mock.Mock()
        engine = object.__new__(server.StrataEngine)
        engine.proc = replacement
        self.request.executing(0, 1, owner=old)
        self.now = 100
        with self.assertRaises(server.EngineDied):
            engine._execution_line(self.request, queue.Queue())
        replacement.kill.assert_not_called()
        old.kill.assert_not_called()

    def test_restart_during_a_watchdog_poll_does_not_kill_the_new_process(self):
        old, replacement = mock.Mock(), mock.Mock()
        engine = object.__new__(server.StrataEngine)
        engine.proc = old
        self.request.executing(0, 1, owner=old)
        def restarted(**kwargs):
            engine.proc = replacement
            self.now = 100
            raise queue.Empty
        lines = mock.Mock()
        lines.get.side_effect = restarted
        with self.assertRaises(server.EngineDied):
            engine._execution_line(self.request, lines)
        replacement.kill.assert_not_called()

    def test_return_to_queue_disarms_execution_and_accounts_for_each_wait(self):
        self.now = 2
        self.request.executing(0, 1)
        self.now = 2.5
        self.request.queued()
        self.now = 100
        self.assertGreater(self.request.execution_wait(), 0)
        self.request.executing(0, 1)
        self.assertAlmostEqual(self.request.queue_s, 99.5)
        self.assertEqual(self.transitions, ['executing', 'queued', 'executing'])

    def test_native_scheduler_pause_preserves_the_remaining_execution_time(self):
        self.request.executing(0, 2)
        self.now = 1
        self.request.execution_paused(True)
        self.now = 100
        self.assertGreater(self.request.execution_wait(), 0)
        self.request.execution_paused(False)
        self.now = 100.5
        self.assertGreater(self.request.execution_wait(), 0)
        self.now = 101.1
        with self.assertRaises(ExecutionStalled):
            self.request.execution_wait()

    def test_slot_adapter_pauses_only_while_another_control_reader_is_watched(self):
        engine = object.__new__(server.StrataEngine)
        engine.proc = mock.Mock()
        self.request.executing(0, 2, owner=engine.proc)
        control = RequestLifecycle(threading.Event(), clock=lambda: self.now)
        control.executing(0, 2, owner=engine.proc)
        engine._control_request = control
        lines = mock.Mock()
        def silent(**kwargs):
            self.now += 100
            raise queue.Empty
        lines.get.side_effect = silent
        self.assertIs(engine._execution_line(self.request, lines, native_slot=True), server.NO_LINE)
        self.assertIsNotNone(self.request.paused_at)
        with self.assertRaises(ExecutionStalled):
            control.execution_wait()  # the operation that holds up slot decoding stays watched
        engine._control_request = None
        lines.get.side_effect = None
        lines.get.return_value = 'BT 1 65'
        self.assertEqual(engine._execution_line(self.request, lines, native_slot=True), 'BT 1 65')
        self.assertIsNone(self.request.paused_at)
        self.now += 2.1
        with self.assertRaises(server.EngineSilent):
            engine._execution_line(self.request, lines, native_slot=True)
        engine.proc.kill.assert_called_once()

    def test_cancellation_and_heartbeat_while_queued(self):
        self.now = 10
        self.assertTrue(self.request.heartbeat())
        self.assertFalse(self.request.heartbeat())
        self.cancel.set()
        self.assertEqual(self.request.queue_wait(), 0)

    def test_disabled_execution_watchdog_still_has_queue_deadlines(self):
        self.request.executing(0, 0)
        self.now = 10000
        self.assertGreater(self.request.execution_wait(), 0)
        self.request.queued()
        self.cancel._strata_queue_deadline = 1
        with self.assertRaises(QueueWaitExpired):
            self.request.queue_wait()

    def test_cancellation_at_submission_does_not_arm_execution(self):
        engine = object.__new__(server.StrataEngine)
        self.cancel._strata_lifecycle = self.request
        self.cancel.set()
        with self.assertRaises(server.QueueCancelled):
            engine._execution_start(self.cancel, 1)
        self.assertEqual(self.request.phase, 'queued')
        self.assertIsNone(self.request.heard)


class ProtocolContract(unittest.TestCase):
    def engine(self, root, script, args):
        path = root / 'native.py'
        path.write_text(script)
        popen = server.subprocess.Popen
        with mock.patch.object(server.subprocess, 'Popen',
                               lambda cmd, **kw: popen([sys.executable, str(path), *cmd[1:]], **kw)):
            return server.StrataEngine('strata', args)

    def test_serial_request_id_and_slot_protocols_report_the_same_lifecycle(self):
        from serve.test_ds_parallel import FAKE_DS
        from serve.test_parallel import FAKE_BATCH
        for name, script, args, limit in (
                ('serial', FAKE_BATCH, ['--batch', '0'], 8),
                ('request-id', FAKE_DS, ['--parallel', '2'], 8),
                ('native-slot', FAKE_BATCH, ['--batch', '2'], 1)):
            with self.subTest(protocol=name), tempfile.TemporaryDirectory() as directory:
                engine = self.engine(Path(directory), script, args)
                cancel = threading.Event()
                transitions = []
                request = RequestLifecycle(cancel, changed=transitions.append)
                cancel._strata_lifecycle = request
                try:
                    tokens = []
                    for token in engine.generate([65], limit, {}, cancel):
                        self.assertEqual(request.phase, 'executing')
                        if token is not None:
                            tokens.append(token)
                    self.assertTrue(tokens)
                    self.assertEqual(transitions, ['executing', 'finished'])
                    self.assertEqual(request.phase, 'finished')
                finally:
                    engine.close()

    def test_every_protocol_applies_the_execution_watchdog_after_submission(self):
        for mode in ('serial', 'request-id', 'native-slot'):
            ready = 'INFO batch_slots=2\n' if mode == 'native-slot' else ''
            ready += 'READY 4096 stop' + (' multi=2' if mode == 'request-id' else '')
            script = f'import sys\nprint({ready!r},flush=True)\nfor line in sys.stdin:\n if line.strip()=="QUIT":break\n'
            with self.subTest(protocol=mode), tempfile.TemporaryDirectory() as directory:
                engine = self.engine(Path(directory), script, [])
                engine.silence_s = 0.2
                try:
                    with self.assertRaises(server.EngineSilent):
                        list(engine.generate([65], 1, {}, threading.Event()))
                    self.assertFalse(engine.alive())
                finally:
                    engine.close()


if __name__ == '__main__':
    unittest.main()
