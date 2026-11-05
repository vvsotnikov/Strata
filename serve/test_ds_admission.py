"""DeepSeek requests waiting for native capacity must not run an execution watchdog.

Run with python -m unittest serve.test_ds_admission -v; no GPU or model is used.
The subprocess implements the native engine's finite-capacity, silent waiting queue.
"""
import concurrent.futures
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve import server


FAKE_DS = r'''
import collections, queue, sys, threading, time
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--control') + 1])
n = int(sys.argv[sys.argv.index('--parallel') + 1])
commands = queue.Queue()
def reader():
    for line in sys.stdin:
        commands.put(line.strip())
    commands.put('QUIT')
threading.Thread(target=reader, daemon=True).start()
print(f'READY 4096 stop multi={n}', flush=True)
active, waiting, stopped = {}, collections.deque(), set()
while True:
    try:
        line = commands.get(timeout=0.02)
    except queue.Empty:
        line = ''
    if line == 'QUIT':
        break
    if line.startswith('GEN '):
        fields = line.split()
        rid = next(int(x[4:]) for x in fields if x.startswith('rid='))
        token = int(fields[-1].split(',')[0])
        (root / f'sent-{token}').touch()
        waiting.append((rid, token))
    elif line.startswith('STOP '):
        stopped.add(int(line.split()[1]))
    elif line == 'STOP':
        stopped.update(active)
    while waiting and len(active) < n:
        rid, token = waiting.popleft()
        active[rid] = (token, 0)
        (root / f'started-{token}').touch()
    for rid, (token, count) in list(active.items()):
        if rid in stopped:
            (root / f'stopped-{token}').touch()
            if (root / f'defer-stop-{token}').exists():
                continue
        if rid in stopped or (root / f'release-{token}').exists():
            print(f'DONE {rid} {count} 1 1 1 stop 0 0 0 0 0', flush=True)
            del active[rid]
        elif not (root / f'freeze-{token}').exists():
            print(f'T {rid} {token}', flush=True)
            (root / f'token-{token}').touch()
            active[rid] = (token, count + 1)
    time.sleep(0.02)
'''


class Admission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        script = self.root / 'engine.py'
        script.write_text(FAKE_DS, encoding='utf-8')
        popen = server.subprocess.Popen
        self.patcher = mock.patch.object(
            server.subprocess, 'Popen',
            lambda cmd, **kw: popen([sys.executable, str(script), *cmd[1:]], **kw),
        )
        self.patcher.start()
        self.engine = server.StrataEngine('strata', ['--parallel', '2', '--control', str(self.root)])
        self.engine.silence_s = 0.3
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.cancels = []

    def tearDown(self):
        for cancel in self.cancels:
            cancel.set()
        for path in self.root.glob('defer-stop-*'):
            path.unlink()
        for token in range(65, 70):
            self.touch(f'release-{token}')
        self.pool.shutdown(wait=True)
        self.engine.close()
        self.patcher.stop()
        self.temp.cleanup()

    def touch(self, name):
        (self.root / name).touch()

    def wait_for(self, name):
        deadline = time.monotonic() + 3
        while not (self.root / name).exists():
            self.assertTrue(self.engine.alive(), f'engine died while waiting for {name}')
            self.assertLess(time.monotonic(), deadline, name)
            time.sleep(0.01)

    def request(self, token, stop_after=None):
        cancel = threading.Event()
        self.cancels.append(cancel)

        def run():
            result = []
            gen = self.engine.generate([token], 100000, {}, cancel)
            try:
                for item in gen:
                    if item is not None:
                        result.append(item)
                        if cancel.is_set() or (stop_after and len(result) >= stop_after):
                            break
                return result
            finally:
                gen.close()

        return self.pool.submit(run), cancel

    def fill(self):
        a, _ = self.request(65)
        b, _ = self.request(66)
        self.wait_for('started-65')
        self.wait_for('started-66')
        return a, b

    def test_waiting_longer_than_silence_limit_does_not_kill_active_requests(self):
        a, b = self.fill()
        c, _ = self.request(67)
        time.sleep(0.8)  # active requests keep producing; the third waits past its 0.32 s allowance
        self.assertTrue(self.engine.alive())
        self.assertFalse(c.done())
        self.assertFalse((self.root / 'sent-67').exists())
        self.touch('release-65')
        self.wait_for('started-67')
        self.wait_for('token-67')
        self.touch('release-66')
        self.touch('release-67')
        for token, future in ((65, a), (66, b), (67, c)):
            result = future.result(timeout=3)
            self.assertTrue(result)
            self.assertEqual(set(result), {token})
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.engine.rid_q, {})

    def test_cancellation_while_waiting_sends_no_native_request(self):
        self.fill()
        c, cancel = self.request(67)
        time.sleep(0.1)
        cancel.set()
        self.assertEqual(c.result(timeout=1), [])
        self.assertFalse((self.root / 'sent-67').exists())
        self.assertTrue(self.engine.alive())

    def test_cancelled_active_request_holds_capacity_until_stop_acknowledged(self):
        self.touch('defer-stop-65')
        a, _ = self.request(65, stop_after=1)
        b, _ = self.request(66)
        self.wait_for('stopped-65')
        self.wait_for('started-66')
        c, _ = self.request(67)
        time.sleep(0.1)
        self.assertFalse((self.root / 'sent-67').exists())
        self.assertFalse(a.done())
        (self.root / 'defer-stop-65').unlink()
        a.result(timeout=1)
        self.wait_for('started-67')
        self.touch('release-66')
        self.touch('release-67')
        b.result(timeout=1)
        c.result(timeout=1)

    def test_engine_death_releases_waiting_requests(self):
        a, b = self.fill()
        c, _ = self.request(67)
        time.sleep(0.1)
        self.engine.proc.kill()  # only this test's fake subprocess
        for future in (a, b, c):
            with self.assertRaises(server.EngineDied):
                future.result(timeout=2)

    def test_admitted_request_still_has_a_silence_watchdog(self):
        self.touch('freeze-65')
        a, _ = self.request(65)
        self.wait_for('started-65')
        with self.assertRaises(server.EngineSilent):
            a.result(timeout=2)
        self.assertFalse(self.engine.alive())

    def test_invalid_request_does_not_leak_capacity(self):
        for _ in range(3):
            with self.assertRaises(ValueError):
                list(self.engine.generate([65], 10, {'logit_bias': {'bad': 1}}, threading.Event()))
        a, b = self.fill()
        self.touch('release-65')
        self.touch('release-66')
        a.result(timeout=1)
        b.result(timeout=1)

    def test_restart_does_not_admit_an_old_waiter_to_the_new_process(self):
        a, b = self.fill()
        c, _ = self.request(67)
        time.sleep(0.1)
        self.engine.proc.kill()
        self.engine.restart()
        for future in (a, b, c):
            with self.assertRaises(server.EngineDied):
                future.result(timeout=2)
        self.assertFalse((self.root / 'sent-67').exists())
        self.assertTrue(self.engine.alive())
        d, _ = self.request(68)
        self.wait_for('started-68')
        self.touch('release-68')
        d.result(timeout=1)


if __name__ == '__main__':
    unittest.main()
