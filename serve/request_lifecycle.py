"""Policy shared by native protocol adapters.

A request waits under its queue deadline until the adapter submits native work.
Only executing work has a silence watchdog. Protocol adapters own native slots
and command formats; this module owns cancellation, deadlines and keepalives.
"""
import collections
import threading
import time


class QueueWaitExpired(RuntimeError):
    pass


class ExecutionStalled(RuntimeError):
    pass


class RequestLifecycle:
    def __init__(self, cancel, *, heartbeat_s=10.0, clock=None, changed=None):
        self.cancel = cancel
        self.clock = clock or (lambda: time.monotonic())
        self.heartbeat_s = heartbeat_s
        self.changed = changed
        self.phase = "queued"
        self.beat = self.clock()
        self.queued_at, self.queue_s = self.beat, 0.0
        self.heard = None
        self.owner = None
        self.paused_at = None
        self.silence_s = self.allow_s = 0.0
        self.read_to = 0
        self.chunk_max, self.prompt_floor, self.prompt_slack = 32768, 50.0, 3.0
        self.stall_state = {}

    def _phase(self, phase):
        if self.phase != phase:
            self.phase = phase
            if self.changed is not None:
                self.changed(phase)

    def queued(self):
        if self.phase != "queued":
            self.queued_at = self.clock()
        self.heard = None
        self.paused_at = None
        self.stall_state.clear()
        self._phase("queued")

    def queue_wait(self):
        """A bounded cancellation/deadline poll; never runs the execution watchdog."""
        if self.cancel.is_set():
            return 0.0
        deadline = getattr(self.cancel, "_strata_queue_deadline", None)
        left = deadline - self.clock() if deadline is not None else 0.5
        if left <= 0:
            raise QueueWaitExpired("the inference queue wait deadline expired; retry later")
        return min(0.5, left)

    def executing(self, prompt_tokens, silence_s, *, owner=None, chunk_max=32768, prompt_floor=50.0, prompt_slack=3.0):
        if self.phase == "queued":
            self.queue_s += self.clock() - self.queued_at
        self.cancel._strata_queue_deadline = None
        self.owner = owner
        self.silence_s = float(silence_s or 0)
        self.chunk_max, self.prompt_floor, self.prompt_slack = chunk_max, prompt_floor, prompt_slack
        self.allow_s = self.silence_s + min(prompt_tokens, chunk_max) / prompt_floor if self.silence_s else 0.0
        self.heard = self.beat = self.clock()
        self.paused_at = None
        self.read_to = 0
        self.stall_state.clear()
        self._phase("executing")

    def progress(self, line):
        if self.heard is None:
            return
        self.heard = self.beat = self.clock()
        if self.paused_at is not None:
            self.paused_at = self.heard
        self.stall_state.clear()
        if line.startswith(("T ", "BT ")):
            self.allow_s = self.silence_s
        elif line.startswith("PP "):
            fields = line.split()
            if len(fields) >= 5 and fields[1].isdigit():
                try:
                    read, rate = int(fields[1]), float(fields[4])
                except ValueError:
                    return
                chunk, self.read_to = read - self.read_to, read
                if self.silence_s > 0 and rate > 0 and chunk > 0:
                    self.allow_s = max(self.silence_s, self.prompt_slack * chunk / rate)
        elif line.startswith("RESUME "):
            try:
                self.read_to = int(line.split()[1])
            except (IndexError, ValueError):
                pass

    def execution_wait(self):
        if self.heard is None or self.paused_at is not None or self.allow_s <= 0:
            return 0.5
        left = self.allow_s - (self.clock() - self.heard)
        if left <= 0:
            raise ExecutionStalled(f"the engine said nothing for {self.clock() - self.heard:.0f} s during the request")
        return min(0.5, left)

    def execution_paused(self, paused):
        """An adapter may suspend execution while another watched native operation owns the scheduler."""
        if paused and self.heard is not None and self.paused_at is None:
            self.paused_at = self.clock()
            self.stall_state.clear()
        elif not paused and self.paused_at is not None:
            self.heard += self.clock() - self.paused_at
            self.paused_at = None

    def heartbeat(self):
        if self.clock() - self.beat >= self.heartbeat_s:
            self.beat = self.clock()
            return True
        return False

    def finished(self):
        self.heard = None
        self.paused_at = None
        self._phase("finished")


def lifecycle(cancel, *, heartbeat_s=10.0):
    """Keep the existing Event-based Engine API; all adapters share one request context."""
    current = getattr(cancel, "_strata_lifecycle", None)
    if current is None:
        current = RequestLifecycle(cancel, heartbeat_s=heartbeat_s)
        cancel._strata_lifecycle = current
    return current


class AdmissionQueue:
    """Capacity credits for a protocol whose native engine chooses its own slots.

    Slot-addressed adapters retain their native slot allocator. Both use the same
    RequestLifecycle for the policy and report execution at native submission.
    """
    def __init__(self, owner, capacity):
        self.owner, self.capacity = owner, capacity
        self.cv = threading.Condition()
        self.waiting = collections.deque()
        self.active = 0

    def acquire(self, request, valid):
        ticket = object()
        claimed = False
        request.queued()
        with self.cv:
            self.waiting.append(ticket)
        try:
            while True:
                with self.cv:
                    wait = request.queue_wait()
                    if wait == 0:
                        return False
                    valid(self.owner)
                    if self.waiting[0] is ticket and self.active < self.capacity:
                        self.waiting.popleft()
                        self.active += 1
                        claimed = True
                        self.cv.notify_all()
                        return True
                    self.cv.wait(timeout=wait)
                if request.heartbeat():
                    yield None
        finally:
            if not claimed:
                with self.cv:
                    self.waiting.remove(ticket)
                    self.cv.notify_all()

    def release(self):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()
