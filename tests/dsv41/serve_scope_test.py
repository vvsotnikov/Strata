#!/usr/bin/env python3
"""serve_scope_test.py <ds4_serve_fake>

Cache scopes in the serve loop (ds4_main.cpp over tests/dsv41/serve_fake_model.cpp: no GPU, no weights).  A request
names its scope with GEN scope=<id>; what it reuses - the state in the model, a kept conversation in RAM or in the
folder, a restore point, a shared start, a picture's rows - must be its own scope's, with one conversation and with
--parallel.  The reused count is DONE's; a request that reused another scope's state would show it there.
"""
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

EXE = None
ANSWER = 2          # tokens each request generates
IMAGE_TOKEN = 3     # serve_fake_model.cpp's
SPAN = 4            # the tokens a picture takes there
IDENTITY = b"ds4-fake-model-1"


def prompt(seed, n):
    """n token ids no other seed shares (none of them the image token or the end token)."""
    return [10 + (seed * 7919 + i * 31) % 900 for i in range(n)]


class Engine:
    def __init__(self, folder, parallel=0, env=None):
        self.parallel = parallel
        self.rid = 0
        self.log = open(os.path.join(folder, "engine.log"), "a+")
        args = [EXE, "--serve", "--pack", folder, "--max-context", "65536", "--eos", "1"]
        if parallel:
            args += ["--parallel", str(parallel)]
        full = {**os.environ, "STRATA_DS4_SLOT_KEEP": "1", "STRATA_DS4_SLOT_DIR": os.path.join(folder, "slots"),
                **(env or {})}
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                  bufsize=1, env=full)
        self.q = queue.Queue()
        threading.Thread(target=lambda: [self.q.put(line.rstrip("\n")) for line in self.p.stdout] + [self.q.put(None)],
                         daemon=True).start()
        self.notes = []                                  # NOTE lines so far (the folder's come after READY)
        self.cache = {}                                  # the last CACHE line: what the engine keeps, in numbers
        while True:
            line = self.q.get(timeout=30)
            if line is None:
                raise RuntimeError("the engine ended before READY")
            if line.startswith("READY"):
                break

    def send(self, ids, scope=None, keys="", pictures=None):
        """The request's line written; -> its rid (None with one conversation)."""
        rid = None
        if self.parallel:
            self.rid += 1
            rid = self.rid
            keys += f" rid={rid}"
        if scope is not None:
            keys += f" scope={scope}"
        head = f"GENI {ANSWER}{keys} {pictures}" if pictures else f"GEN {ANSWER}{keys}"
        self.p.stdin.write(f"{head} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        return rid

    def done(self, rid=None):
        """-> (reused, finish) of the request's DONE, or ("ERR", text)."""
        while True:
            line = self.q.get(timeout=30)
            if line is None:
                raise RuntimeError("the engine died")
            f = line.split()
            if f[0] == "NOTE":
                self.notes.append(line)
            if f[0] == "CACHE":
                self.cache = {k: int(v) for k, v in (item.split("=") for item in f[1:])}
            if f[0] == "ERR" and "expected: GEN" in line:    # a line the engine did not take as a request
                return "ERR", line
            if f[0] == "DONE":
                if rid is not None:
                    if f[1] != str(rid):
                        continue
                    f = f[:1] + f[2:]
                return int(f[8]), f[5]

    def gen(self, ids, scope=None, keys="", pictures=None):
        return self.done(self.send(ids, scope, keys, pictures))[0]

    def quit(self):
        self.p.stdin.write("QUIT\n")
        self.p.stdin.flush()
        self.p.wait(timeout=30)
        self.close()

    def close(self):
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait()
        for stream in (self.p.stdin, self.p.stdout, self.log):
            stream.close()


class Scopes(unittest.TestCase):
    parallel = 0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.engines = []

    def tearDown(self):
        for e in self.engines:
            if not e.log.closed:
                e.close()
        self.tmp.cleanup()

    def engine(self, env=None):
        e = Engine(self.dir, self.parallel, env)
        self.engines.append(e)
        return e

    def test_a_repeated_prompt_is_reused_by_its_own_scope_only(self):
        e = self.engine()
        p = prompt(1, 2000)
        self.assertEqual(e.gen(p, "aaaa"), 0)
        self.assertEqual(e.gen(p, "aaaa"), len(p) - 1)
        self.assertEqual(e.gen(p, "bbbb"), 0)            # the same tokens from another scope: nothing reused
        self.assertEqual(e.gen(p, "bbbb"), len(p) - 1)   # its own repeat is
        self.assertEqual(e.gen(p, "aaaa"), len(p) - 1)   # and the first scope's conversation is still there
        self.assertEqual(e.gen(p), 0)                    # no scope is a scope of its own
        self.assertEqual(e.gen(p), len(p) - 1)
        self.assertEqual(e.gen(p, "aaaa"), len(p) - 1)

    def test_a_longer_prompt_goes_on_from_its_own_scope_only(self):
        e = self.engine()
        p = prompt(2, 2000)
        longer = p + prompt(3, 600)
        self.assertEqual(e.gen(p, "aaaa"), 0)
        self.assertEqual(e.gen(longer, "bbbb"), 0)       # not from the first scope's state, nor its restore points
        self.assertGreaterEqual(e.gen(longer, "aaaa"), len(p) - 1)
        self.assertEqual(e.gen(p, "cccc"), 0)            # a prompt that is a prefix of another scope's conversation

    def test_restore_points_are_a_scope_s_own(self):
        e = self.engine()
        base = prompt(4, 3000)
        first = base + prompt(5, 300)
        second = base[:2600] + prompt(6, 300)            # leaves the first prompt after its mark at 2500
        self.assertEqual(e.gen(first, "aaaa", " marks=2500"), 0)
        self.assertEqual(e.gen(second, "bbbb", " marks=2500"), 0)
        self.assertEqual(e.gen(second, "aaaa", " marks=2500"), 2500)

    def test_a_shared_start_is_a_scope_s_own(self):
        for inside in (False, True):                     # (kept at a read stop, and inside one run of chunks)
            with self.subTest(inside=inside):
                env = {"STRATA_DS4_SLOT_KEEP": "0", **({"DS4_FAKE_MARKS_INSIDE": "1"} if inside else {})}
                e = self.engine(env)
                start = prompt(7, 1500)
                self.assertEqual(e.gen(start + prompt(8, 300), "aaaa", " shared=1500"), 0)
                self.assertEqual(e.gen(prompt(9, 2000), "aaaa"), 0)            # another conversation takes the model
                self.assertEqual(e.gen(start + prompt(10, 300), "bbbb", " shared=1500"), 0)
                self.assertEqual(e.gen(prompt(11, 2000), "bbbb"), 0)
                self.assertEqual(e.gen(start + prompt(12, 300), "aaaa", " shared=1500"), 1500)
                self.assertEqual(e.gen(prompt(13, 2000), "cccc"), 0)
                self.assertEqual(e.gen(start + prompt(14, 300), "bbbb", " shared=1500"), 1500)
                self.assertEqual(e.gen(start + prompt(15, 300), "cccc", " shared=1500"), 0)
                e.quit()

    def test_kept_conversations_keep_their_scope_across_a_restart(self):
        e = self.engine()
        p, other = prompt(16, 2000), prompt(17, 2000)
        self.assertEqual(e.gen(p, "aaaa"), 0)
        self.assertEqual(e.gen(other, "bbbb"), 0)
        e.quit()
        e = self.engine()
        # (the engine's end writes the conversation it last ran, and with one conversation the ones set aside before)
        kept = 1 if self.parallel else 2
        self.assertEqual(e.gen(other + prompt(18, 100), "aaaa"), 0)
        self.assertTrue(any(f": {kept} conversation(s) kept from earlier runs" in n for n in e.notes), e.notes)
        self.assertEqual(e.gen(other + prompt(18, 100)), 0)
        self.assertGreaterEqual(e.gen(other + prompt(18, 100), "bbbb"), len(other) - 1)
        if not self.parallel:
            self.assertEqual(e.gen(p + prompt(19, 100), "bbbb"), 0)
            self.assertGreaterEqual(e.gen(p + prompt(19, 100), "aaaa"), len(p) - 1)

    def test_a_conversation_kept_before_scopes_is_removed_and_reused_by_nobody(self):
        slots = os.path.join(self.dir, "slots")
        os.makedirs(slots)
        p = prompt(20, 2000)
        state = b"state of a build without scopes"
        with open(os.path.join(slots, "slot-1-1.bin"), "wb") as f:
            f.write(state)
        with open(os.path.join(slots, "slot-1-1.meta"), "wb") as f:      # the layout before scopes: "DS4META2"
            f.write(b"DS4META2" + struct.pack("<Q", len(IDENTITY)) + IDENTITY + struct.pack("<QQ", len(state), len(p)) +
                    struct.pack(f"<{len(p)}i", *p) + struct.pack("<Q", 0) + b"DS4META2")
        e = self.engine()
        self.assertEqual(e.gen(p + prompt(21, 100)), 0)
        self.assertEqual(os.listdir(slots), [])
        self.assertTrue(any("1 not this model's, incomplete or without a scope - removed" in n for n in e.notes), e.notes)
        self.assertEqual(e.gen(p + prompt(22, 100), "aaaa"), 0)

    def test_a_scope_the_engine_cannot_keep_is_refused(self):
        e = self.engine()
        p = prompt(23, 2000)
        self.assertEqual(e.gen(p, "aaaa"), 0)
        for bad in ("a=b", "a/b", "x" * 129):
            e.send(p, bad)
            self.assertEqual(e.done()[0], "ERR", bad)    # (not read as "no scope", nor as the scope before it)
        self.assertEqual(e.gen(p, "aaaa"), len(p) - 1)

    def test_a_picture_is_encoded_again_for_another_scope(self):
        log = os.path.join(self.dir, "encodes.log")
        e = self.engine({"DS4_FAKE_ENCODE_LOG": log})
        pictures = os.path.join(self.dir, "pictures.dsi")
        with open(pictures, "wb") as f:
            f.write(b"DSI1" + struct.pack("<i", 1) + struct.pack("<qii", 5, 2, 2) + bytes(12))

        def ask(seed, scope):
            ids = prompt(seed, 5) + [IMAGE_TOKEN] * SPAN + prompt(seed + 1, 40)
            self.assertEqual(e.gen(ids, scope, pictures=pictures), 0)
            with open(log) as f:
                return len(f.readlines())
        self.assertEqual(ask(30, "aaaa"), 1)
        self.assertEqual(ask(32, "aaaa"), 1)             # the same picture in the scope's next conversation: its rows kept
        self.assertEqual(ask(34, "bbbb"), 2)             # another scope's request encodes it itself
        self.assertEqual(ask(36, "bbbb"), 2)
        self.assertEqual(ask(38, None), 3)


    def test_the_folder_s_cap_gives_up_the_least_recently_used_and_the_cache_line_counts_it(self):
        # (a 2000-token conversation of the fake model is 8 KB: the folder holds two, RAM one)
        e = self.engine({"STRATA_DS4_SLOT_CAP": "1", "STRATA_DS4_SLOT_GB": str(20000 / 2**30)})
        kept = [prompt(70 + i, 2000) for i in range(10)]
        for i, p in enumerate(kept):
            self.assertEqual(e.gen(p, "aaaa" if i % 2 else "bbbb"), 0)
        self.assertEqual(e.gen(prompt(80, 2000), "cccc"), 0)     # (the last leaves the model: every one is set aside)
        c = e.cache
        self.assertEqual((c["ram_cap"], c["folder_cap_bytes"]), (1, 20000))
        self.assertLessEqual(c["folder_bytes"], c["folder_cap_bytes"])
        self.assertGreaterEqual(c["saved"], 3)                   # (--parallel keeps four in the model)
        self.assertGreater(c["evicted"], 0)
        self.assertGreater(c["evicted_bytes"], 0)
        self.assertEqual(c["folder"] * 2, len(os.listdir(os.path.join(self.dir, "slots"))))
        if not self.parallel:
            self.assertEqual((c["ram"], c["folder"], c["saved"], c["evicted"]), (1, 2, 10, 7))
            self.assertEqual(c["scopes"], 2)
            # the ones given up are the least recently used, whoever owns them; the latest comes back
            self.assertGreaterEqual(e.gen(kept[9] + prompt(81, 50), "aaaa"), 1999)
            self.assertEqual(e.cache["taken"], 1)
            self.assertEqual(e.gen(kept[0] + prompt(81, 50), "bbbb"), 0)
            self.assertEqual(e.gen(kept[3] + prompt(81, 50), "aaaa"), 0)

    def test_a_scope_that_fills_the_folder_gives_up_its_own_conversations(self):
        # (the folder holds two 2000-token conversations, RAM one)
        e = self.engine({"STRATA_DS4_SLOT_CAP": "1", "STRATA_DS4_SLOT_GB": str(20000 / 2**30)})
        quiet = prompt(100, 2000)
        self.assertEqual(e.gen(quiet, "bbbb"), 0)
        for i in range(12):                              # another scope keeps opening conversations
            self.assertEqual(e.gen(prompt(101 + i, 2000), "aaaa"), 0)
        self.assertGreater(e.cache["evicted"], 0)
        self.assertEqual(e.cache["scopes"], 2)
        # the quiet scope's only conversation is the least recently used of all, and is still there
        self.assertGreaterEqual(e.gen(quiet + prompt(120, 50), "bbbb"), 1999)

    def test_a_scope_that_fills_the_shared_starts_gives_up_its_own(self):
        e = self.engine({"STRATA_DS4_SLOT_KEEP": "0", "STRATA_DS4_PREFIX_KEEP": "2"})
        quiet = prompt(130, 1500)
        self.assertEqual(e.gen(quiet + prompt(131, 300), "bbbb", " shared=1500"), 0)
        for i in range(3):                               # another scope keeps three starts where two fit
            self.assertEqual(e.gen(prompt(132 + i, 1500) + prompt(140, 300), "aaaa", " shared=1500"), 0)
        self.assertEqual(e.cache["starts"], 2)
        for i in range(5):                               # (every conversation of the model is taken by others)
            self.assertEqual(e.gen(prompt(150 + i, 2000), "cccc"), 0)
        self.assertEqual(e.gen(quiet + prompt(160, 300), "bbbb", " shared=1500"), 1500)

    def test_a_scope_that_fills_the_pictures_gives_up_its_own(self):
        log = os.path.join(self.dir, "encodes.log")
        e = self.engine({"DS4_FAKE_ENCODE_LOG": log, "STRATA_DS4_PICTURE_KEEP": "2"})
        files = []
        for n, side in enumerate((2, 4, 6, 8)):
            files.append(os.path.join(self.dir, f"picture-{n}.dsi"))
            with open(files[-1], "wb") as f:
                f.write(b"DSI1" + struct.pack("<i", 1) + struct.pack("<qii", 5, side, side) + bytes(side * side * 3))

        def ask(seed, scope, picture):
            ids = prompt(seed, 5) + [IMAGE_TOKEN] * SPAN + prompt(seed + 1, 40)
            self.assertEqual(e.gen(ids, scope, pictures=picture), 0)
            with open(log) as f:
                return len(f.readlines())
        self.assertEqual(ask(170, "bbbb", files[0]), 1)
        for n in (1, 2, 3):                              # another scope sends three pictures where two fit
            self.assertEqual(ask(170 + 2 * n, "aaaa", files[n]), 1 + n)
        self.assertEqual(ask(180, "bbbb", files[0]), 4)  # the quiet scope's picture was not encoded again
        self.assertEqual(ask(182, "aaaa", files[1]), 5)  # the busy scope's oldest was

    def test_the_number_of_kept_pictures_is_a_setting(self):
        for keep, encodes in ((None, 2), ("1", 3)):
            with self.subTest(keep=keep):
                log = os.path.join(self.dir, f"encodes-{keep}.log")
                e = self.engine({"DS4_FAKE_ENCODE_LOG": log, **({"STRATA_DS4_PICTURE_KEEP": keep} if keep else {})})
                files = []
                for n, side in enumerate((2, 4)):
                    files.append(os.path.join(self.dir, f"picture-{keep}-{n}.dsi"))
                    with open(files[-1], "wb") as f:
                        f.write(b"DSI1" + struct.pack("<i", 1) + struct.pack("<qii", 5, side, side) + bytes(side * side * 3))
                for seed, picture in ((90, files[0]), (92, files[1]), (94, files[0])):
                    ids = prompt(seed, 5) + [IMAGE_TOKEN] * SPAN + prompt(seed + 1, 40)
                    self.assertEqual(e.gen(ids, "aaaa", pictures=picture), 0)
                with open(log) as f:
                    self.assertEqual(len(f.readlines()), encodes)
                self.assertEqual(e.cache["pictures_cap"], int(keep or 8))
                e.quit()


    def test_a_request_that_keeps_nothing_reuses_nothing(self):
        e = self.engine()
        p = prompt(200, 2000)
        self.assertEqual(e.gen(p, "aaaa", " keep=0"), 0)
        self.assertEqual(e.gen(p, "aaaa", " keep=0"), 0)            # its repeat: nothing was left to reuse
        self.assertEqual(e.gen(p, "aaaa"), 0)                       # nor for the scope's next request
        self.assertEqual(e.gen(p, "aaaa"), len(p) - 1)              # which keeps as before
        self.assertEqual(e.gen(p + prompt(201, 100), "aaaa", " keep=0"), 0)   # what the scope kept is not read
        # (and not lost: the scope's own conversation was set aside, not dropped, when the request took the model)
        self.assertGreaterEqual(e.gen(p + prompt(202, 100), "aaaa"), len(p) - 1)

    def test_a_request_that_keeps_nothing_leaves_nothing(self):
        e = self.engine({"STRATA_DS4_SLOT_CAP": "1"})
        start = prompt(210, 1500)
        for i in range(6):                               # more than the model holds, so each would be set aside
            self.assertEqual(e.gen(start + prompt(211 + i, 600), "aaaa", " keep=0 shared=1500 marks=1600 tail=2000"), 0)
        for i in range(5):                               # other requests take every conversation of the model
            self.assertEqual(e.gen(prompt(220 + i, 2000), "bbbb", " keep=0"), 0)
        c = e.cache
        self.assertEqual((c["ram"], c["folder"], c["starts"], c["scopes"], c["saved"], c["taken"]), (0, 0, 0, 0, 0, 0))
        self.assertEqual(os.listdir(os.path.join(self.dir, "slots")), [])
        self.assertEqual(e.gen(start + prompt(211, 600), "aaaa"), 0)          # no shared start or restore point
        e.quit()                                         # (the engine's end writes what the model holds: that one only)
        e = self.engine()
        self.assertEqual(e.gen(prompt(225, 2000), "cccc"), 0)
        self.assertTrue(any(": 1 conversation(s) kept from earlier runs" in n for n in e.notes), e.notes)

    def test_a_picture_of_a_request_that_keeps_nothing_is_not_kept(self):
        log = os.path.join(self.dir, "encodes.log")
        e = self.engine({"DS4_FAKE_ENCODE_LOG": log})
        pictures = os.path.join(self.dir, "pictures.dsi")
        with open(pictures, "wb") as f:
            f.write(b"DSI1" + struct.pack("<i", 1) + struct.pack("<qii", 5, 2, 2) + bytes(12))

        def ask(seed, keys=""):
            ids = prompt(seed, 5) + [IMAGE_TOKEN] * SPAN + prompt(seed + 1, 40)
            self.assertEqual(e.gen(ids, "aaaa", keys, pictures=pictures), 0)
            with open(log) as f:
                return len(f.readlines())
        self.assertEqual(ask(230, " keep=0"), 1)
        self.assertEqual(ask(232, " keep=0"), 2)          # encoded again: its rows were not kept
        self.assertEqual(ask(234), 3)                      # nor for the scope's next request
        self.assertEqual(ask(236), 3)                      # whose rows are kept as before


    def test_kept_state_past_its_time_is_deleted(self):
        ttl = 3                                          # seconds (STRATA_CACHE_TTL_MINUTES is 0.05)
        log = os.path.join(self.dir, "encodes.log")
        e = self.engine({"STRATA_CACHE_TTL_MINUTES": str(ttl / 60), "STRATA_DS4_SLOT_CAP": "1",
                         "DS4_FAKE_ENCODE_LOG": log})
        pictures = os.path.join(self.dir, "pictures.dsi")
        with open(pictures, "wb") as f:
            f.write(b"DSI1" + struct.pack("<i", 1) + struct.pack("<qii", 5, 2, 2) + bytes(12))
        start = prompt(240, 1500)
        convs = [prompt(241 + i, 2000) for i in range(6)]
        self.assertEqual(e.gen(start + prompt(250, 300), "aaaa", " shared=1500"), 0)
        for p in convs:                                  # in the model, in RAM and in the folder
            self.assertEqual(e.gen(p, "aaaa"), 0)
        self.assertEqual(e.gen(prompt(251, 5) + [IMAGE_TOKEN] * SPAN + prompt(252, 40), "aaaa", pictures=pictures), 0)
        slots = os.path.join(self.dir, "slots")
        self.assertGreater(len(os.listdir(slots)), 0)
        self.assertGreater(e.cache["starts"], 0)
        time.sleep(ttl + 1.5)                            # no request: the engine deletes them on its own
        self.assertEqual(os.listdir(slots), [])
        for p in convs:                                  # nothing past its time is reused, the model's own included
            self.assertEqual(e.gen(p + prompt(253, 50), "aaaa"), 0)
        self.assertEqual(e.gen(start + prompt(254, 300), "aaaa", " shared=1500"), 0)
        self.assertEqual(e.gen(prompt(255, 5) + [IMAGE_TOKEN] * SPAN + prompt(256, 40), "aaaa", pictures=pictures), 0)
        with open(log) as f:
            self.assertEqual(len(f.readlines()), 2)      # the picture was encoded again
        self.assertGreater(e.cache["expired"], 0)
        self.assertEqual(e.cache["ttl_s"], ttl)
        # within its time an entry is reused as before
        self.assertGreaterEqual(e.gen(convs[-1] + prompt(253, 50) + prompt(257, 50), "aaaa"), len(convs[-1]))

    def test_a_folder_left_past_its_time_is_emptied_at_start(self):
        e = self.engine()
        for i in range(3):
            self.assertEqual(e.gen(prompt(260 + i, 2000), "aaaa"), 0)
        e.quit()
        slots = os.path.join(self.dir, "slots")
        self.assertGreater(len(os.listdir(slots)), 0)
        old = time.time() - 120
        for name in os.listdir(slots):
            os.utime(os.path.join(slots, name), (old, old))
        e = self.engine({"STRATA_CACHE_TTL_MINUTES": "1"})
        self.assertEqual(e.gen(prompt(260, 2000) + prompt(263, 50), "aaaa"), 0)
        self.assertEqual(os.listdir(slots), [])


class ParallelScopes(Scopes):
    parallel = 4

    def test_a_scope_s_conversation_survives_another_scope_taking_its_place(self):
        e = self.engine()
        mine = [prompt(40 + i, 2000) for i in range(4)]
        for p in mine:                                   # every conversation of the model holds the first scope's
            self.assertEqual(e.gen(p, "aaaa"), 0)
        for p in mine:                                   # another scope sends the same prompts: none reused
            self.assertEqual(e.gen(p, "bbbb"), 0)
        for p in mine:                                   # each scope's own are taken back (the model's, RAM or folder)
            self.assertGreaterEqual(e.gen(p + prompt(50, 100), "aaaa"), len(p) - 1)
        for p in mine:
            self.assertGreaterEqual(e.gen(p + prompt(51, 100), "bbbb"), len(p) - 1)

    def test_requests_of_two_scopes_at_once_share_nothing(self):
        e = self.engine()
        p = prompt(60, 2000)
        self.assertEqual(e.gen(p, "aaaa"), 0)
        first, second = e.send(p + prompt(61, 50), "aaaa"), e.send(p + prompt(61, 50), "bbbb")
        got = {}
        while len(got) < 2:
            line = e.q.get(timeout=30)
            f = line.split()
            if f[0] == "DONE":
                got[int(f[1])] = int(f[9])
        self.assertGreaterEqual(got[first], len(p) - 1)
        self.assertEqual(got[second], 0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    EXE = sys.argv.pop(1)
    unittest.main()
