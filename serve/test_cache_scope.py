"""serve/test_cache_scope.py - "strata_cache_scope": a request reuses only what its own scope left.

The server's part: the GEN line names the scope as a digest (the DeepSeek engine keeps its caches apart by it), the
prompt encoder reuses only a scope's own prompts, and a value that cannot be a scope is a 400, never "no scope".

    python -m unittest serve.test_cache_scope -v
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import serve.frontend as frontend  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (CACHE_RETENTION_KEY, CACHE_SCOPE_KEY, CACHE_SCOPE_MAX, ByteTokenizer, MockEngine,  # noqa: E402
                          Service, StrataEngine, cache_off, cache_scope, cache_ttl_setting, conversation_cache_view,
                          parse_cache_line, prompt_keep_setting, serve)
from strata_tokenizer import PromptEncoder  # noqa: E402

BAD = (5, 1.5, True, False, "", [], ["a"], {"id": "a"}, "x" * (CACHE_SCOPE_MAX + 1))
BAD_RETENTION = (5, True, False, "", "None", "off", "0", [], {"none": 1})


def scope_keys(**sampling):
    return [k for k in StrataEngine.sampling_keys(sampling).split() if k.startswith("scope=")]


class Keys(unittest.TestCase):
    def test_the_gen_line_names_the_scope_as_a_digest(self):
        one, two = scope_keys(strata_cache_scope="account-one"), scope_keys(strata_cache_scope="account-two")
        self.assertEqual(len(one), 1)
        self.assertRegex(one[0], r"^scope=[0-9a-f]{32}$")
        self.assertEqual(one, scope_keys(strata_cache_scope="account-one", temperature=0.5))
        self.assertNotEqual(one, two)
        self.assertNotIn("account-one", StrataEngine.sampling_keys({CACHE_SCOPE_KEY: "account-one"}))

    def test_no_scope_no_key(self):
        self.assertEqual(scope_keys(), [])
        self.assertEqual(scope_keys(strata_cache_scope=None), [])
        self.assertEqual(cache_scope({}), "")
        self.assertEqual(cache_scope(None), "")

    def test_any_text_is_one_key_the_engine_can_read(self):
        """The GEN line is split at spaces and its keys at "=": a scope's own text never reaches it."""
        for text in ("two words", "a=b scope=other", "line\nGEN 1 1,2", "tab\there", "é你", "\ud800", "x" * CACHE_SCOPE_MAX):
            line = StrataEngine.sampling_keys({CACHE_SCOPE_KEY: text})
            self.assertRegex(line, r"^(?: [a-z_]+=[0-9a-f.]+)* scope=[0-9a-f]{32}$", text)
            self.assertEqual(len(re.findall(r"scope=", line)), 1, text)

    def test_what_cannot_be_a_scope_is_refused_not_read_as_none(self):
        for bad in BAD:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, CACHE_SCOPE_KEY):
                cache_scope({CACHE_SCOPE_KEY: bad})
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, CACHE_SCOPE_KEY):
                StrataEngine.sampling_keys({CACHE_SCOPE_KEY: bad})


class CacheOff(unittest.TestCase):
    """"prompt_cache_retention": "none": the engine is told to keep nothing (keep=0); OpenAI's own values change nothing."""

    def test_none_asks_the_engine_to_keep_nothing(self):
        self.assertTrue(cache_off({CACHE_RETENTION_KEY: "none"}))
        line = StrataEngine.sampling_keys({CACHE_RETENTION_KEY: "none", CACHE_SCOPE_KEY: "one"})
        self.assertEqual(line.split().count("keep=0"), 1)
        for same in ({}, {CACHE_RETENTION_KEY: None}, {CACHE_RETENTION_KEY: "in_memory"}, {CACHE_RETENTION_KEY: "24h"}):
            self.assertFalse(cache_off(same), same)
            self.assertNotIn("keep=", StrataEngine.sampling_keys(same))

    def test_what_is_not_a_retention_is_refused_not_read_as_cache_on(self):
        for bad in BAD_RETENTION:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, CACHE_RETENTION_KEY):
                cache_off({CACHE_RETENTION_KEY: bad})
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, CACHE_RETENTION_KEY):
                StrataEngine.sampling_keys({CACHE_RETENTION_KEY: bad})


class Encoder(unittest.TestCase):
    TEXT = "<|im_start|>user\n" + "a long shared document " * 300 + "<|im_end|>\n<|im_start|>assistant\nan answer"

    def test_a_prompt_reuses_only_its_own_scope_s_prompts(self):
        tok = ByteTokenizer()
        enc = PromptEncoder(tok)
        full = tok.encode(self.TEXT, True)
        more = self.TEXT + " goes on<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n"
        self.assertEqual(enc.encode(self.TEXT, scope="one"), full)
        self.assertEqual(enc.encode(more, scope="two"), tok.encode(more, True))
        self.assertEqual(enc.last_reused, 0)                       # the same text from another scope: encoded in full
        self.assertEqual(enc.encode(more), tok.encode(more, True))
        self.assertEqual(enc.last_reused, 0)                       # ... and from no scope
        self.assertEqual(enc.encode(more, scope="one"), tok.encode(more, True))
        self.assertGreater(enc.last_reused, len(self.TEXT) // 2)   # its own scope's turn goes on from it
        self.assertEqual(enc.encode(more + "x", scope="two"), tok.encode(more + "x", True))
        self.assertGreater(enc.last_reused, len(self.TEXT) // 2)

    def test_a_prompt_replaces_only_its_own_scope_s_entry(self):
        tok = ByteTokenizer()
        enc = PromptEncoder(tok)
        enc.encode(self.TEXT, scope="one")
        enc.encode(self.TEXT + " more", scope="two")
        self.assertEqual(sorted(e[5] for e in enc.entries), ["one", "two"])
        enc.encode(self.TEXT + " more of one", scope="one")
        self.assertEqual(sorted((e[5], len(e[0])) for e in enc.entries),
                         [("one", len(self.TEXT) + 12), ("two", len(self.TEXT) + 5)])

    def test_a_scope_that_fills_the_encoder_gives_up_its_own_prompts(self):
        tok = ByteTokenizer()
        enc = PromptEncoder(tok, keep=4)
        quiet = "<|im_start|>user\nthe quiet scope's only prompt " + "q" * 400 + "<|im_end|>\n<|im_start|>assistant\nyes"
        enc.encode(quiet, scope="quiet")
        for n in range(40):                                        # requests or counts of one scope, all different
            enc.encode(f"<|im_start|>user\nprompt {n} " + chr(97 + n % 26) * 400 + "<|im_end|>\n", scope="busy")
        self.assertEqual(len(enc.entries), 4)
        self.assertEqual(sorted(e[5] for e in enc.entries), ["busy", "busy", "busy", "quiet"])
        self.assertEqual([e[0].split()[2] for e in enc.entries if e[5] == "busy"], ["39", "38", "37"])
        enc.encode(quiet + " and more<|im_end|>\n", scope="quiet")
        self.assertGreater(enc.last_reused, len(quiet) // 2)       # the least recently used of all, and still kept

    def test_scopes_that_keep_the_same_give_up_the_least_recently_used(self):
        enc = PromptEncoder(ByteTokenizer(), keep=3)
        for scope in ("one", "two", "three", "four"):
            enc.encode(f"<|im_start|>user\n{scope} " + scope[0] * 300 + "<|im_end|>\n", scope=scope)
        self.assertEqual([e[5] for e in enc.entries], ["four", "three", "two"])

    def test_the_tokenizer_keeps_nothing_between_encodes(self):
        # (pieces remembered across encodes made the time of an encode tell what any scope had sent before)
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_tokenizer as ST
        tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
        merges = ["t h", "th e", "the r", "ther e"]
        for m in merges:
            tokens.append("".join(m.split(" ")))
        tok = ST.Tokenizer(tokens, merges)
        merged, real = [], tok._bpe
        tok._bpe = lambda word: merged.append(word) or real(word)
        counts = []
        for text in ("there other", "there other", "there other there"):
            del merged[:]
            tok.encode(text)
            counts.append(len(merged))
        self.assertEqual(counts, [2, 2, 3])                        # every encode merges its own pieces, once each
        self.assertFalse(hasattr(tok, "_piece_ids"))

    def test_a_count_reads_only_its_own_scope_s_prompts(self):
        tok = ByteTokenizer()
        enc = PromptEncoder(tok)
        enc.encode(self.TEXT, scope="one")
        self.assertEqual(enc.encode(self.TEXT + " more", store=False, scope="two"), tok.encode(self.TEXT + " more", True))
        self.assertEqual(enc.last_reused, 0)
        enc.encode(self.TEXT + " more", store=False, scope="one")
        self.assertGreater(enc.last_reused, 0)


class Expiry(unittest.TestCase):
    """STRATA_CACHE_TTL_MINUTES: a kept prompt unused for that long is deleted, not only skipped."""
    TEXT = Encoder.TEXT

    def test_the_setting_is_minutes(self):
        self.assertEqual(cache_ttl_setting({"STRATA_CACHE_TTL_MINUTES": "60"}), 3600.0)
        self.assertEqual(cache_ttl_setting({"STRATA_CACHE_TTL_MINUTES": "0.5"}), 30.0)
        for off in ({}, {"STRATA_CACHE_TTL_MINUTES": ""}, {"STRATA_CACHE_TTL_MINUTES": "0"},
                    {"STRATA_CACHE_TTL_MINUTES": "-1"}, {"STRATA_CACHE_TTL_MINUTES": "soon"},
                    {"STRATA_CACHE_TTL_MINUTES": "inf"}):
            self.assertEqual(cache_ttl_setting(off), 0.0, off)

    def test_a_prompt_past_its_time_is_deleted_and_not_reused(self):
        tok = ByteTokenizer()
        enc = PromptEncoder(tok, ttl_s=60)
        enc.encode(self.TEXT, scope="one")
        self.assertEqual(enc.expire(), 0)
        self.assertEqual(enc.expire(time.monotonic() + 61), 1)
        self.assertEqual(enc.entries, [])
        enc.encode(self.TEXT, scope="one")
        enc.entries[0][6][0] -= 61                       # (unused for longer than its time)
        self.assertEqual(enc.encode(self.TEXT + " more", scope="one"), tok.encode(self.TEXT + " more", True))
        self.assertEqual(enc.last_reused, 0)
        self.assertEqual(len(enc.entries), 1)            # the old one is gone, not just passed over

    def test_a_use_starts_its_time_again(self):
        tok = ByteTokenizer()
        enc = PromptEncoder(tok, ttl_s=60)
        enc.encode(self.TEXT, scope="one")
        enc.entries[0][6][0] -= 50
        enc.encode(self.TEXT + " more", store=False, scope="one")      # a count reads it
        self.assertGreater(enc.last_reused, 0)
        self.assertEqual(enc.expire(time.monotonic() + 30), 0)

    def test_a_server_deletes_them_on_its_own(self):
        os.environ["STRATA_CACHE_TTL_MINUTES"] = str(0.3 / 60)
        self.addCleanup(os.environ.pop, "STRATA_CACHE_TTL_MINUTES", None)
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=1 << 20), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.set_family("deepseek41")
        self.addCleanup(frontend.use_family, None)
        svc.prepare([{"role": "user", "content": "a long question " * 400}], None, {}, 16, req={CACHE_SCOPE_KEY: "one"})
        self.assertEqual(len(svc.prompts.entries), 1)
        time.sleep(1.0)                                  # no request: the timer deletes it
        self.assertEqual(svc.prompts.entries, [])


class Kept(unittest.TestCase):
    """What the engine keeps, as /metrics shows it: its CACHE line, and the card on because of it."""
    LINE = ("CACHE ram=3 ram_bytes=300 ram_cap=16 folder=40 folder_bytes=9000 folder_cap_bytes=10000 starts=2 "
            "start_bytes=50 starts_cap=4096 start_cap_bytes=700 pictures=1 pictures_cap=256 scopes=7 saved=90 taken=12 "
            "evicted=5 evicted_bytes=1234\n")

    def test_the_line_is_read_as_numbers(self):
        kept = parse_cache_line(self.LINE)
        self.assertEqual(kept["folder_bytes"], 9000)
        self.assertEqual((kept["ram"], kept["folder"], kept["scopes"], kept["evicted"], kept["evicted_bytes"]),
                         (3, 40, 7, 5, 1234))
        self.assertEqual(len(kept), 17)
        for other in ("STAT tok_s=1.0", "CACHED ram=1", "CACHE", "CACHE ram=x", "NOTE sessions: saved"):
            self.assertIsNone(parse_cache_line(other), other)
        self.assertEqual(parse_cache_line("CACHE ram=1 new_key=2 odd=-1 word"), {"ram": 1, "new_key": 2})

    def test_the_card_is_on_when_the_engine_keeps_conversations(self):
        off = conversation_cache_view({}, [], {}, {})
        self.assertEqual((off["enabled"], off["kept"]), (False, None))
        on = conversation_cache_view({}, [], {}, {}, parse_cache_line(self.LINE))
        self.assertTrue(on["enabled"])
        self.assertEqual(on["kept"]["folder_cap_bytes"], 10000)
        self.assertTrue(conversation_cache_view({"conversation_cache_mib": 64}, [], {}, {})["enabled"])

    def test_metrics_carry_the_engine_s_count(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "ok")
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertFalse(svc.metrics()["conversation_cache"]["enabled"])
        self.assertNotIn("strata:conversation_cache_bytes", svc.prometheus())
        engine.cache_stat = parse_cache_line(self.LINE)
        self.assertEqual(svc.metrics()["conversation_cache"]["kept"]["evicted"], 5)
        text = svc.prometheus()
        for row in ("strata:conversation_cache_entries 43", "strata:conversation_cache_bytes 9300",
                    "strata:conversation_cache_folder_cap_bytes 10000", "strata:conversation_cache_evictions_total 5",
                    "strata:conversation_cache_scopes 7"):
            self.assertIn(row + "\n", text)

    def test_the_number_of_kept_prompts_is_a_setting(self):
        self.assertEqual(prompt_keep_setting({"STRATA_PROMPT_KEEP": "256"}), 256)
        for bad in ({}, {"STRATA_PROMPT_KEEP": ""}, {"STRATA_PROMPT_KEEP": "0"}, {"STRATA_PROMPT_KEEP": "-4"},
                    {"STRATA_PROMPT_KEEP": "many"}):
            self.assertIsNone(prompt_keep_setting(bad), bad)
        tok = ByteTokenizer()
        os.environ["STRATA_PROMPT_KEEP"] = "37"
        self.addCleanup(os.environ.pop, "STRATA_PROMPT_KEEP", None)
        self.assertEqual(Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja")).prompts.keep, 37)


class Recording(MockEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.seen.append(dict(sampling or {}))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class Served(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = Recording(tok, "</think>\n\nok", max_context=1 << 20)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.set_family("deepseek41")
        self.addCleanup(frontend.use_family, None)
        self.messages = [{"role": "user", "content": "a long question " * 400}]

    def test_the_encoder_gets_the_request_s_scope(self):
        self.assertIsNotNone(self.svc.prompts)
        self.svc.prepare(list(self.messages), None, {}, 16, req={CACHE_SCOPE_KEY: "one"})
        turn = self.messages + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "more"}]
        self.svc.prepare(list(turn), None, {}, 16, req={CACHE_SCOPE_KEY: "two"})
        self.assertEqual(self.svc.prompts.last_reused, 0)
        self.svc.prepare(list(turn), None, {}, 16, req={})
        self.assertEqual(self.svc.prompts.last_reused, 0)
        self.svc.prepare(list(turn), None, {}, 16, req={CACHE_SCOPE_KEY: "one"})
        self.assertGreater(self.svc.prompts.last_reused, len(self.messages[0]["content"]) // 2)

    def test_a_bad_scope_is_refused_before_anything_is_kept(self):
        for bad in BAD:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, CACHE_SCOPE_KEY):
                self.svc.prepare(list(self.messages), None, {}, 16, req={CACHE_SCOPE_KEY: bad})
        self.assertEqual(self.svc.prompts.entries, [])

    def test_an_engine_that_keeps_no_scopes_refuses_one(self):
        self.svc.set_family(None)
        with self.assertRaisesRegex(ValueError, "only the DeepSeek engine"):
            self.svc.prepare(list(self.messages), None, {}, 16, req={CACHE_SCOPE_KEY: "one"})
        self.svc.prepare(list(self.messages), None, {}, 16, req={})

    def test_a_request_that_keeps_nothing_reuses_and_keeps_no_prompt(self):
        off = {CACHE_SCOPE_KEY: "one", CACHE_RETENTION_KEY: "none"}
        req = dict(off)
        self.svc.prepare(list(self.messages), [], {}, 16, req=req)
        self.assertEqual(self.svc.prompts.entries, [])
        self.assertNotIn("strata_prefix", req)
        self.assertNotIn("_restore", req)
        system = [{"role": "system", "content": "a shared system prompt " * 200}] + self.messages
        req = {**off, "strata_prefix": {"messages": 1}}
        self.svc.prepare(list(system), None, {}, 16, req=req)
        self.assertEqual((self.svc.prompts.entries, "strata_prefix" in req, "_restore" in req), ([], False, False))
        self.svc.prepare(list(self.messages), None, {}, 16, req={CACHE_SCOPE_KEY: "one"})   # kept as before
        self.assertEqual(len(self.svc.prompts.entries), 1)
        turn = self.messages + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "more"}]
        self.svc.prompts.last_reused = 0
        self.svc.prepare(list(turn), None, {}, 16, req=dict(off))
        self.assertEqual((self.svc.prompts.last_reused, len(self.svc.prompts.entries)), (0, 1))   # not read either

    def test_a_bad_retention_is_refused_before_anything_is_kept(self):
        for bad in BAD_RETENTION:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, CACHE_RETENTION_KEY):
                self.svc.prepare(list(self.messages), None, {}, 16, req={CACHE_RETENTION_KEY: bad})
        self.assertEqual(self.svc.prompts.entries, [])

    def test_an_engine_that_cannot_keep_nothing_refuses_it(self):
        self.svc.set_family(None)
        with self.assertRaisesRegex(ValueError, "only the DeepSeek engine"):
            self.svc.prepare(list(self.messages), None, {}, 16, req={CACHE_RETENTION_KEY: "none"})
        self.svc.prepare(list(self.messages), None, {}, 16, req={CACHE_RETENTION_KEY: "in_memory"})

    def test_a_request_that_keeps_nothing_reaches_the_engine_and_keeps_nothing_over_http(self):
        httpd = serve(self.svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        port = httpd.server_address[1]
        off = {CACHE_SCOPE_KEY: "one", CACHE_RETENTION_KEY: "none"}
        chat = {"messages": self.messages, "max_tokens": 16, **off}
        anthropic = {"model": "m", "messages": self.messages, "max_tokens": 16, **off}
        for path, body in (("/v1/chat/completions", chat), ("/v1/chat/completions/input_tokens", chat),
                           ("/v1/messages", anthropic), ("/v1/messages/count_tokens", anthropic)):
            status, answer = self.post(port, path, body)
            self.assertEqual(status, 200, (path, answer))
            for bad in BAD_RETENTION:
                self.assertEqual(self.post(port, path, {**body, CACHE_RETENTION_KEY: bad})[0], 400, (path, bad))
        self.assertEqual(self.svc.prompts.entries, [])
        self.assertEqual(len(self.engine.seen), 2)
        for sampling in self.engine.seen:
            self.assertIn(" keep=0", StrataEngine.sampling_keys(sampling))
        self.assertEqual(self.svc.metrics()["conversation_cache"]["prompts_kept"], 0)
        self.assertIn("strata:prompt_cache_prompts 0\n", self.svc.prometheus())
        self.assertEqual(self.post(port, "/v1/chat/completions", {**chat, CACHE_RETENTION_KEY: "24h"})[0], 200)
        self.assertNotIn(" keep=0", StrataEngine.sampling_keys(self.engine.seen[-1]))
        self.assertIn("strata:prompt_cache_prompts 1\n", self.svc.prometheus())

    def post(self, port, path, body):
        client = HTTPConnection("127.0.0.1", port, timeout=15)
        try:
            client.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            response = client.getresponse()
            return response.status, json.loads(response.read())
        finally:
            client.close()

    def test_a_request_s_scope_reaches_the_engine_s_line(self):
        httpd = serve(self.svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        port = httpd.server_address[1]
        body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}
        for path in ("/v1/chat/completions", "/v1/chat/completions/input_tokens"):
            for bad in BAD:
                status, answer = self.post(port, path, {**body, CACHE_SCOPE_KEY: bad})
                self.assertEqual(status, 400, (path, bad, answer))
        self.assertEqual(self.engine.seen, [])
        for scope in ("one", "two", None):
            status, answer = self.post(port, "/v1/chat/completions", {**body, **({CACHE_SCOPE_KEY: scope} if scope else {})})
            self.assertEqual(status, 200, answer)
            status, count = self.post(port, "/v1/chat/completions/input_tokens",
                                      {**body, **({CACHE_SCOPE_KEY: scope} if scope else {})})
            self.assertEqual((status, count["input_tokens"]), (200, answer["usage"]["prompt_tokens"]))
        lines = [StrataEngine.sampling_keys(s) for s in self.engine.seen]
        self.assertEqual(len(lines), 3)
        self.assertIn(f" scope={cache_scope({CACHE_SCOPE_KEY: 'one'})}", lines[0])
        self.assertIn(f" scope={cache_scope({CACHE_SCOPE_KEY: 'two'})}", lines[1])
        self.assertNotIn("scope=", lines[2])

    def test_an_anthropic_count_reads_only_its_own_scope_s_prompts(self):
        httpd = serve(self.svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        port = httpd.server_address[1]
        body = {"model": "m", "messages": self.messages, "max_tokens": 16}
        self.assertEqual(self.post(port, "/v1/messages", {**body, CACHE_SCOPE_KEY: "one"})[0], 200)
        turn = {**body, "messages": self.messages + [{"role": "assistant", "content": "a"},
                                                     {"role": "user", "content": "more"}]}
        self.assertEqual(self.post(port, "/v1/messages/count_tokens", {**turn, CACHE_SCOPE_KEY: "two"})[0], 200)
        self.assertEqual(self.svc.prompts.last_reused, 0)
        self.assertEqual(self.post(port, "/v1/messages/count_tokens", turn)[0], 200)
        self.assertEqual(self.svc.prompts.last_reused, 0)
        self.assertEqual(self.post(port, "/v1/messages/count_tokens", {**turn, CACHE_SCOPE_KEY: "one"})[0], 200)
        self.assertGreater(self.svc.prompts.last_reused, len(self.messages[0]["content"]) // 2)
        self.assertEqual(self.post(port, "/v1/messages/count_tokens", {**turn, CACHE_SCOPE_KEY: 5})[0], 400)


if __name__ == "__main__":
    unittest.main()
