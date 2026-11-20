"""Hy4 token counts use the same image and reasoning preparation as generation."""

import copy
import json
import tempfile
import unittest
import urllib.request
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from serve.frontend import ChatTemplate, anthropic_to_messages, use_family
from serve.server import ByteTokenizer, MockEngine, Service, serve

NS = ":6124c78e"
START = f"<｜hy_image▁start{NS}｜>"
PAD = f"<｜hy_image▁pad{NS}｜>"
END = f"<｜hy_image▁end{NS}｜>"


class Tokenizer(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + [
        START,
        PAD,
        END,
        f"<think{NS}>",
        f"</think{NS}>",
    ]
    special_ids: ClassVar[dict[str, int]] = {"eos": 257}


class Counts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        path = self.root / "chat.jinja"
        path.write_text(
            "{% set HYTK = ':6124c78e' %}{% for m in messages %}{{m.role}}:{% if m.content is string %}{{m.content}}{% else %}{% for p in m.content %}{% if p.type == 'image' %}"
            + START
            + PAD
            + END
            + "{% else %}{{p.text}}{% endif %}{% endfor %}{% endif %}{% endfor %}assistant:{% if reasoning_effort == 'no_think' %}<think:6124c78e></think:6124c78e>{% endif %}"
        )

        class Vision:
            def __init__(self, folder):
                self.dir = folder
                self.path = folder / "image.sve"
                self.path.write_bytes(b"encoded")

            def encode(self, source):
                return self.path, 6

        self.tok = Tokenizer()
        self.engine = MockEngine(self.tok, "A", max_context=4096)
        self.svc = Service(
            self.engine, self.tok, ChatTemplate(path), vision=Vision(self.root)
        )
        self.svc.set_family("hy4")
        self.addCleanup(use_family, None)

    def test_anthropic_disabled_and_unasked_map_to_hy4_no_think(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        for request, unasked in (
            (body | {"thinking": {"type": "disabled"}}, True),
            (body, False),
        ):
            self.assertEqual(
                anthropic_to_messages(request, unasked)[2],
                {"enable_thinking": False, "reasoning_effort": "no_think"},
            )
        use_family(None)
        self.assertEqual(
            anthropic_to_messages(body | {"thinking": {"type": "disabled"}})[2],
            {"enable_thinking": False},
        )

    def test_count_expands_real_namespace_images_without_request_file_or_prompt_store(
        self,
    ):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "source": "data:image/png;base64,AAAA"}],
            }
        ]
        original = copy.deepcopy(messages)
        self.svc.embeddings.path, self.svc.embeddings.keys = "keep", ["keep"]
        entries = copy.deepcopy(self.svc.prompts.entries)
        with (
            patch.object(
                self.svc, "encode_prompt", wraps=self.svc.encode_prompt
            ) as encode,
            patch(
                "serve.server.combined_embeddings_path",
                side_effect=AssertionError("count wrote request file"),
            ),
        ):
            ids, thinking, maximum = self.svc.prepare(
                messages,
                None,
                {"enable_thinking": False, "reasoning_effort": "no_think"},
                count_only=True,
            )
        self.assertEqual(ids.count(self.tok.encode(PAD, True)[0]), 6)
        self.assertFalse(thinking)
        self.assertIsNone(maximum)
        self.assertFalse(encode.call_args.kwargs["store"])
        self.assertEqual(self.svc.prompts.entries, entries)
        self.assertEqual(
            (self.svc.embeddings.path, self.svc.embeddings.keys), ("keep", ["keep"])
        )
        self.assertEqual(messages, original)
        self.assertEqual(self.engine.turns, 0)

    def test_text_count_is_read_only_and_can_exceed_model_context(self):
        messages = [{"role": "user", "content": "a long text"}]
        self.engine.max_context = 1
        with patch.object(
            self.svc, "encode_prompt", wraps=self.svc.encode_prompt
        ) as encode:
            ids, _, maximum = self.svc.prepare(
                messages, None, {}, max_new=999, count_only=True
            )
        self.assertGreater(len(ids), self.engine.max_context)
        self.assertFalse(encode.call_args.kwargs["store"])
        self.assertIsNone(maximum)
        self.assertEqual(self.engine.turns, 0)

    def test_count_does_not_rewrite_missing_tool_image_references(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": "strata-tool-media:" + "a" * 64 + ".png",
                    }
                ],
            }
        ]
        original = copy.deepcopy(messages)
        ids, _, _ = self.svc.prepare(messages, None, {}, count_only=True)
        self.assertIn("image omitted", self.tok.decode(ids))
        self.assertEqual(messages, original)

    def test_http_count_matches_same_anthropic_image_generation(self):
        httpd = serve(self.svc, port=0)
        try:
            body = {
                "model": "m",
                "max_tokens": 1,
                "thinking": {"type": "disabled"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "colors?"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "AAAA",
                                },
                            },
                        ],
                    }
                ],
            }

            def post(path):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{httpd.server_address[1]}" + path,
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    return json.load(response)

            counted = post("/v1/messages/count_tokens")["input_tokens"]
            self.assertEqual(self.engine.turns, 0)
            generated = post("/v1/messages")["usage"]
            actual = sum(
                generated.get(key, 0)
                for key in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )
            self.assertEqual(counted, actual)
            self.assertEqual(
                self.engine.last_prompt.count(self.tok.encode(PAD, True)[0]), 6
            )
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
