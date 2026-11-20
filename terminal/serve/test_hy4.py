"""Hy4 parser boundaries, reasoning policy and engine launch isolation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serve.frontend import ChatTemplate, effort_kwargs, parser_for_template, use_family
from serve.hy4 import Hy4OutputParser, force_text
from serve.server import Hy4Engine, start_engine_or_fallback
from serve.server_parts.http_api import clean_shared_defaults

NS = ":6124c78e"
TOOLS = [
    {
        "name": "weather",
        "parameters": {
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}
        },
    }
]
CALL = f"<tool_calls{NS}><tool_call{NS}>weather<arg_key{NS}>city</arg_key{NS}><arg_value{NS}>Paris</arg_value{NS}><arg_key{NS}>days</arg_key{NS}><arg_value{NS}>2</arg_value{NS}></tool_call{NS}></tool_calls{NS}>"


class Hy4Parsing(unittest.TestCase):
    def tearDown(self):
        use_family(None)

    def test_every_stream_boundary(self):
        text = f"plan</think{NS}>" + CALL
        for at in range(len(text) + 1):
            p = Hy4OutputParser(thinking=True, tools=TOOLS, declared_only=True)
            events = p.feed(text[:at]) + p.feed(text[at:]) + p.finish()
            calls = [e.call for e in events if e.kind == "tool_call"]
            self.assertEqual(
                [(c.name, c.arguments) for c in calls],
                [("weather", {"city": "Paris", "days": 2})],
                at,
            )
            self.assertEqual(
                "".join(e.text for e in events if e.kind == "reasoning"), "plan", at
            )

    def test_single_character_stream_and_literal_call_closer(self):
        value = f"00123 </tool_call{NS}> café"
        text = CALL.replace("Paris", value)
        p = Hy4OutputParser(thinking=False, tools=TOOLS, declared_only=True)
        events = []
        for ch in text:
            events += p.feed(ch)
        events += p.finish()
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual(calls[0].arguments["city"], value)

    def test_incomplete_and_undeclared_are_not_calls(self):
        for text in (
            CALL[: CALL.index(f"</arg_value{NS}>")],
            CALL.replace("weather", "undeclared"),
            CALL.replace(f"</arg_key{NS}>", "", 1),
        ):
            p = Hy4OutputParser(thinking=False, tools=TOOLS, declared_only=True)
            events = p.feed(text) + p.finish()
            self.assertFalse(any(e.kind == "tool_call" for e in events))

    def test_dynamic_namespace_and_forced_prefixes(self):
        source = "{% set HYTK = ':test123' %} <｜hy_start{{HYTK}}｜> <arg_key{{HYTK}}>"
        parser = parser_for_template(source)(thinking=False, tools=TOOLS)
        events = parser.feed(CALL.replace(NS, ":test123")) + parser.finish()
        self.assertTrue(any(e.kind == "tool_call" for e in events))
        self.assertIsNone(force_text(None, NS))
        self.assertEqual(
            force_text("<tool_call>\n", NS), f"<tool_calls{NS}><tool_call{NS}>"
        )
        self.assertEqual(
            force_text("<tool_call>\n<function=weather>\n", NS),
            f"<tool_calls{NS}><tool_call{NS}>weather",
        )

    def test_effort_is_model_specific(self):
        use_family("hy4")
        self.assertEqual(effort_kwargs("max"), {"reasoning_effort": "max"})
        self.assertEqual(effort_kwargs("high"), {"reasoning_effort": "high"})
        self.assertEqual(
            effort_kwargs(False),
            {"enable_thinking": False, "reasoning_effort": "no_think"},
        )
        self.assertEqual(
            clean_shared_defaults({"reasoning_effort": "max"}),
            {"reasoning_effort": "max"},
        )
        with self.assertRaises(ValueError):
            effort_kwargs("medium")
        use_family(None)
        self.assertEqual(effort_kwargs("high"), {"reasoning_effort": "xhigh"})
        with self.assertRaises(ValueError):
            clean_shared_defaults({"reasoning_effort": "max"})

    def test_top_k_not_capped_at_qwen_limit(self):
        self.assertIn(" top_k=0", Hy4Engine.sampling_keys({"top_k": 0}))
        self.assertIn(" top_k=1000", Hy4Engine.sampling_keys({"top_k": 1000}))
        with self.assertRaises(ValueError):
            Hy4Engine.sampling_keys({"logprobs": True})

    def test_preserved_history_uses_hy4_template_keyword(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.jinja"
            path.write_text(
                "{% set HYTK = ':test123' %}<｜hy_start{{HYTK}}｜><arg_key{{HYTK}}>"
                "{% for m in messages %}{% if preserved_thinking %}{{m.reasoning_content}}{% endif %}{% endfor %}"
            )
            template = ChatTemplate(path)
            messages = [
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "history",
                }
            ]
            self.assertIn("history", template.render(messages, preserve_thinking=True))
            self.assertNotIn(
                "history", template.render(messages, preserve_thinking=False)
            )

    def test_does_not_apply_qwen_flags_or_fallback(self):
        cfg = {
            "arch": "hy_v4_flash",
            "args": ["--model", "hy4.gguf"],
            "drafting": "off",
        }
        with (
            patch("serve.server.Hy4Engine") as engine,
            patch("serve.server.engine_args", side_effect=AssertionError("Qwen flags")),
        ):
            start_engine_or_fallback(cfg, "strata-hy4", None, {}, False, None)
            self.assertEqual(
                engine.call_args.args, ("strata-hy4", ["--model", "hy4.gguf"])
            )
        for extra in (
            {"drafting": "mtp"},
            {"parallel": 4},
            {"vision": {"exe": "vision"}},
        ):
            with self.assertRaises(ValueError):
                start_engine_or_fallback(
                    cfg | extra, "strata-hy4", None, {}, False, None
                )


if __name__ == "__main__":
    unittest.main()
