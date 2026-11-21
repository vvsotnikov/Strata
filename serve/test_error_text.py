"""serve/test_error_text.py - an error never quotes the request's or the model's text.

A front end logs the errors it gets and keeps them with a failed request's records, so an error that quoted a value
would store prompt or output text there.  A marker is put in the prompt, the schema's values and the model's answer;
every error path below must keep it out of the error body and out of the server's own output.

    python -m unittest serve.test_error_text -v
"""
import contextlib
import io
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import structured
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

try:
    import jsonschema  # noqa: F401
    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False

MARK = "zq7canary"


def schema(**extra):
    return {"type": "object", "properties": {"source": {"type": "string", "enum": ["A", "B"]},
                                             "items": {"type": "array", "items": {"type": "integer"}}},
            "required": ["source"], "additionalProperties": False, **extra}


def fmt(s):
    return {"type": "json_schema", "json_schema": {"name": "t", "strict": True, "schema": s}}


class ErrorText(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.svc = Service(MockEngine(self.tok, "{}", max_context=16384), self.tok,
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.out = io.StringIO()
        redirect = contextlib.redirect_stdout(self.out)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self, answer, response_format, messages=None):
        self.svc.engine = MockEngine(self.tok, answer, max_context=16384)
        body = {"messages": messages or [{"role": "user", "content": f"say {MARK} please"}],
                "response_format": response_format, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.read().decode()

    def assert_quotes_nothing(self, status, body, want_status):
        self.assertEqual(status, want_status, body)
        self.assertNotIn(MARK, body)
        self.assertNotIn(MARK, self.out.getvalue())

    def test_an_answer_that_is_not_json_quotes_none_of_it(self):
        for answer in (f'{{"source": "{MARK}" "x"}}', f'{{"{MARK}": 1, "{MARK}": 2}}', f'{{"{MARK}": NaN}}',
                       f'{{"{MARK}": 1e999}}', f"{MARK} is not JSON", f'{{"source": "{MARK}'):
            with self.subTest(answer=answer):
                self.assert_quotes_nothing(*self.chat(answer, {"type": "json_object"}), 502)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_an_answer_that_breaks_the_schema_names_the_place_and_the_rule_only(self):
        cases = ((json.dumps({"source": MARK}), "/source", "enum"),
                 (json.dumps({"source": "A", MARK: 1}), "/", "additionalProperties"),
                 (json.dumps({"source": "A", "items": [1, MARK]}), "/items/1", "type"),
                 (json.dumps({"source": 5}), "/source", "type"),
                 (json.dumps({"items": [1]}), "/", "required"))
        for answer, where, rule in cases:
            with self.subTest(answer=answer):
                status, body = self.chat(answer, fmt(schema()))
                self.assert_quotes_nothing(status, body, 502)
                message = json.loads(body)["error"]["message"]
                self.assertEqual(message, f"model output failed the JSON Schema at {where}: it breaks the '{rule}' rule")

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_a_key_the_model_made_up_is_not_named(self):
        s = {"type": "object", "additionalProperties": {"type": "integer"}}
        status, body = self.chat(json.dumps({MARK: "x"}), fmt(s))
        self.assert_quotes_nothing(status, body, 502)
        self.assertIn("at /*:", json.loads(body)["error"]["message"])

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_an_invalid_schema_quotes_none_of_its_values(self):
        for bad in ({"type": "object", "properties": {"x": {"type": MARK}}},
                    {"type": "object", "properties": {"x": {"enum": MARK}}},
                    {"type": "object", "required": MARK}):
            with self.subTest(schema=bad):
                status, body = self.chat("{}", fmt(bad))
                self.assert_quotes_nothing(status, body, 400)
                self.assertIn("invalid response_format schema at /", json.loads(body)["error"]["message"])

    def test_the_rule_and_place_helpers_never_pass_text_through(self):
        class Odd:
            validator = f"{MARK} rule"
        self.assertEqual(structured._rule(Odd()), "unknown")
        self.assertEqual(structured._pointer(["a/b", 3, MARK, True], {"a/b"}), "/a~1b/3/*/*")


if __name__ == "__main__":
    unittest.main()
