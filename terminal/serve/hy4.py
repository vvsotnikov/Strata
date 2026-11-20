"""Hy4's namespaced reasoning and tool markers, adapted to Strata's event parser."""

from __future__ import annotations

import json
import re

from serve.frontend import Event, JsonOutputParser


def namespace_from_template(source: str) -> str:
    match = re.search(r"\bset\s+HYTK\s*=\s*['\"]([^'\"]+)['\"]", source)
    if not match or not match[1] or len(match[1]) > 64:
        raise ValueError("Hy4 template does not declare its HYTK namespace")
    return match[1]


def parse_call(body: str, namespace: str, schemas: dict | None = None) -> dict:
    ko, kc = f"<arg_key{namespace}>", f"</arg_key{namespace}>"
    vo, vc = f"<arg_value{namespace}>", f"</arg_value{namespace}>"
    first = body.find(ko)
    name = (body if first < 0 else body[:first]).strip()
    if not name or len(name) > 512 or "<" in name or ">" in name:
        raise ValueError("invalid Hy4 tool name")
    properties = ((schemas or {}).get(name, {}).get("parameters") or {}).get(
        "properties"
    ) or {}
    args = {}
    rest = "" if first < 0 else body[first:]
    while rest.strip():
        rest = rest.lstrip()
        if not rest.startswith(ko) or (end := rest.find(kc, len(ko))) < 0:
            raise ValueError("incomplete Hy4 argument key")
        key = rest[len(ko) : end]
        if not key or key in args or "<" in key or ">" in key:
            raise ValueError("empty or duplicate Hy4 argument key")
        rest = rest[end + len(kc) :].lstrip()
        if not rest.startswith(vo) or (end := rest.find(vc, len(vo))) < 0:
            raise ValueError("incomplete Hy4 argument value")
        value = rest[len(vo) : end]
        if (properties.get(key) or {}).get("type") != "string":
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                pass
        args[key] = value
        rest = rest[end + len(vc) :]
    return {"name": name, "arguments": args}


class Hy4OutputParser(JsonOutputParser):
    namespace = ":6124c78e"

    @property
    def clean(self):
        return not getattr(self, "hy_buffer", "") and super().clean

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ns = self.namespace
        self.hy_buffer = ""
        self.hy_call = f"<tool_call{ns}>"
        self.hy_end = f"</tool_call{ns}>"
        self.hy_value = f"<arg_value{ns}>"
        self.hy_value_end = f"</arg_value{ns}>"
        self.hy_markers = {
            f"<think{ns}>": "<think>",
            f"</think{ns}>": "</think>",
            f"<tool_calls{ns}>": "",
            f"</tool_calls{ns}>": "",
        }

    def _call_end(self):
        pos = len(self.hy_call)
        while True:
            end = self.hy_buffer.find(self.hy_end, pos)
            value = self.hy_buffer.find(self.hy_value, pos)
            if value < 0 or (0 <= end < value):
                return end
            close = self.hy_buffer.find(self.hy_value_end, value + len(self.hy_value))
            if close < 0:
                return -1
            pos = close + len(self.hy_value_end)

    def feed(self, delta: str):
        self.hy_buffer += delta
        if len(self.hy_buffer) > 8 * 1024 * 1024:
            raise ValueError("Hy4 tool call exceeds the 8 MiB parser limit")
        events = []
        markers = [self.hy_call, *self.hy_markers]
        while self.hy_buffer:
            if self.hy_buffer.startswith(self.hy_call):
                end = self._call_end()
                if end < 0:
                    break
                raw = self.hy_buffer[: end + len(self.hy_end)]
                self.hy_buffer = self.hy_buffer[len(raw) :]
                try:
                    call = parse_call(
                        raw[len(self.hy_call) : -len(self.hy_end)],
                        self.namespace,
                        self.schemas,
                    )
                except ValueError:
                    events.append(
                        Event(
                            "reasoning" if self.state == "reasoning" else "content", raw
                        )
                    )
                else:
                    events.extend(
                        super().feed(
                            "<tool_call>"
                            + json.dumps(call, ensure_ascii=False)
                            + "</tool_call>"
                        )
                    )
                continue
            marker = next(
                (m for m in self.hy_markers if self.hy_buffer.startswith(m)), None
            )
            if marker:
                self.hy_buffer = self.hy_buffer[len(marker) :]
                events.extend(super().feed(self.hy_markers[marker]))
                continue
            found = [self.hy_buffer.find(m) for m in markers if m in self.hy_buffer]
            if found:
                take = min(found)
            else:
                keep = max(
                    (
                        i
                        for m in markers
                        for i in range(1, min(len(m), len(self.hy_buffer)) + 1)
                        if self.hy_buffer.endswith(m[:i])
                    ),
                    default=0,
                )
                take = len(self.hy_buffer) - keep
            if not take:
                break
            events.extend(super().feed(self.hy_buffer[:take]))
            self.hy_buffer = self.hy_buffer[take:]
        return events

    def finish(self, reason=None):
        rest, self.hy_buffer = self.hy_buffer, ""
        kind = "reasoning" if self.state == "reasoning" else "content"
        return super().finish(reason) + ([Event(kind, rest)] if rest else [])


def parser_for_source(source: str):
    return type(
        "ModelHy4OutputParser",
        (Hy4OutputParser,),
        {"namespace": namespace_from_template(source)},
    )


def force_text(force: str | None, namespace: str) -> str | None:
    if force is None:
        return None
    if force.strip() == "<tool_call>":
        return f"<tool_calls{namespace}><tool_call{namespace}>"
    match = re.fullmatch(r"<tool_call>\s*<function=([^>]+)>\s*", force)
    if not match:
        raise ValueError("unsupported forced-call prefix for Hy4")
    return f"<tool_calls{namespace}><tool_call{namespace}>{match[1]}"


def template_calls(prompt: str, namespace: str, tools: list[dict]) -> list[dict]:
    pattern = (
        re.escape(f"<tool_call{namespace}>")
        + r"(strata_caps_call_\d+.*?)"
        + re.escape(f"</tool_call{namespace}>")
    )
    schemas = {t["name"]: t for t in tools}
    return [
        parse_call(m, namespace, schemas)
        for m in re.findall(pattern, prompt, re.DOTALL)
    ]
