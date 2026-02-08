"""Mock LLM provider for manual or random responses."""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from nanobot.providers.base import LLMProvider, LLMResponse


_DEFAULT_RESPONSES = [
    "Sure thing. What should I do next?",
    "Got it. Tell me more.",
    "I can help with that.",
    "Okay. What are the constraints?",
    "Acknowledged. What's the desired outcome?",
    "Understood. Any examples to follow?",
    "Makes sense. What should I prioritize?",
    "Thanks. What is the timeline?",
]


class MockProvider(LLMProvider):
    """Mock provider with manual and RNG modes."""

    def __init__(
        self,
        mode: str = "manual",
        responses: list[str] | None = None,
    ):
        super().__init__(api_key=None, api_base=None)
        self.mode = mode
        self.responses = responses or list(_DEFAULT_RESPONSES)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        if self.mode == "manual":
            payload = json.dumps(messages, indent=2, ensure_ascii=True, default=str)
            print("\n[MockProvider manual] Messages JSON:\n" + payload)
            content = await asyncio.to_thread(_read_multiline_response)
            if not content:
                return LLMResponse(
                    content="Mock provider: no input received.",
                    finish_reason="error",
                )
            return LLMResponse(content=content)

        if self.mode == "rng":
            return LLMResponse(content=random.choice(self.responses))

        return LLMResponse(
            content=f"Mock provider: unsupported mode '{self.mode}'.",
            finish_reason="error",
        )

    def get_default_model(self) -> str:
        return "mock/manual"


def _read_multiline_response() -> str:
    """Read multi-line input until a blank line or EOF."""
    print("\nType response (end with empty line):")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()
