"""Shared native Anthropic and legacy OpenAI-compatible LLM routing."""
import os
from typing import Any

from livekit.agents.llm import ChatContext
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins import anthropic, openai

DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_LLM_ROUTE = "gateway"


class HaikuChatContext(ChatContext):
    def to_provider_format(self, format, **kwargs):
        # Plugin 1.6.7 predates Haiku 5.5's no-assistant-prefill policy.
        # Use LiveKit's existing provider serializer rather than mutating history.
        if format == "anthropic":
            kwargs["inject_trailing_user_message"] = True
        return super().to_provider_format(format, **kwargs)


class HaikuLLM(anthropic.LLM):
    def chat(self, *, chat_ctx: ChatContext, extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN, **kwargs):
        # LiveKit Anthropic 1.6.7 exposes provider kwargs on chat(), not __init__.
        extra = dict(extra_kwargs) if is_given(extra_kwargs) else {}
        extra.update(thinking={"type": "disabled"})
        return super().chat(chat_ctx=HaikuChatContext(items=list(chat_ctx.items)), extra_kwargs=extra, **kwargs)


def create_llm(
    call_config: dict[str, Any],
    *,
    gateway_base: str,
    gateway_key: str,
    default_base: str,
    default_key: str,
    reasoning_effort: str,
):
    """Build the selected provider without borrowing gateway credentials."""
    route = call_config.get("llm_route", DEFAULT_LLM_ROUTE)
    model = call_config.get("llm_model", DEFAULT_LLM_MODEL)
    if route == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the anthropic LLM route")
        return HaikuLLM(
            model=model,
            api_key=api_key,
            base_url="https://api.anthropic.com",
            tool_choice="auto",
        )
    # Preserve the existing gateway/default routing behavior.
    return openai.LLM(
        model=model,
        base_url=gateway_base if route == "gateway" else default_base,
        api_key=gateway_key if route == "gateway" else default_key,
        reasoning_effort=reasoning_effort,
    )
