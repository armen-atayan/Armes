from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

from models import CallDispatchResult, CallRequest


class CallWithPersonaDispatcher:
    """Small adapter around the agent's reusable call dispatch entry point."""

    def __init__(self, *, agent_dir: Path | None = None, extra_kwargs: dict[str, Any] | None = None):
        agent_dir = agent_dir or Path("/home/ubuntu/livekit-stack/agent")
        if str(agent_dir) not in sys.path:
            sys.path.insert(0, str(agent_dir))
        try:
            module = importlib.import_module("call_with_persona")
        except ImportError as exc:
            raise RuntimeError("call_with_persona is unavailable") from exc
        function = getattr(module, "dispatch_call", None) or getattr(module, "call_with_persona", None)
        if function is None or not inspect.iscoroutinefunction(function):
            raise RuntimeError("call_with_persona has no reusable async dispatch function")
        self._function = function
        self._extra_kwargs = extra_kwargs or {}

    async def dispatch(self, request: CallRequest, *, session_id: str, room_name: str) -> CallDispatchResult:
        kwargs = {
            "persona": "armen_personal_assistant",
            "phone_number": request.phone_number,
            "phone": request.phone_number,
            "target_name": request.contact_name,
            "name": request.contact_name,
            "target_identity": "callee",
            "task": request.task,
            "task_details": request.details,
            "details": request.details,
            "room_name": room_name,
            "room": room_name,
            "origin": "web_demo",
            "owner_channel": "web",
            "result_channel": "web",
            "demo_session_id": session_id,
            **self._extra_kwargs,
        }
        signature = inspect.signature(self._function)
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
        selected = kwargs if accepts_kwargs else {key: value for key, value in kwargs.items() if key in signature.parameters}
        raw = await self._function(**selected)
        if isinstance(raw, CallDispatchResult):
            return raw
        if isinstance(raw, dict):
            return CallDispatchResult(
                room_name=str(raw.get("room_name") or room_name),
                call_id=raw.get("sip_call_id") or raw.get("call_id"),
            )
        return CallDispatchResult(room_name=room_name)


class UnavailableDispatcher:
    def __init__(self, error: Exception):
        self.error = error

    async def dispatch(self, request: CallRequest, *, session_id: str, room_name: str) -> CallDispatchResult:
        raise RuntimeError(str(self.error))


def default_dispatcher():
    try:
        return CallWithPersonaDispatcher()
    except RuntimeError as exc:
        return UnavailableDispatcher(exc)
