"""Deterministic hangup after a spoken farewell, with caller-question override."""
from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable

_FAREWELL = re.compile(r"\b(?:до\s+свидания|всего\s+доброго|всего\s+хорошего)\b", re.I)
_QUESTION = re.compile(
    r"\?|\b(?:а|но|подождите|скажите|подскажите|уточните|сколько|когда|куда|кто|почему|зачем|какой|какая|какие)\b",
    re.I,
)


def is_farewell(text: str) -> bool:
    return bool(_FAREWELL.search(text.strip().replace("ё", "е")))


def is_question_or_clarification(text: str) -> bool:
    return bool(_QUESTION.search(text.strip().replace("ё", "е")))


class FarewellHangupController:
    def __init__(self, hang_up: Callable[[str], Awaitable[None]], delay: float = 3.0):
        self._hang_up = hang_up
        self._delay = delay
        self._farewell_spoken = False
        self._task: asyncio.Task[None] | None = None

    def on_assistant_final(self, text: str) -> None:
        if is_farewell(text):
            self._farewell_spoken = True

    def on_agent_state(self, state: str) -> None:
        if state == "listening" and self._farewell_spoken and self._task is None:
            self._task = asyncio.create_task(self._delayed_hangup())

    def on_user_state(self, state: str) -> None:
        if state == "speaking":
            self._cancel_timer()

    def on_user_final(self, text: str) -> str:
        if not self._farewell_spoken:
            return "continue"
        self._cancel_timer()
        if is_farewell(text):
            self._task = asyncio.create_task(self._run_hangup("callee_goodbye"))
            return "hangup"
        if is_question_or_clarification(text):
            self._farewell_spoken = False
            return "continue"
        self._farewell_spoken = False
        return "continue"

    async def _delayed_hangup(self) -> None:
        try:
            await asyncio.sleep(self._delay)
            await self._run_hangup("farewell_silence")
        except asyncio.CancelledError:
            return

    async def _run_hangup(self, reason: str) -> None:
        result = self._hang_up(reason)
        if inspect.isawaitable(result):
            await result

    def _cancel_timer(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def wait(self) -> None:
        if self._task is not None:
            await self._task

    async def close(self) -> None:
        self._cancel_timer()
        await asyncio.sleep(0)
