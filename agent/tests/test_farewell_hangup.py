import asyncio
from unittest.mock import AsyncMock

import pytest

from farewell_hangup import FarewellHangupController


@pytest.mark.asyncio
async def test_farewell_followed_by_three_seconds_of_silence_hangs_up(monkeypatch):
    elapsed = []
    hang_up = AsyncMock()

    async def sleep(seconds):
        elapsed.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    controller = FarewellHangupController(hang_up)

    controller.on_assistant_final("Спасибо, до свидания.")
    controller.on_agent_state("listening")
    await controller.wait()

    assert elapsed == [3]
    hang_up.assert_awaited_once_with("farewell_silence")


@pytest.mark.asyncio
async def test_callee_question_cancels_pending_hangup_and_is_left_for_agent():
    hang_up = AsyncMock()
    controller = FarewellHangupController(hang_up)

    controller.on_assistant_final("Спасибо, до свидания.")
    controller.on_agent_state("listening")
    controller.on_user_state("speaking")
    action = controller.on_user_final("Подождите, а на сколько человек?")
    await asyncio.sleep(0)

    assert action == "continue"
    hang_up.assert_not_awaited()


@pytest.mark.asyncio
async def test_callee_goodbye_after_agent_farewell_hangs_up_immediately():
    hang_up = AsyncMock()
    controller = FarewellHangupController(hang_up)

    controller.on_assistant_final("Спасибо, до свидания.")
    controller.on_agent_state("listening")
    controller.on_user_state("speaking")
    action = controller.on_user_final("До свидания.")
    await controller.wait()

    assert action == "hangup"
    hang_up.assert_awaited_once_with("callee_goodbye")


@pytest.mark.asyncio
async def test_non_farewell_assistant_turn_never_arms_hangup():
    hang_up = AsyncMock()
    controller = FarewellHangupController(hang_up)

    controller.on_assistant_final("Можете, пожалуйста, уточнить время?")
    controller.on_agent_state("listening")
    await asyncio.sleep(0)

    hang_up.assert_not_awaited()
