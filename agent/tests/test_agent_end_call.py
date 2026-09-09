from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gen2b_agent


class FakeSpeech:
    async def wait_for_playout(self):
        return None


class FakeSession:
    def __init__(self):
        self.spoken = []
        self.shutdown_calls = []

    def say(self, text, **kwargs):
        self.spoken.append((text, kwargs))
        return FakeSpeech()

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


@pytest.mark.asyncio
async def test_end_call_plays_one_farewell_then_hangs_up(monkeypatch):
    hung_up = []
    stopped = []

    async def fake_hang_up(room_name, participant_identity):
        hung_up.append((room_name, participant_identity))

    async def fake_stop_recording():
        stopped.append(True)

    monkeypatch.setattr(gen2b_agent, "hang_up_sip_participant", fake_hang_up)

    agent = object.__new__(gen2b_agent.Gen2BAssistant)
    agent._room_name = "test-room"
    agent._participant_identity = "callee"
    agent._stop_recording = fake_stop_recording
    agent._pending_outcome = None
    agent._result_channel = "telegram"
    agent._demo_session_id = ""
    session = FakeSession()
    async def wait_for_playout():
        return None
    ctx = SimpleNamespace(session=session, wait_for_playout=wait_for_playout)

    result = await gen2b_agent.Gen2BAssistant.end_call._func(agent, ctx)

    assert session.spoken == [("Спасибо, до свидания.", {"allow_interruptions": False})]
    assert stopped == [True]
    assert hung_up == [("test-room", "callee")]
    assert session.shutdown_calls == [{"drain": False}]
    assert result == "Звонок завершён."


@pytest.mark.asyncio
async def test_end_call_is_idempotent_when_automatic_hangup_already_started(monkeypatch):
    hang_up = AsyncMock()
    monkeypatch.setattr(gen2b_agent, "hang_up_sip_participant", hang_up)
    agent = object.__new__(gen2b_agent.Gen2BAssistant)
    agent._hangup_started = True
    agent._latest_user_text = ""

    result = await gen2b_agent.Gen2BAssistant.end_call._func(
        agent, SimpleNamespace(session=FakeSession())
    )

    hang_up.assert_not_awaited()
    assert result == "Звонок уже завершается."
