from live_callback import enqueue_instruction, pop_instruction
from gen2b_agent import apply_live_instruction


class FakeSession:
    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []

    async def interrupt(self, *, force):
        self.calls.append(("interrupt", force))

    def generate_reply(self, **kwargs):
        self.calls.append(("generate_reply", kwargs))


def test_live_instruction_queue_is_room_scoped_and_consumed_once(tmp_path):
    first = enqueue_instruction(
        tmp_path,
        session_id="demo-one",
        room_name="room-one",
        text="Узнай про окно",
    )
    enqueue_instruction(
        tmp_path,
        session_id="demo-two",
        room_name="room-two",
        text="Узнай про парковку",
    )

    item = pop_instruction(tmp_path, session_id="demo-one", room_name="room-one")

    assert item == {"instruction_id": first, "text": "Узнай про окно"}
    assert pop_instruction(tmp_path, session_id="demo-one", room_name="room-one") is None
    assert pop_instruction(tmp_path, session_id="demo-two", room_name="wrong-room") is None


def test_live_instruction_interrupts_and_continues_the_same_session():
    session = FakeSession()

    import asyncio
    asyncio.run(apply_live_instruction(session, "Узнай, стол у окна или нет?"))

    assert session.calls[0] == ("interrupt", True)
    kind, kwargs = session.calls[1]
    assert kind == "generate_reply"
    assert "Узнай, стол у окна или нет?" in kwargs["user_input"]
    assert "текущего звонка" in kwargs["user_input"]
    assert "задай собеседнику сам вопрос сразу" in kwargs["instructions"].lower()
    assert "не говори, что ты уточнил" in kwargs["instructions"].lower()
    assert "не проси подождать" in kwargs["instructions"].lower()
    assert kwargs["allow_interruptions"] is False


def test_end_call_waits_for_farewell_playout_before_hangup(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    import gen2b_agent as g

    calls = []

    async def wait_for_playout():
        calls.append("prior_playout")

    async def wait_for_farewell():
        calls.append("farewell_playout")

    def say(text, **kwargs):
        calls.append(("say", text, kwargs))
        return SimpleNamespace(wait_for_playout=wait_for_farewell)

    async def stop_recording():
        calls.append("stop_recording")

    agent = g.Gen2BAssistant(
        room_name="room-one", participant_identity="callee",
        on_outcome_saved=lambda: None, system_prompt="test",
        recording_path_getter=lambda: None, stop_recording=stop_recording,
        booking_confirmation_required=False, result_channel="web",
        demo_session_id="demo-one",
    )

    async def pause(seconds):
        calls.append(f"pause:{seconds}")

    async def hangup(*args):
        calls.append("hangup")

    agent._pending_outcome = {"outcome": "agreed", "summary": "Бронь подтверждена"}
    monkeypatch.setattr(g, "emit_demo_event", lambda *args: calls.append(args[2]))
    monkeypatch.setattr(g.asyncio, "sleep", pause)
    monkeypatch.setattr(g, "hang_up_sip_participant", hangup)

    import asyncio
    asyncio.run(g.Gen2BAssistant.end_call._func(
        agent,
        SimpleNamespace(wait_for_playout=wait_for_playout, session=SimpleNamespace(say=say, shutdown=Mock())),
    ))

    assert calls == [
        "prior_playout",
        ("say", "Спасибо, до свидания.", {"allow_interruptions": False}),
        "farewell_playout",
        "pause:3",
        "call.outcome",
        "stop_recording",
        "hangup",
        "call.ended",
    ]
