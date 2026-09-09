import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import gen2b_agent as g
import live_callback


@pytest.mark.parametrize(("question", "context", "expected"), [
    (
        "Денис говорит, что русского бильярда нет. Есть только снукер и пул. Армен, какой вам выбрать?",
        "",
        ["Снукер", "Пул"],
    ),
    (
        "Что выбрать: чай или кофе?",
        "В кафе доступны чай и кофе.",
        ["Чай", "Кофе"],
    ),
    (
        "Армен, какой бильярдный стол бронировать: русский или пул?",
        "",
        ["Русский", "Пул"],
    ),
    (
        "Какой бильярд забронировать: русский, пул или снукер?",
        "Бильярдная уточняет тип стола.",
        ["Русский", "Пул", "Снукер"],
    ),
])
def test_infer_owner_options_from_explicit_alternatives(question, context, expected):
    assert g.infer_owner_options(question, context) == expected


@pytest.mark.asyncio
async def test_web_owner_question_emits_event_without_telegram(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "LIVE_CALLBACK_DIR", tmp_path / "callbacks")
    monkeypatch.setattr(g, "LIVE_CALLBACK_TIMEOUT", 0.2)
    emitted = []
    monkeypatch.setattr(g, "emit_demo_event", lambda sid, room, kind, payload: emitted.append((sid, room, kind, payload)))
    telegram = Mock(side_effect=AssertionError("Telegram must not be used"))
    monkeypatch.setattr(g, "send_live_callback_to_telegram", telegram)

    async def resolve_when_created():
        for _ in range(20):
            pending = list((tmp_path / "callbacks" / "pending").glob("*.json"))
            if pending:
                rid = pending[0].stem
                live_callback.resolve_request(tmp_path / "callbacks", rid, "yes")
                return
            await asyncio.sleep(0.01)
        raise AssertionError("owner request was not created")

    session = SimpleNamespace(say=Mock(), generate_reply=Mock(), interrupt=AsyncMock())
    agent = object.__new__(g.Gen2BAssistant)
    agent._room_name = "room-web"
    agent._owner_channel = "web"
    agent._demo_session_id = "demo_web"
    resolver = asyncio.create_task(resolve_when_created())

    result = json.loads(await g.Gen2BAssistant.ask_owner._func(
        agent,
        SimpleNamespace(session=session),
        "Встречу перенести на 15:00?",
        "В 14:00 занято.",
        ["Перенести", "Оставить как есть"],
    ))
    await resolver
    await asyncio.sleep(0)

    assert result["action"] == "approve"
    telegram.assert_not_called()
    assert [event[2] for event in emitted] == ["owner.question", "owner.answer"]
    assert emitted[0][3]["question"] == "Встречу перенести на 15:00?"
    assert emitted[0][3]["options"] == ["Перенести", "Оставить как есть"]


def test_demo_transcript_emitter_keeps_partial_and_final_text(monkeypatch):
    emitted = []
    monkeypatch.setattr(g, "emit_demo_event", lambda sid, room, kind, payload: emitted.append((kind, payload)))
    emitter = g.DemoTranscriptEmitter("demo_web", "room-web")

    emitter.on_user_transcript(SimpleNamespace(is_final=False, transcript="Добрый", item_id="u1"))
    emitter.on_user_transcript(SimpleNamespace(is_final=True, transcript="Добрый день", item_id="u1"))
    emitter.on_user_transcript(SimpleNamespace(is_final=True, transcript=" ", item_id="u2"))

    assert emitted == [
        ("transcript.caller.partial", {"utterance_id": "u1", "text": "Добрый"}),
        ("transcript.caller.final", {"utterance_id": "u1", "text": "Добрый день"}),
    ]
