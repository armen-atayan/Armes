import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import live_callback
import gen2b_agent


def test_create_request_persists_owner_question_and_options(tmp_path):
    request = live_callback.create_request(
        tmp_path,
        room_name="call-1",
        chat_id="135001671",
        question="В 20:00 недоступно, есть в 21:00. Подтверждаем?",
    )

    payload = json.loads((tmp_path / "pending" / f"{request}.json").read_text())
    assert payload["room_name"] == "call-1"
    assert payload["chat_id"] == "135001671"
    assert payload["question"].endswith("Подтверждаем?")
    assert payload["status"] == "pending"


def test_response_from_another_chat_is_rejected(tmp_path):
    request = live_callback.create_request(
        tmp_path, room_name="call-1", chat_id="135001671", question="Подтверждаем?"
    )

    assert not live_callback.resolve_request(tmp_path, request, "yes", chat_id="999")
    assert not (tmp_path / "responses" / f"{request}.txt").exists()


@pytest.mark.asyncio
async def test_wait_for_response_returns_button_choice(tmp_path):
    request = live_callback.create_request(
        tmp_path, room_name="call-1", chat_id="135001671", question="Подтверждаем?"
    )
    live_callback.resolve_request(tmp_path, request, "yes")

    answer = await live_callback.wait_for_response(tmp_path, request, timeout=0.2, poll_interval=0.01)

    assert answer == "yes"
    assert not (tmp_path / "pending" / f"{request}.json").exists()


@pytest.mark.asyncio
async def test_wait_for_response_times_out(tmp_path):
    request = live_callback.create_request(
        tmp_path, room_name="call-1", chat_id="135001671", question="Подтверждаем?"
    )

    answer = await live_callback.wait_for_response(tmp_path, request, timeout=0.03, poll_interval=0.01)

    assert answer == "timeout"


def test_other_mode_captures_next_matching_telegram_message(tmp_path):
    request = live_callback.create_request(
        tmp_path, room_name="call-1", chat_id="135001671", question="Подтверждаем?"
    )
    assert live_callback.mark_awaiting_text(tmp_path, request, "135001671")

    captured = live_callback.capture_text_response(
        tmp_path, chat_id="135001671", text="Проверь завтра в 20:00"
    )

    assert captured == request
    assert (tmp_path / "responses" / f"{request}.txt").read_text() == "Проверь завтра в 20:00"


@pytest.mark.asyncio
async def test_agent_asks_owner_and_returns_instruction(monkeypatch, tmp_path):
    sent = []

    def fake_send(request_id, question):
        sent.append((request_id, question))
        live_callback.resolve_request(tmp_path, request_id, "yes")

    monkeypatch.setattr(gen2b_agent, "LIVE_CALLBACK_DIR", tmp_path)
    monkeypatch.setattr(gen2b_agent, "send_live_callback_to_telegram", fake_send)

    agent = object.__new__(gen2b_agent.Gen2BAssistant)
    agent._room_name = "call-1"
    spoken = []

    class Speech:
        async def wait_for_playout(self):
            return None

    ctx = SimpleNamespace(
        session=SimpleNamespace(
            say=lambda text, **kwargs: (spoken.append(text), Speech())[1]
        )
    )
    result = await gen2b_agent.Gen2BAssistant.ask_owner._func(
        agent,
        ctx,
        "В 20:00 недоступно, есть в 21:00. Подтверждаем?",
    )

    assert spoken == ["Секундочку, сейчас уточню."]
    assert sent and sent[0][1].endswith("Подтверждаем?")
    result = json.loads(result)
    assert result['action'] == 'approve'
    assert result['response'] == 'yes'
    assert result['question'] == sent[0][1]
