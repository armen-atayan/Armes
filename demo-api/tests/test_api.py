import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import create_app
from models import CallDispatchResult
from settings import Settings


class FakeDispatcher:
    def __init__(self):
        self.requests = []

    async def dispatch(self, request, *, session_id, room_name):
        self.requests.append((request, session_id, room_name))
        return CallDispatchResult(room_name=room_name, call_id="fake-call")


class FakeTranscriber:
    async def transcribe(self, audio: bytes, content_type: str) -> str:
        assert audio == b"voice-bytes"
        assert content_type == "audio/webm"
        return "Позвони Артуру и узнай про машину"


@pytest.fixture
def harness(tmp_path):
    dispatcher = FakeDispatcher()
    settings = Settings(
        data_dir=tmp_path / "data",
        callback_dir=tmp_path / "callbacks",
        recordings_dir=tmp_path / "recordings",
        frontend_dist=tmp_path / "missing-dist",
        demo_token=None,
        websocket_poll_interval=0.01,
    )
    app = create_app(settings=settings, dispatcher=dispatcher, transcriber=FakeTranscriber())
    with TestClient(app) as client:
        yield client, dispatcher, settings, app


def create_call(client):
    response = client.post(
        "/api/calls",
        json={
            "contact_name": "Ресторан",
            "phone_number": "+77001234567",
            "task": "Забронировать стол",
            "details": "Сегодня в 20:00",
        },
    )
    assert response.status_code == 201
    return response.json()


def test_health_reports_ready(harness):
    client, _, _, _ = harness
    assert client.get("/health").json() == {"status": "ok"}


def test_transcribe_voice_message(harness):
    client, _, _, _ = harness
    response = client.post("/api/transcribe", content=b"voice-bytes", headers={"content-type": "audio/webm"})
    assert response.status_code == 200
    assert response.json() == {"text": "Позвони Артуру и узнай про машину"}


def test_create_call_dispatches_web_personal_assistant_and_persists_created_event(harness):
    client, dispatcher, _, _ = harness

    body = create_call(client)

    assert body["session_id"].startswith("demo_")
    assert body["room_name"].startswith("armen_personal_assistant-")
    request, session_id, room_name = dispatcher.requests[0]
    assert request.task == "Забронировать стол"
    assert request.origin == "web_demo"
    assert request.owner_channel == request.result_channel == "web"
    assert request.demo_session_id == session_id
    replay = client.get(f"/api/calls/{session_id}").json()
    assert replay["status"] == "created"
    assert replay["events"][0]["type"] == "call.created"
    assert replay["events"][0]["seq"] == 1
    assert replay["events"][0]["room_name"] == room_name


def test_list_calls_returns_newest_first_with_display_fields(harness):
    client, _, _, app = harness
    first = create_call(client)
    second = create_call(client)
    app.state.store.update_session(first["session_id"], created_at=10, status="ended")
    app.state.store.update_session(second["session_id"], created_at=20, status="connected")

    response = client.get("/api/calls")

    assert response.status_code == 200
    calls = response.json()["calls"]
    assert [call["session_id"] for call in calls] == [second["session_id"], first["session_id"]]
    assert calls[0]["contact_name"] == "Ресторан"
    assert calls[0]["phone_number"] == app.state.store.get_session(second["session_id"])["request"]["phone_number"]
    assert calls[0]["task"] == "Забронировать стол"
    assert calls[0]["status"] == "connected"


@pytest.mark.parametrize(
    "payload",
    [
        {"contact_name": "A", "phone_number": "not-a-phone", "task": "Call"},
        {"contact_name": "A", "phone_number": "+77001234567", "task": "   "},
    ],
)
def test_create_call_rejects_invalid_input_without_dispatch(harness, payload):
    client, dispatcher, _, _ = harness
    assert client.post("/api/calls", json=payload).status_code == 422
    assert dispatcher.requests == []


def test_get_unknown_session_is_404(harness):
    client, _, _, _ = harness
    assert client.get("/api/calls/demo_missing").status_code == 404


def test_get_session_replays_events_after_sequence_cursor(harness):
    client, _, _, app = harness
    call = create_call(client)
    app.state.store.append_event(call["session_id"], "call.dialing", {})
    app.state.store.append_event(call["session_id"], "call.connected", {})

    response = client.get(f'/api/calls/{call["session_id"]}?after_seq=1')

    assert [event["seq"] for event in response.json()["events"]] == [2, 3]
    assert response.json()["status"] == "connected"


def test_websocket_replays_then_streams_new_events(harness):
    client, _, _, app = harness
    call = create_call(client)
    app.state.store.append_event(call["session_id"], "call.dialing", {})

    with client.websocket_connect(f'/api/calls/{call["session_id"]}/events?after_seq=1') as websocket:
        assert websocket.receive_json()["type"] == "call.dialing"
        app.state.store.append_event(call["session_id"], "call.connected", {"label": "online"})
        live = websocket.receive_json()
        assert live["seq"] == 3
        assert live["payload"] == {"label": "online"}


def test_websocket_unknown_session_closes(harness):
    client, _, _, _ = harness
    with pytest.raises(Exception):
        with client.websocket_connect("/api/calls/demo_missing/events") as websocket:
            websocket.receive_json()


def test_owner_response_resolves_matching_pending_request_once(harness):
    client, _, settings, _ = harness
    call = create_call(client)
    pending = settings.callback_dir / "pending"
    pending.mkdir(parents=True)
    (pending / "req-1.json").write_text(
        json.dumps({"request_id": "req-1", "room_name": call["room_name"], "chat_id": call["session_id"], "status": "pending"}),
        encoding="utf-8",
    )

    first = client.post(
        f'/api/calls/{call["session_id"]}/owner-response',
        json={"request_id": "req-1", "response": "yes"},
    )
    second = client.post(
        f'/api/calls/{call["session_id"]}/owner-response',
        json={"request_id": "req-1", "response": "no"},
    )

    assert first.status_code == 200
    assert first.json() == {"accepted": True}
    assert second.status_code == 409
    assert (settings.callback_dir / "responses" / "req-1.txt").read_text() == "yes"
    events = client.get(f'/api/calls/{call["session_id"]}').json()["events"]
    assert events[-1]["type"] == "owner.answer"
    assert events[-1]["payload"] == {"request_id": "req-1", "response": "yes"}


def test_owner_can_add_instruction_to_the_active_call(harness):
    client, _, settings, app = harness
    call = create_call(client)
    app.state.store.append_event(call["session_id"], "call.connected", {})

    response = client.post(
        f'/api/calls/{call["session_id"]}/instructions',
        json={"text": "Узнай, стол у окна или нет?"},
    )

    assert response.status_code == 202
    instruction_id = response.json()["instruction_id"]
    queued = json.loads((settings.callback_dir / "instructions" / call["session_id"] / f"{instruction_id}.json").read_text())
    assert queued["room_name"] == call["room_name"]
    assert queued["text"] == "Узнай, стол у окна или нет?"
    events = client.get(f'/api/calls/{call["session_id"]}').json()["events"]
    assert events[-1]["type"] == "owner.instruction"
    assert events[-1]["payload"] == {"instruction_id": instruction_id, "text": "Узнай, стол у окна или нет?"}


def test_owner_instruction_rejects_an_ended_call(harness):
    client, _, _, app = harness
    call = create_call(client)
    app.state.store.append_event(call["session_id"], "call.ended", {})

    response = client.post(
        f'/api/calls/{call["session_id"]}/instructions',
        json={"text": "Уточни ещё один вопрос"},
    )

    assert response.status_code == 409


def test_owner_instruction_rejects_non_active_created_call(harness):
    client, _, _, _ = harness
    call = create_call(client)

    response = client.post(
        f'/api/calls/{call["session_id"]}/instructions',
        json={"text": "Уточни ещё один вопрос"},
    )

    assert response.status_code == 409


@pytest.mark.parametrize("action", ["continue", "cancel"])
def test_completed_call_can_start_a_contextual_follow_up(harness, action):
    client, dispatcher, _, app = harness
    original = create_call(client)
    session_id = original["session_id"]
    app.state.store.append_event(session_id, "transcript.caller.final", {"text": "Да, бронируем на 20:00"})
    app.state.store.append_event(session_id, "transcript.assistant.final", {"text": "Спасибо, договорились"})
    app.state.store.append_event(session_id, "call.outcome", {
        "outcome": "agreed", "summary": "Забронирован стол на 20:00", "next_step": "Прийти к 20:00",
    })
    app.state.store.append_event(session_id, "call.ended", {})

    payload = {"action": action}
    if action == "continue":
        payload["instruction"] = "Уточнить, есть ли столик у окна"
    response = client.post(f"/api/calls/{session_id}/follow-up", json=payload)

    assert response.status_code == 201
    created = response.json()
    assert created["session_id"] != session_id
    request, new_session_id, _ = dispatcher.requests[-1]
    assert new_session_id == created["session_id"]
    assert request.phone_number == app.state.store.get_session(session_id)["request"]["phone_number"]
    assert request.contact_name == "Ресторан"
    assert "предыдущ" in request.details.lower()
    assert "Забронирован стол на 20:00" in request.details
    if action == "continue":
        assert request.task == "Уточнить, есть ли столик у окна"
        assert "Новое поручение Армена: Уточнить, есть ли столик у окна" in request.details
    assert "Собеседник: Да, бронируем на 20:00" in request.details
    assert "Ассистент: Спасибо, договорились" in request.details
    if action == "continue":
        assert request.task == "Уточнить, есть ли столик у окна"
    else:
        assert "отмен" in request.task.lower()
        assert "не договаривайся о новых условиях" in request.details.lower()
    replay = client.get(f"/api/calls/{created['session_id']}").json()
    assert replay["request"]["parent_session_id"] == session_id
    assert replay["request"]["follow_up_action"] == action


def test_follow_up_rejects_an_active_call(harness):
    client, dispatcher, _, app = harness
    original = create_call(client)
    app.state.store.append_event(original["session_id"], "call.connected", {})

    response = client.post(f"/api/calls/{original['session_id']}/follow-up", json={"action": "continue"})

    assert response.status_code == 409
    assert len(dispatcher.requests) == 1


def test_owner_response_rejects_request_from_another_call(harness):
    client, _, settings, _ = harness
    call = create_call(client)
    pending = settings.callback_dir / "pending"
    pending.mkdir(parents=True)
    (pending / "wrong.json").write_text(
        json.dumps({"request_id": "wrong", "room_name": "another-room", "status": "pending"}),
        encoding="utf-8",
    )
    response = client.post(
        f'/api/calls/{call["session_id"]}/owner-response',
        json={"request_id": "wrong", "response": "yes"},
    )
    assert response.status_code == 404


def test_owner_response_is_first_response_wins_under_concurrency(harness):
    client, _, settings, _ = harness
    call = create_call(client)
    pending = settings.callback_dir / "pending"
    pending.mkdir(parents=True)
    (pending / "race.json").write_text(
        json.dumps({"request_id": "race", "room_name": call["room_name"], "chat_id": call["session_id"], "status": "pending"}),
        encoding="utf-8",
    )

    def answer(value):
        return client.post(
            f'/api/calls/{call["session_id"]}/owner-response',
            json={"request_id": "race", "response": value},
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        codes = sorted(executor.map(answer, ["yes", "no"]))
    assert codes == [200, 409]


def test_recording_is_hidden_until_finalized_and_served_when_safe(harness):
    client, _, settings, app = harness
    call = create_call(client)
    recording = settings.recordings_dir / "call.ogg"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"OggS-final-audio")
    app.state.store.update_session(call["session_id"], recording_path=str(recording), recording_finalized=False)
    assert client.get(f'/api/calls/{call["session_id"]}/recording').status_code == 409

    app.state.store.update_session(call["session_id"], recording_finalized=True, status="ended")
    response = client.get(f'/api/calls/{call["session_id"]}/recording')
    assert response.status_code == 200
    assert response.content == b"OggS-final-audio"
    assert response.headers["content-type"].startswith("audio/ogg")


def test_recording_path_must_be_confined_to_recordings_directory(harness, tmp_path):
    client, _, _, app = harness
    call = create_call(client)
    outside = tmp_path / "secret.ogg"
    outside.write_bytes(b"secret")
    app.state.store.update_session(
        call["session_id"], recording_path=str(outside), recording_finalized=True, status="ended"
    )
    assert client.get(f'/api/calls/{call["session_id"]}/recording').status_code == 404


def test_token_protects_api_and_websocket(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data", callback_dir=tmp_path / "callbacks",
        recordings_dir=tmp_path / "recordings", frontend_dist=tmp_path / "dist",
        demo_token="top-secret", websocket_poll_interval=0.01,
    )
    with TestClient(create_app(settings=settings, dispatcher=FakeDispatcher())) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/api/calls", json={}).status_code == 401
        assert client.post("/api/calls", json={}, headers={"X-Demo-Token": "top-secret"}).status_code == 422
