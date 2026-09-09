import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app
from settings import Settings


class FakeDispatcher:
    async def dispatch(self, *args, **kwargs):
        raise AssertionError("not used")


def test_api_reads_agent_appended_events_and_serves_finalized_recording(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        callback_dir=tmp_path / "callbacks",
        recordings_dir=tmp_path / "recordings",
        frontend_dist=tmp_path / "missing",
        demo_token=None,
    )
    app = create_app(settings=settings, dispatcher=FakeDispatcher())
    store = app.state.store
    session_id = "demo_external_agent"
    room_name = "armen_personal_assistant-demo_external_agent"
    store.create_session(session_id, room_name, {})
    recording = settings.recordings_dir / f"{room_name}.ogg"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"OggS-final-audio")
    event_path = settings.data_dir / "events" / f"{session_id}.jsonl"
    external_events = [
        {"session_id": session_id, "room_name": room_name, "seq": 1, "timestamp": time.time(), "type": "call.outcome", "payload": {"outcome": "agreed"}},
        {"session_id": session_id, "room_name": room_name, "seq": 2, "timestamp": time.time(), "type": "recording.ready", "payload": {"url": f"/api/calls/{session_id}/recording"}},
        {"session_id": session_id, "room_name": room_name, "seq": 3, "timestamp": time.time(), "type": "call.ended", "payload": {"reason": "completed"}},
    ]
    event_path.write_text("".join(json.dumps(event) + "\n" for event in external_events), encoding="utf-8")

    with TestClient(app) as client:
        snapshot = client.get(f"/api/calls/{session_id}")
        response = client.get(f"/api/calls/{session_id}/recording")

    assert snapshot.status_code == 200
    assert snapshot.json()["status"] == "ended"
    assert response.status_code == 200
    assert response.content == b"OggS-final-audio"


def test_api_recovers_finalization_when_worker_crashes_after_outcome(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        callback_dir=tmp_path / "callbacks",
        recordings_dir=tmp_path / "recordings",
        frontend_dist=tmp_path / "missing",
        demo_token=None,
    )
    app = create_app(settings=settings, dispatcher=FakeDispatcher())
    store = app.state.store
    session_id = "demo_crashed_after_outcome"
    room_name = "armen_personal_assistant-demo_crashed_after_outcome"
    store.create_session(session_id, room_name, {})
    recording = settings.recordings_dir / f"{room_name}.ogg"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"OggS-final-audio")
    (settings.recordings_dir / "EG_finished.json").write_text(json.dumps({
        "room_name": room_name,
        "ended_at": time.time_ns(),
        "files": [{"filename": f"/out/{room_name}.ogg"}],
    }), encoding="utf-8")
    event_path = settings.data_dir / "events" / f"{session_id}.jsonl"
    event_path.write_text(json.dumps({
        "session_id": session_id, "room_name": room_name, "seq": 1,
        "timestamp": time.time(), "type": "call.outcome",
        "payload": {"outcome": "agreed", "summary": "Бронь подтверждена"},
    }) + "\n", encoding="utf-8")

    with TestClient(app) as client:
        snapshot = client.get(f"/api/calls/{session_id}")
        response = client.get(f"/api/calls/{session_id}/recording")

    assert snapshot.status_code == 200
    assert snapshot.json()["status"] == "ended"
    assert [event["type"] for event in snapshot.json()["events"]][-3:] == [
        "call.outcome", "recording.ready", "call.ended",
    ]
    assert response.status_code == 200
    assert response.content == b"OggS-final-audio"
