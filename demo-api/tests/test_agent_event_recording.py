from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app
from settings import Settings


class FakeDispatcher:
    async def dispatch(self, *args, **kwargs):
        raise AssertionError("not used")


def test_recording_from_agent_event_is_served_without_exposing_host_path(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        callback_dir=tmp_path / "callbacks",
        recordings_dir=tmp_path / "recordings",
        frontend_dist=tmp_path / "missing",
        demo_token=None,
    )
    app = create_app(settings=settings, dispatcher=FakeDispatcher())
    store = app.state.store
    session_id = "demo_agent_event"
    room_name = "armen_personal_assistant-demo_agent_event"
    store.create_session(session_id, room_name, {})
    recording = settings.recordings_dir / f"{room_name}.ogg"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"OggS-final-audio")
    # This is the exact public event written by the voice worker, not an API mutation.
    store.append_event(session_id, "recording.ready", {"url": f"/api/calls/{session_id}/recording"})
    store.append_event(session_id, "call.ended", {"reason": "completed"})

    with TestClient(app) as client:
        response = client.get(f"/api/calls/{session_id}/recording")

    assert response.status_code == 200
    assert response.content == b"OggS-final-audio"
