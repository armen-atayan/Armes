from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app
from settings import Settings


class FakeDispatcher:
    async def dispatch(self, *args, **kwargs):
        raise AssertionError("not used")


def test_owner_response_rejects_pending_request_with_other_demo_session(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        callback_dir=tmp_path / "callbacks",
        recordings_dir=tmp_path / "recordings",
        frontend_dist=tmp_path / "missing",
        demo_token=None,
    )
    app = create_app(settings=settings, dispatcher=FakeDispatcher())
    session_id = "demo_session_one"
    room_name = "room-one"
    app.state.store.create_session(session_id, room_name, {})
    pending = settings.callback_dir / "pending"
    pending.mkdir(parents=True)
    (pending / "foreign.json").write_text(
        '{"request_id":"foreign","room_name":"room-one","chat_id":"demo_session_two","status":"pending"}',
        encoding="utf-8",
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/calls/{session_id}/owner-response",
            json={"request_id": "foreign", "response": "yes"},
        )

    assert response.status_code == 404
