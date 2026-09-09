import json
from types import SimpleNamespace

import pytest

import call_with_persona as c


class FakeDispatchService:
    def __init__(self):
        self.request = None

    async def create_dispatch(self, request):
        self.request = request


class FakeSipService:
    def __init__(self):
        self.request = None

    async def create_sip_participant(self, request):
        self.request = request
        return SimpleNamespace(sip_call_id="sip-call", participant_id="participant")


class FakeLiveKitAPI:
    def __init__(self):
        self.agent_dispatch = FakeDispatchService()
        self.sip = FakeSipService()
        self.closed = False

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_dispatch_call_uses_web_metadata_and_returns_call_handle(monkeypatch):
    fake = FakeLiveKitAPI()
    monkeypatch.setattr(c, "load_personas", lambda: {
        "armen_personal_assistant": {
            "name": "Ассистент Армена",
            "sip_trunk_id": "ST_demo",
            "sip_number": "74950000000",
        }
    })

    result = await c.dispatch_call(
        persona="armen_personal_assistant",
        phone_number="+77077080038",
        target_name="Антон",
        target_identity="anton",
        task="Согласовать встречу",
        task_details="Завтра",
        room_name="demo-room",
        origin="web_demo",
        owner_channel="web",
        result_channel="web",
        demo_session_id="demo_123",
        livekit_api=fake,
    )

    metadata = json.loads(fake.agent_dispatch.request.metadata)
    assert metadata["demo_session_id"] == "demo_123"
    assert metadata["owner_channel"] == metadata["result_channel"] == "web"
    assert fake.sip.request.room_name == "demo-room"
    assert not fake.sip.request.sip_call_to.startswith("+")
    assert result["sip_call_id"] == "sip-call"
    assert fake.closed is False
