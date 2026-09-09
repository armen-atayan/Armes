import json

import pytest

from call_control import hang_up_sip_participant, write_call_outcome


class FakeRoomService:
    def __init__(self):
        self.requests = []

    async def remove_participant(self, request):
        self.requests.append(request)


class FakeLiveKitAPI:
    def __init__(self):
        self.room = FakeRoomService()
        self.closed = False

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_hang_up_sip_participant_removes_the_callee_and_closes_api_client():
    fake_api = FakeLiveKitAPI()

    await hang_up_sip_participant(
        room_name="negotiation-room",
        participant_identity="armen",
        api_factory=lambda: fake_api,
    )

    assert len(fake_api.room.requests) == 1
    request = fake_api.room.requests[0]
    assert request.room == "negotiation-room"
    assert request.identity == "armen"
    assert fake_api.closed is True


def test_write_call_outcome_creates_atomic_pending_result(tmp_path):
    path = write_call_outcome(
        results_dir=tmp_path,
        room_name="negotiation-room",
        outcome="agreed",
        summary="Согласовали 3 000 ₸/м³ за 550 м³.",
        agreed_price_kzt=3000,
        volume_m3=550,
        next_step="Продавец пишет Армену в Telegram.",
    )

    assert path.name == "negotiation-room.json"
    assert path.parent.name == "pending"
    result = json.loads(path.read_text())
    assert result == {
        "room_name": "negotiation-room",
        "outcome": "agreed",
        "summary": "Согласовали 3 000 ₸/м³ за 550 м³.",
        "agreed_price_kzt": 3000,
        "agreed_price_gbp": None,
        "volume_m3": 550,
        "next_step": "Продавец пишет Армену в Telegram.",
        "recording_path": None,
    }
    assert not list(tmp_path.rglob("*.tmp"))
