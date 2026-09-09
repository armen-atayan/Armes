import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import demo_events


def test_emit_assigns_monotonic_sequence_and_replays(tmp_path):
    store = demo_events.DemoEventStore(tmp_path)

    first = store.emit("demo_one", "room-one", "call.created", {"contact": "Антон"})
    second = store.emit("demo_one", "room-one", "call.state", {"state": "dialing"})

    assert first["seq"] == 1
    assert second["seq"] == 2
    assert [event["type"] for event in store.replay("demo_one")] == [
        "call.created",
        "call.state",
    ]


def test_emit_allowlists_payload_fields(tmp_path):
    store = demo_events.DemoEventStore(tmp_path)

    event = store.emit(
        "demo_one",
        "room-one",
        "transcript.caller.final",
        {"utterance_id": "u1", "text": "Здравствуйте", "system_prompt": "secret"},
    )

    assert event["payload"] == {"utterance_id": "u1", "text": "Здравствуйте"}
    assert "secret" not in json.dumps(event, ensure_ascii=False)


def test_unknown_event_type_is_rejected(tmp_path):
    store = demo_events.DemoEventStore(tmp_path)

    with pytest.raises(ValueError, match="unsupported demo event"):
        store.emit("demo_one", "room-one", "debug.log", {"text": "no"})


def test_concurrent_emit_keeps_unique_ordered_sequence(tmp_path):
    store = demo_events.DemoEventStore(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda index: store.emit(
                    "demo_one", "room-one", "call.state", {"state": f"s{index}"}
                ),
                range(30),
            )
        )

    events = store.replay("demo_one")
    assert [event["seq"] for event in events] == list(range(1, 31))


def test_invalid_session_identifier_is_rejected(tmp_path):
    store = demo_events.DemoEventStore(tmp_path)

    with pytest.raises(ValueError, match="invalid session"):
        store.emit("../../escape", "room", "call.created", {})
@pytest.mark.asyncio
async def test_web_outcome_is_published_before_recording_wait(monkeypatch, tmp_path):
    import gen2b_agent as g
    from types import SimpleNamespace

    emitted = []
    monkeypatch.setattr(g, "emit_demo_event", lambda sid, room, kind, payload: emitted.append(kind))
    monkeypatch.setattr(g, "CALL_RECORDINGS_DIR", tmp_path)

    class Pending:
        outcome = {"outcome": "agreed", "summary": "Бронь подтверждена", "agreed_price_kzt": None,
                   "agreed_price_gbp": None, "volume_m3": None, "next_step": ""}

    # Outcome publication must not be held behind a finalized recording or worker close.
    await g.publish_web_outcome("demo-1", "room-1", Pending.outcome)
    assert emitted == ["call.outcome"]
