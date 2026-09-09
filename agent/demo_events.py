"""Safe append-only event transport for the browser voice-agent demo."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,120}$")
_EVENT_FIELDS: dict[str, frozenset[str]] = {
    "call.created": frozenset({"contact", "phone", "task"}),
    "call.dialing": frozenset({"contact"}),
    "call.connected": frozenset({"contact"}),
    "call.state": frozenset({"state"}),
    "transcript.caller.partial": frozenset({"utterance_id", "text"}),
    "transcript.caller.final": frozenset({"utterance_id", "text"}),
    "transcript.assistant.delta": frozenset({"utterance_id", "text"}),
    "transcript.assistant.final": frozenset({"utterance_id", "text"}),
    "transcript.assistant.interrupted": frozenset({"utterance_id", "text"}),
    "owner.question": frozenset({"request_id", "question", "context", "options"}),
    "owner.answer": frozenset({"request_id", "answer", "action"}),
    "owner.instruction.accepted": frozenset({"instruction_id"}),
    "call.outcome": frozenset(
        {"outcome", "summary", "next_step", "agreed_price_kzt", "agreed_price_gbp", "volume_m3"}
    ),
    "recording.ready": frozenset({"url"}),
    "call.ended": frozenset({"reason"}),
    "call.failed": frozenset({"message"}),
}


class DemoEventStore:
    """Persist sanitized events as one JSONL file per unguessable demo session."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._lock = threading.Lock()

    def _path(self, session_id: str) -> Path:
        if not _SAFE_ID.fullmatch(session_id):
            raise ValueError("invalid session identifier")
        return self.root / "events" / f"{session_id}.jsonl"

    def replay(self, session_id: str, after: int = 0) -> list[dict[str, Any]]:
        path = self._path(session_id)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        events = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(event.get("seq", 0)) > after:
                events.append(event)
        return events

    def emit(
        self,
        session_id: str,
        room_name: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        allowed = _EVENT_FIELDS.get(event_type)
        if allowed is None:
            raise ValueError(f"unsupported demo event: {event_type}")
        path = self._path(session_id)
        sanitized = {key: value for key, value in payload.items() if key in allowed}
        with self._lock:
            existing = self.replay(session_id)
            event = {
                "session_id": session_id,
                "room_name": room_name,
                "seq": (int(existing[-1]["seq"]) + 1) if existing else 1,
                "timestamp": time.time(),
                "type": event_type,
                "payload": sanitized,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
        return event


_default_store = DemoEventStore(
    Path("/home/ubuntu/livekit-stack/demo-data")
)


def emit_demo_event(
    session_id: str,
    room_name: str,
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return _default_store.emit(session_id, room_name, event_type, payload)
