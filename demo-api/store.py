from __future__ import annotations

import json
import fcntl
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any


_STATUS_BY_EVENT = {
    "call.created": "created", "call.dialing": "dialing", "call.connected": "connected",
    "call.ended": "ended", "call.failed": "failed", "call.outcome": "completed",
}


class EventStore:
    """Append-only per-session JSONL events plus atomic session metadata."""

    def __init__(self, root: Path, *, recordings_dir: Path | None = None):
        self.root = Path(root)
        self.recordings_dir = Path(recordings_dir) if recordings_dir else self.root.parent / "call-recordings"
        self.sessions_dir = self.root / "sessions"
        self.events_dir = self.root / "events"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _valid_session_id(session_id: str) -> bool:
        return session_id.startswith("demo_") and all(c.isalnum() or c in "_-" for c in session_id)

    def _session_path(self, session_id: str) -> Path:
        if not self._valid_session_id(session_id):
            raise KeyError(session_id)
        return self.sessions_dir / f"{session_id}.json"

    def _events_path(self, session_id: str) -> Path:
        if not self._valid_session_id(session_id):
            raise KeyError(session_id)
        return self.events_dir / f"{session_id}.jsonl"

    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)

    def create_session(self, session_id: str, room_name: str, request: dict[str, Any]) -> dict[str, Any]:
        session = {
            "session_id": session_id, "room_name": room_name, "status": "created",
            "created_at": time.time(), "request": request, "recording_path": None,
            "recording_finalized": False,
        }
        with self._lock:
            if self._session_path(session_id).exists():
                raise ValueError("session already exists")
            self._atomic_write(self._session_path(session_id), session)
            self._events_path(session_id).touch(exist_ok=False)
        return session

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        try:
            path = self._session_path(session_id)
            return json.loads(path.read_text(encoding="utf-8"))
        except (KeyError, FileNotFoundError, json.JSONDecodeError):
            return None

    def update_session(self, session_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            session = self.get_session(session_id)
            if session is None:
                raise KeyError(session_id)
            session.update(changes)
            self._atomic_write(self._session_path(session_id), session)
            return session

    def _recover_finalized_recording_events(self, session_id: str, session: dict[str, Any]) -> None:
        """Complete web delivery if a worker died after egress finalized the file."""
        events = self.read_events(session_id)
        types = [str(event.get("type", "")) for event in events]
        if "call.outcome" not in types or ("recording.ready" in types and "call.ended" in types):
            return
        recording = self.recordings_dir / f"{session['room_name']}.ogg"
        metadata_finalized = False
        for metadata in self.recordings_dir.glob("EG_*.json"):
            try:
                payload = json.loads(metadata.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("room_name") == session["room_name"] and int(payload.get("ended_at") or 0) > 0:
                metadata_finalized = True
                break
        if not recording.is_file() or recording.stat().st_size == 0 or not metadata_finalized:
            return
        if "recording.ready" not in types:
            self.append_event(session_id, "recording.ready", {
                "url": f"/api/calls/{session_id}/recording",
            })
        if "call.ended" not in types:
            self.append_event(session_id, "call.ended", {"reason": "recovered_after_worker_exit"})

    def projected_session(self, session_id: str) -> dict[str, Any] | None:
        """Project agent-appended event state without requiring the agent to call this API."""
        session = self.get_session(session_id)
        if session is None:
            return None
        self._recover_finalized_recording_events(session_id, session)
        for event in self.read_events(session_id):
            event_type = str(event.get("type", ""))
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            status = (
                payload.get("state")
                if event_type == "call.state"
                else (_STATUS_BY_EVENT.get(event_type) if event_type != "call.created" else None)
            )
            if status:
                session["status"] = status
            if event_type == "recording.ready":
                session["recording_path"] = str(self.recordings_dir / f"{session['room_name']}.ogg")
                session["recording_finalized"] = True
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in self.sessions_dir.glob("demo_*.json"):
            session = self.projected_session(path.stem)
            if session is None:
                continue
            request = session.get("request") if isinstance(session.get("request"), dict) else {}
            sessions.append({
                "session_id": session["session_id"],
                "contact_name": str(request.get("contact_name") or "Собеседник"),
                "phone_number": str(request.get("phone_number") or ""),
                "task": str(request.get("task") or ""),
                "status": session.get("status", "created"),
                "created_at": session.get("created_at", 0),
            })
        return sorted(sessions, key=lambda item: float(item["created_at"]), reverse=True)

    def read_events(self, session_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        if self.get_session(session_id) is None:
            raise KeyError(session_id)
        events = []
        try:
            lines = self._events_path(session_id).read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return events
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(event.get("seq", 0)) > after_seq:
                events.append(event)
        return events

    def append_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session = self.get_session(session_id)
            if session is None:
                raise KeyError(session_id)
            with self._events_path(session_id).with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                return self._append_event_locked(session_id, session, event_type, payload)

    def _append_event_locked(self, session_id, session, event_type, payload):
        existing = self.read_events(session_id)
        event = {
            "session_id": session_id, "room_name": session["room_name"],
            "seq": (existing[-1]["seq"] if existing else 0) + 1,
            "timestamp": time.time(), "type": event_type, "payload": payload,
        }
        with self._events_path(session_id).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        status = payload.get("state") if event_type == "call.state" else _STATUS_BY_EVENT.get(event_type)
        changes: dict[str, Any] = {}
        if status:
            changes["status"] = status
        if event_type == "recording.ready":
            changes.update(
                recording_path=str(self.recordings_dir / f"{session['room_name']}.ogg"),
                recording_finalized=True,
            )
        if changes:
            self.update_session(session_id, **changes)
        return event
