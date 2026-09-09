"""Filesystem-backed live approval bridge for an active voice call."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def create_request(root: Path, *, room_name: str, chat_id: str, question: str) -> str:
    request_id = uuid.uuid4().hex[:12]
    _atomic_json(
        root / "pending" / f"{request_id}.json",
        {
            "request_id": request_id,
            "room_name": room_name,
            "chat_id": str(chat_id),
            "question": question,
            "status": "pending",
            "created_at": time.time(),
        },
    )
    return request_id


def resolve_request(root: Path, request_id: str, response: str, *, chat_id: str | None = None) -> bool:
    pending = root / "pending" / f"{request_id}.json"
    if not pending.exists():
        return False
    if chat_id is not None:
        try:
            payload = json.loads(pending.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if str(payload.get("chat_id")) != str(chat_id):
            return False
    response = response.strip()
    if not response:
        return False
    responses = root / "responses"
    responses.mkdir(parents=True, exist_ok=True)
    temporary = responses / f".{request_id}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(response, encoding="utf-8")
    os.replace(temporary, responses / f"{request_id}.txt")
    return True


def mark_awaiting_text(root: Path, request_id: str, chat_id: str) -> bool:
    pending = root / "pending" / f"{request_id}.json"
    if not pending.exists():
        return False
    payload = json.loads(pending.read_text(encoding="utf-8"))
    if str(payload.get("chat_id")) != str(chat_id):
        return False
    payload["status"] = "awaiting_text"
    _atomic_json(pending, payload)
    return True


def capture_text_response(root: Path, *, chat_id: str, text: str) -> str | None:
    pending_dir = root / "pending"
    if not pending_dir.exists():
        return None
    candidates = []
    for path in pending_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "awaiting_text" and str(payload.get("chat_id")) == str(chat_id):
            candidates.append((float(payload.get("created_at", 0)), path, payload))
    if not candidates:
        return None
    _, path, payload = max(candidates, key=lambda item: item[0])
    request_id = str(payload["request_id"])
    if not resolve_request(root, request_id, text):
        return None
    payload["status"] = "answered"
    _atomic_json(path, payload)
    return request_id


def enqueue_instruction(root: Path, *, session_id: str, room_name: str, text: str) -> str:
    """Queue a new owner-initiated instruction for one active web call."""
    instruction_id = uuid.uuid4().hex[:12]
    _atomic_json(
        root / "instructions" / session_id / f"{instruction_id}.json",
        {
            "instruction_id": instruction_id,
            "session_id": session_id,
            "room_name": room_name,
            "text": text.strip(),
            "created_at": time.time(),
        },
    )
    return instruction_id


def pop_instruction(root: Path, *, session_id: str, room_name: str) -> dict | None:
    """Atomically claim the oldest queued instruction for this room."""
    directory = root / "instructions" / session_id
    for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime):
        claimed = path.with_suffix(".claimed")
        try:
            os.replace(path, claimed)
        except FileNotFoundError:
            continue
        try:
            payload = json.loads(claimed.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            claimed.unlink(missing_ok=True)
            continue
        if payload.get("room_name") != room_name:
            os.replace(claimed, path)
            return None
        claimed.unlink(missing_ok=True)
        return {"instruction_id": str(payload["instruction_id"]), "text": str(payload["text"])}
    return None


async def wait_for_response(
    root: Path,
    request_id: str,
    *,
    timeout: float = 180.0,
    poll_interval: float = 0.25,
) -> str:
    response_path = root / "responses" / f"{request_id}.txt"
    pending_path = root / "pending" / f"{request_id}.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = response_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            response = ""
        if response:
            response_path.unlink(missing_ok=True)
            pending_path.unlink(missing_ok=True)
            return response
        await asyncio.sleep(poll_interval)
    pending_path.unlink(missing_ok=True)
    return "timeout"
