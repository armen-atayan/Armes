from __future__ import annotations

import asyncio
import hmac
import json
import mimetypes
import secrets
import threading
from pathlib import Path
from typing import Any, Protocol

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dispatch import default_dispatcher
from models import CallCreated, CallDispatchResult, CallRequest, FollowUpRequest, LiveInstruction, OwnerResponse
from settings import Settings
from store import EventStore
from transcription import Gen2BTranscriber


class Dispatcher(Protocol):
    async def dispatch(self, request: CallRequest, *, session_id: str, room_name: str) -> CallDispatchResult: ...


class Transcriber(Protocol):
    async def transcribe(self, audio: bytes, content_type: str) -> str: ...


def _authorized(expected: str | None, supplied: str | None) -> bool:
    return expected is None or (supplied is not None and hmac.compare_digest(expected, supplied))


def _load_callback_resolver():
    import importlib.util

    path = Path("/home/ubuntu/livekit-stack/agent/live_callback.py")
    spec = importlib.util.spec_from_file_location("demo_live_callback", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("live callback bridge is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve_request, module.enqueue_instruction


def create_app(*, settings: Settings | None = None, dispatcher: Dispatcher | None = None, transcriber: Transcriber | None = None) -> FastAPI:
    settings = settings or Settings()
    store = EventStore(settings.data_dir, recordings_dir=settings.recordings_dir)
    dispatcher = dispatcher or default_dispatcher()
    transcriber = transcriber or Gen2BTranscriber()
    resolve_request, enqueue_instruction = _load_callback_resolver()
    owner_response_lock = threading.Lock()
    app = FastAPI(title="LiveKit Personal Assistant Demo API", version="1.0.0")
    app.state.store = store
    app.state.settings = settings
    app.state.dispatcher = dispatcher

    def require_token(x_demo_token: str | None = Header(default=None)) -> None:
        if not _authorized(settings.demo_token, x_demo_token):
            raise HTTPException(status_code=401, detail="Invalid demo token")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    protected = [Depends(require_token)]

    @app.post("/api/transcribe", dependencies=protected)
    async def transcribe_voice(request: Request, audio: bytes = Body()) -> dict[str, str]:
        content_type = request.headers.get("content-type", "application/octet-stream").split(";", 1)[0]
        if len(audio) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Voice message is too large")
        try:
            return {"text": await transcriber.transcribe(audio, content_type)}
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Voice transcription failed") from exc

    @app.get("/api/calls", dependencies=protected)
    async def list_calls() -> dict[str, list[dict[str, Any]]]:
        return {"calls": store.list_sessions()}

    @app.post("/api/calls", response_model=CallCreated, status_code=status.HTTP_201_CREATED, dependencies=protected)
    async def create_call(payload: CallRequest) -> CallCreated:
        session_id = f"demo_{secrets.token_hex(8)}"
        room_name = f"armen_personal_assistant-{session_id}"
        payload.demo_session_id = session_id
        store.create_session(session_id, room_name, payload.model_dump())
        store.append_event(session_id, "call.created", {"contact_name": payload.contact_name})
        try:
            result = await dispatcher.dispatch(payload, session_id=session_id, room_name=room_name)
        except Exception as exc:
            store.append_event(session_id, "call.failed", {"message": "Call dispatch failed"})
            raise HTTPException(status_code=503, detail="Call dispatch unavailable") from exc
        if result.room_name != room_name:
            store.update_session(session_id, room_name=result.room_name)
        return CallCreated(session_id=session_id, room_name=result.room_name)

    @app.get("/api/calls/{session_id}", dependencies=protected)
    async def get_call(
        session_id: str, after_seq: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        session = store.projected_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {**session, "events": store.read_events(session_id, after_seq)}

    @app.post("/api/calls/{session_id}/follow-up", response_model=CallCreated, status_code=status.HTTP_201_CREATED, dependencies=protected)
    async def follow_up_call(session_id: str, payload: FollowUpRequest) -> CallCreated:
        original = store.projected_session(session_id)
        if original is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if original.get("status") not in {"ended", "completed", "failed"}:
            raise HTTPException(status_code=409, detail="Call is still active")
        request_data = original.get("request") if isinstance(original.get("request"), dict) else {}
        phone_number = str(request_data.get("phone_number") or "")
        if not phone_number:
            raise HTTPException(status_code=409, detail="Original phone number is unavailable")
        events = store.read_events(session_id)
        outcome_payload = next((event.get("payload", {}) for event in reversed(events) if event.get("type") == "call.outcome"), {})
        summary = str(outcome_payload.get("summary") or outcome_payload.get("outcome") or request_data.get("task") or "предыдущая договорённость")
        next_step = str(outcome_payload.get("next_step") or "")
        if payload.action == "cancel":
            task = f"Перезвони и отмени предыдущую договорённость: {summary}"
            action_rules = "Цель звонка — только отменить договорённость. Не договаривайся о новых условиях и не заменяй её другой договорённостью без отдельного поручения Армена. Получи явное подтверждение отмены."
        else:
            if not payload.instruction:
                raise HTTPException(status_code=422, detail="Follow-up instruction is required")
            task = payload.instruction
            action_rules = "Продолжи предыдущий разговор с учётом достигнутой договорённости. Не представляй это как первый звонок и не выдумывай отсутствующие детали."
        context = "\n".join(part for part in [
            f"Контекст предыдущего завершённого звонка: {summary}.",
            f"Следующий шаг из прошлого звонка: {next_step}." if next_step else "",
            f"Исходное поручение: {request_data.get('task', '')}.",
            f"Новое поручение Армена: {payload.instruction}." if payload.action == "continue" else "",
            action_rules,
        ] if part)
        follow_request = CallRequest(
            contact_name=str(request_data.get("contact_name") or "Собеседник"),
            phone_number=phone_number,
            task=task,
            details=context,
        )
        new_session_id = f"demo_{secrets.token_hex(8)}"
        room_name = f"armen_personal_assistant-{new_session_id}"
        follow_request.demo_session_id = new_session_id
        stored_request = follow_request.model_dump() | {"parent_session_id": session_id, "follow_up_action": payload.action}
        store.create_session(new_session_id, room_name, stored_request)
        store.append_event(new_session_id, "call.created", {"contact_name": follow_request.contact_name})
        try:
            result = await dispatcher.dispatch(follow_request, session_id=new_session_id, room_name=room_name)
        except Exception as exc:
            store.append_event(new_session_id, "call.failed", {"message": "Call dispatch failed"})
            raise HTTPException(status_code=503, detail="Call dispatch unavailable") from exc
        if result.room_name != room_name:
            store.update_session(new_session_id, room_name=result.room_name)
        return CallCreated(session_id=new_session_id, room_name=result.room_name)

    @app.websocket("/api/calls/{session_id}/events")
    async def events(websocket: WebSocket, session_id: str, after_seq: int = Query(default=0, ge=0)) -> None:
        supplied = websocket.headers.get("x-demo-token") or websocket.query_params.get("token")
        if not _authorized(settings.demo_token, supplied):
            await websocket.close(code=4401)
            return
        if store.get_session(session_id) is None:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        cursor = after_seq
        try:
            while True:
                for event in store.read_events(session_id, cursor):
                    await websocket.send_json(event)
                    cursor = event["seq"]
                await asyncio.sleep(settings.websocket_poll_interval)
        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            raise

    @app.post("/api/calls/{session_id}/owner-response", dependencies=protected)
    async def owner_response(
        session_id: str, payload: OwnerResponse,
    ) -> dict[str, bool]:
        session = store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        pending_path = settings.callback_dir / "pending" / f"{payload.request_id}.json"
        try:
            pending = json.loads(pending_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            raise HTTPException(status_code=404, detail="Owner request not found")
        if pending.get("room_name") != session["room_name"] or str(pending.get("chat_id", "")) != session_id:
            raise HTTPException(status_code=404, detail="Owner request not found")
        response_path = settings.callback_dir / "responses" / f"{payload.request_id}.txt"
        with owner_response_lock:
            if response_path.exists() or pending.get("status") == "answered":
                raise HTTPException(status_code=409, detail="Owner request already answered")
            if not resolve_request(settings.callback_dir, payload.request_id, payload.response):
                raise HTTPException(status_code=409, detail="Owner response rejected")
        store.append_event(
            session_id, "owner.answer",
            {"request_id": payload.request_id, "response": payload.response},
        )
        return {"accepted": True}

    @app.post("/api/calls/{session_id}/instructions", status_code=status.HTTP_202_ACCEPTED, dependencies=protected)
    async def add_instruction(session_id: str, payload: LiveInstruction) -> dict[str, str]:
        session = store.projected_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.get("status") not in {"connected", "listening", "thinking", "speaking", "waiting_owner"}:
            raise HTTPException(status_code=409, detail="Call is not active")
        instruction_id = enqueue_instruction(
            settings.callback_dir,
            session_id=session_id,
            room_name=session["room_name"],
            text=payload.text,
        )
        store.append_event(session_id, "owner.instruction", {"instruction_id": instruction_id, "text": payload.text})
        return {"instruction_id": instruction_id, "status": "queued"}

    @app.get("/api/calls/{session_id}/recording", dependencies=protected)
    async def recording(session_id: str) -> FileResponse:
        session = store.projected_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if not session.get("recording_finalized") or session.get("status") not in {"ended", "completed"}:
            raise HTTPException(status_code=409, detail="Recording is not finalized")
        configured = session.get("recording_path")
        if not configured:
            raise HTTPException(status_code=404, detail="Recording not found")
        root = settings.recordings_dir.resolve()
        path = Path(configured).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise HTTPException(status_code=404, detail="Recording not found")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Recording not found")
        media_type, _ = mimetypes.guess_type(path.name)
        return FileResponse(path, media_type=media_type or "application/octet-stream", filename=path.name)

    # Serve a built UI if present, without shadowing API routes.
    if settings.frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")

    return app


app = create_app()
