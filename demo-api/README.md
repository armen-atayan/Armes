# LiveKit Demo API

FastAPI backend for the personal-assistant browser demo. Session metadata and ordered JSONL events are stored below `DEMO_DATA_DIR` (default: `/home/ubuntu/livekit-stack/demo-data`).

## Run

```bash
cd /home/ubuntu/livekit-stack/demo-api
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8080
```

Set `DEMO_TOKEN` when binding beyond localhost. Send it as `X-Demo-Token`; WebSockets may use that header or `?token=...`.

The production dispatcher imports the agent's `call_with_persona` module and requires it to expose an async `dispatch_call(...)` or `call_with_persona(...)`. Tests inject a fake dispatcher and never place calls.

## Endpoints

- `GET /health`
- `POST /api/calls`
- `GET /api/calls/{session_id}?after_seq=N`
- `WS /api/calls/{session_id}/events?after_seq=N`
- `POST /api/calls/{session_id}/owner-response`
- `GET /api/calls/{session_id}/recording`

A built frontend is served from `DEMO_FRONTEND_DIST` when that directory exists.

## Test

```bash
.venv/bin/pytest -q
```
