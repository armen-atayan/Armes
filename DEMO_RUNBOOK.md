# T2 Personal Assistant Demo — Runbook

## What it shows

A desktop browser page in an iPhone frame. A presenter creates a personal-assistant call, sees caller STT and assistant TTS text live, answers `ask_owner` callbacks in the page, then receives the final outcome and recording — without Telegram messages for that web-originated call.

## Start / restart

```bash
cd /home/ubuntu/livekit-stack/demo-ui && npm run build
systemctl --user daemon-reload
systemctl --user restart t2-demo-api.service
systemctl --user restart gen2b-agent.service
journalctl --user -u gen2b-agent.service -n 80 --no-pager
```

Wait for a fresh `registered worker` line before placing a real call.

Open locally:

```text
http://127.0.0.1:8099/
```

For a presentation computer that is not this host, use a protected HTTPS reverse proxy or a controlled SSH tunnel. Do not expose port 8099 publicly without setting `DEMO_TOKEN`.

## One-time setup

Create `/home/ubuntu/livekit-stack/.demo.env` mode `0600`:

```bash
DEMO_TOKEN=<long-random-value>
```

When `DEMO_TOKEN` is set, the API expects `X-Demo-Token`. For browser use behind a reverse proxy, inject the header there, or use the same token as a controlled query parameter for WebSocket only. For local rehearsal, leave the token absent.

Enable the service:

```bash
cp /home/ubuntu/livekit-stack/demo-api/t2-demo-api.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now t2-demo-api.service
curl -fsS http://127.0.0.1:8099/health
```

## Rehearsal

1. Start a task from the browser.
2. Confirm the web UI shows caller text and assistant text in the live feed.
3. Trigger a task that needs a decision; confirm only the web card appears, with no Telegram callback.
4. Press `Подтвердить`, `Отказаться`, or `Другое`; confirm the same call continues.
5. After hangup, confirm the result and recording player appear.
6. Check event evidence without showing it to the audience:

```bash
ls -lt /home/ubuntu/livekit-stack/demo-data/events
journalctl --user -u gen2b-agent.service -n 200 --no-pager
```

## Reset between rehearsals

Browser: press `Новая задача` after a finished call. It clears only the browser's active-session pointer.

Server event history is intentionally retained for refresh recovery. To remove only old demo artifacts between rehearsals:

```bash
rm -f /home/ubuntu/livekit-stack/demo-data/events/demo_*.jsonl
rm -f /home/ubuntu/livekit-stack/demo-data/sessions/demo_*.json
```

Do not delete live callbacks while a call is waiting for an owner decision.
