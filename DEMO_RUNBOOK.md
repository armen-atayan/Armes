# T2 Personal Assistant Demo — Runbook

## What it shows

A desktop browser page in an iPhone frame. A presenter creates a personal-assistant call, sees caller STT and assistant TTS text live, answers `ask_owner` callbacks in the page, then receives the final outcome and recording — without Telegram messages for that web-originated call.

## Voice LLM deployment (Hermes, after review)

All 13 Gen2B personas use exactly `claude-haiku-5-5` with `llm_route=anthropic`.
Missing persona route/model overrides default to Anthropic/Haiku 5.5; the existing
`GEN2B_LLM_MODEL` environment override still applies to a missing model field.
The shared worker factory uses the native Anthropic plugin and fixes the endpoint
to `https://api.anthropic.com`. It requires a dedicated `ANTHROPIC_API_KEY` and
fails clearly when absent or blank; gateway credentials are never substituted.
See [the non-secret LLM environment example](agent/llm.env.example).

Hermes installs `agent/requirements.txt` and propagates the dedicated credential
to the worker service environment after review. The Anthropic plugin is pinned
to `1.6.7`, matching the inspected Agents/OpenAI plugins; SDK `anthropic==0.125.0`
is pinned with `thinking` and `output_config` support. SDK 1.12.1 uses httpx2 and
rejects this plugin's httpx.AsyncClient, so do not upgrade it independently.
The native plugin's `chat(extra_kwargs=...)` sends `thinking={"type":"adaptive"}`
and `output_config={"effort":"low"}` on each streamed request, with automatic
tool choice. `none` reasoning effort is not sent to Anthropic. Explicit legacy
`gateway` and `default` routes retain their OpenAI-compatible endpoints,
credentials, and reasoning configuration. STT/TTS/SIP configuration is unchanged.

The offline suite checks the real plugin's generated streaming request and SDK
signature with a mocked provider boundary. It does not establish production
model access or validate streamed thinking/tool-result round trips. Hermes owns
production preflight and the live native-plugin test, including `ask_owner`,
`finalize_call`/`end_call`, follow-up turns, greeting context, and credential
propagation, before deployment. Plugin 1.6.7's no-prefill model list predates
Haiku 5.5; HaikuChatContext therefore uses LiveKit's native trailing-user-message
serialization on a copy of the history, covered by a regression test. No model
alias or gateway fallback should replace the requested model if preflight fails.

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
