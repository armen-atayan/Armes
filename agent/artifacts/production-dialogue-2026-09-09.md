# Production dialogue investigation — 9 September 2026

Investigated at HEAD `5def769`, with a clean initial worktree. No service restart,
production call, event rewrite, commit, or push was performed.

## Evidence and reconstruction

Sources: `demo-data/sessions/demo_276a78ff7a76d595.json`,
`demo-data/sessions/demo_3b91d08158adf0e7.json`, their corresponding
`demo-data/events/*.jsonl`, and
`journalctl --user -u gen2b-agent.service --since '2026-09-09 07:10:00' --until '2026-09-09 07:20:00' --no-pager`.
Times below are host-local UTC+05:00 (journal JSON timestamps are UTC).

Read all 518 and 226 canonical events, including states, owner decisions, outcome,
and recording events. Sequence numbers are contiguous. Concatenating every assistant
delta by utterance ID exactly matches all 8 and 4 assistant finals respectively;
there are no orphan delta streams. Each owner response appears twice in the event
stream (`response` from API, then `answer`/`action` from worker), with the same
request ID and value. These are not two owner decisions.

First call, `demo_276a78ff7a76d595`:

- Initial metadata requests a restaurant table for ten, with no date/time/name.
- 07:11:04, seq 95: assistant asks whether a table for ten can be booked.
- Seq 100/130–131: restaurant asks when; owner supplies tomorrow at 20:00.
- Seq 220/222: assistant requests that slot; restaurant offers 19:00 or 22:00.
- Seq 224/254–255: owner selects 22:00.
- Seq 348: assistant requests tomorrow at 22:00 for ten and asks to confirm.
- Seq 350–352: restaurant confirms subject to deposit options: 20,000 rubles
  without discount, 50,000 with 20%, or 100,000 with 50%.
- Seq 354/384–385: owner selects 50,000 rubles with 20% discount.
- 07:12:55: journal `FINALIZE_BLOCKED` correctly rejects an attempt before
  the new condition has been accepted by the restaurant.
- 07:12:58, seq 486: assistant asks to book ten tomorrow at 22:00 with the
  owner-approved deposit/discount. Seq 488: restaurant says **«Да да получится.»**
- 07:13:12 and 07:13:14: journal `FINALIZE_BLOCKED` incorrectly rejects this
  positive reply. Seq 514: assistant says goodbye. At 07:13:21 seq 516–517,
  the three-second farewell timer records `incomplete`, reason `farewell_silence`.
- No booking name was collected or communicated. The evidence supports acceptance
  of the spoken conditions; it does not establish an independently verified
  reservation record or a reservation under a particular name.

Follow-up, `demo_3b91d08158adf0e7`:

- Metadata links `parent_session_id` to the first call, action `continue`.
  New task: **«Есть ли время послезавтра в 21:00?»**. Details contain the prior
  task, incomplete summary, and all 16 prior caller/assistant final utterances.
- 07:18:37, seq 130: assistant asks availability for ten at the new time.
  Seq 132: restaurant replies **«Да есть.»** This establishes availability only.
- Seq 134/164–165: assistant asks permission to book under Армен; owner answers
  **«Оформить на имя Армен»**.
- Seq 167: assistant unnecessarily asks which name to use, speculating that a
  full name might be needed. The restaurant never requested a full name.
  Seq 197–198: owner answers **«Указать только Армен»**.
- 07:19:25: journal confirms a staged outcome. Seq 222 is only a goodbye.
  No booking request/name was spoken after either owner answer, and the restaurant
  made no further utterance. Seq 224 nevertheless reports `agreed`,
  **«Бронь оформлена на имя Армен»**, with no further action needed: unsupported.

The raw session JSON retains `created` and null recording fields; the API derives
current status from events (`EventStore.projected_session`). Both streams end in
`call.ended` and `recording.ready`; raw session fields are not proof of active calls.

## Root causes and minimal changes

1. `booking_close_guard.py`'s anchored positive-answer grammar omitted
   «получится» and repeated «да» in that construction. Added those tokens without
   accepting availability-only, negative, garbled, or conditional replies.
2. `resolve_call_config` scoped the guard using only `task`. The follow-up's
   booking context is in `task_details`, so all booking closure checks were off.
   Include those details in the existing persona-scoped detector.
3. The guard looked only at the latest callee text; it had no invalidation after
   owner permission. An earlier «Да» could therefore validate unsent new details.
   A completed owner clarification now invalidates the staged result and requires
   a subsequent real callee STT final before an agreed outcome. Synthetic owner
   `user_input` and empty STT cannot release this requirement. Existing reply
   checks still run, and explicit callee hangup can still end with an honest
   incomplete result.
4. Owner continuation said only to apply the answer, while general policy said
   to ask again if information was insufficient. It did not explicitly distinguish
   a supplied booking name from a hypothetical full-name requirement. Clarified
   continuation instructions: use supplied data, ask only for necessary missing
   information, pass the permitted decision to the callee, and do not equate owner
   permission with completed action. One question and no repeat confirmation of
   unchanged accepted conditions remain explicit.

Inspected persona/global prompts, owner tool/policy, finalization/end-call guards,
farewell timer, transcript wiring, API dispatch/context construction, and git history.
Only two commits exist: initial import `711a5cd` and `5def769` at 07:17, which changes
only UI free-text owner answers. The latter contains no agent/prompt/guard change;
there is no evidence it caused these failures. Worker logs show owner continuations
queued promptly, not lost replies. The relevant model route was `gpt-5.6-luna`.

## TDD and validation

Before production edits, the booking tests produced **3 expected assertion failures**:
follow-up guard disabled, positive reply rejected, stale callee yes accepted after
owner permission. A separate continuation instruction test also failed before the
fix. An initial test import typo was corrected and the three assertion failures
were rerun before changing production code.

Targeted closure/owner/SDK/hangup tests initially passed **42/42** after the fix.
Expanded tests cover the exact follow-up availability reply, negative/conditional
variants, staged-result invalidation, synthetic/empty input, blocked premature
hangup, and successful closure on one subsequent callee acceptance.

Final checks:

- `PYTHONPATH=agent agent/venv/bin/python -m pytest agent/tests/test_booking_close_guard.py agent/tests/test_universal_owner_tool.py -q`: **38 passed**.
- `PYTHONPATH=agent agent/venv/bin/python -m pytest agent/tests -q`: **193 passed, 4 baseline failures**.
- From `demo-api`, `.venv/bin/python -m pytest -q`: **28 passed**.
- `git diff --check`: passed.

The unchanged HEAD full agent suite, run from a temporary `git archive` export, gives **185 passed,
4 failed**. The four baseline failures concern mixed persona model routes, the
Bito voice, Bito greeting wording, and the personal-assistant greeting's «дословно».
They are unrelated to this patch and were not modified.

Limitations: deterministic tests verify guard behavior and instruction delivery,
not stochastic live-model compliance. The supplied-name improvement is prompt
guidance. The existing guard remains a conservative Russian heuristic, not a
complete reservation state machine; it does not validate every summary detail.
No historical outcome was relabeled as success, and the honest incomplete fallback
remains available if a call ends without a valid structured result.


Exact changed files:

- `agent/booking_close_guard.py`
- `agent/gen2b_agent.py`
- `agent/tests/test_booking_close_guard.py`
- `agent/tests/test_universal_owner_tool.py`
- `agent/artifacts/production-dialogue-2026-09-09.md` (this report)
