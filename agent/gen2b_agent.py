#!/usr/bin/env python3
"""
Gen2B LiveKit voice agent using the official AgentSession framework.

Persona-driven: the dispatch metadata (JSON) selects which persona from
personas.json to run and who the call target is. Falls back to a default
persona if no metadata is provided (so `dev`/manual dispatch still works).

Uses:
- Krisp VIVA plugin (ML-team reference implementation):
  * KrispVivaFilterFrameProcessor — noise cancellation (vi-tel-v2), applied on
    the room audio input BEFORE VAD/STT
  * KrispVivaVAD (vad-v2) wrapped in FanoutVAD — one VAD pass shared by the
    non-streaming STT StreamAdapter and the turn-detection pipeline
  * KrispVivaTurnDetector (tp-v3) — audio-only end-of-turn detection wired
    through TurnHandlingOptions instead of hacking the VAD silence window
  * KrispVivaInterruptDetector (ip-v1) — distinguishes a real interruption
    from background noise / backchannel, drives session.interrupt(force=True)
  * turn_rescue patches — stop user turns from being silently dropped when
    they land during an uninterruptible agent reply
- openai.LLM plugin pointed at ai-kz.gen2b.ai (LiteLLM proxy -> Gemini)
- openai.STT plugin pointed at gen2asr (Gen2B's own STT)
- openai.TTS plugin pointed at gen2tts (Gen2B's own TTS, persona-selected voice)
"""
import asyncio
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import time
import uuid
from collections.abc import AsyncIterable, Callable
from pathlib import Path
from typing import Any

from call_control import (
    hang_up_sip_participant,
    start_call_recording,
    stop_call_recording,
    write_call_outcome,
)
from livekit import agents, rtc
from livekit.agents import (
    AgentSession,
    Agent,
    EndpointingOptions,
    InterruptionOptions,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents.voice import room_io
from livekit.agents.utils.participant import wait_for_participant_attribute
from livekit.plugins import openai
from krisp import (
    FanoutVAD,
    KrispVivaFilterFrameProcessor,
    KrispVivaInterruptDetector,
    KrispVivaTurnDetector,
    KrispVivaVAD,
    attach_backchannel_gate,
    mark_no_rescue,
    patch_false_interruption_min_timeout,
    patch_stale_turn_commit,
    patch_uninterruptible_turn_drop,
    set_keep_all_user_speech,
)
from live_callback import create_request, pop_instruction, wait_for_response
from spoken_ack import normalize_personal_assistant_text
from owner_clarification import OWNER_CLARIFICATION_POLICY, decision_result
from booking_close_guard import booking_close_reason, callee_requested_hangup
from farewell_hangup import FarewellHangupController
from demo_events import emit_demo_event
from livekit.agents.llm import ToolFlag

# Framework-level patches from the ML-team plugin. Must run once at import
# time, before any AgentSession is constructed:
# - uninterruptible_turn_drop: a user turn committed while an uninterruptible
#   agent reply is playing was silently destroyed by livekit-agents. This is a
#   prime suspect for "the caller spoke and the bot just ignored it".
# - stale_turn_commit / false_interruption_min_timeout: guards against a lost
#   STT final and against resuming too eagerly after a false interruption.
patch_uninterruptible_turn_drop()
patch_false_interruption_min_timeout()
patch_stale_turn_commit()


async def wait_for_sip_call_active(room: Any, identity: str, *, waiter=wait_for_participant_attribute) -> None:
    """Do not generate outbound greeting into ringback/early media."""
    participant = room.remote_participants.get(identity)
    if participant is None:
        return
    if participant.attributes.get("sip.callStatus") == "active":
        return
    await waiter(
        room,
        identity=identity,
        attribute="sip.callStatus",
        value="active",
    )

# gen2b/gen2tts is a plain REST endpoint that returns a full WAV body (like tts-1),
# not an SSE/token-streamed model like gpt-4o-mini-tts. Register it so the OpenAI
# plugin picks the AudioChunkedStream path instead of SSEChunkedStream.
from livekit.plugins.openai import tts as _openai_tts
_openai_tts.AUDIO_STREAM_MODELS.add("gen2b/gen2tts")
_openai_tts.AUDIO_STREAM_MODELS.add("gen2tts")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gen2b-agent")

# Make question punctuation sufficiently emphatic for Gen2TTS to retain Russian
# interrogative intonation. Keep an unfinished trailing run between streamed LLM
# chunks so `?` + `!` is normalized once, without buffering ordinary text.
_TTS_PUNCT_TAIL_RE = re.compile(r"[?!]+$")
_TTS_PUNCT_RUN_RE = re.compile(r"[?!]+")


async def adjust_tts_question_punctuation(text: AsyncIterable[str]) -> AsyncIterable[str]:
    pending = ""
    async for chunk in text:
        if not chunk:
            continue
        combined = pending + chunk
        match = _TTS_PUNCT_TAIL_RE.search(combined)
        if match:
            head, pending = combined[:match.start()], combined[match.start():]
        else:
            head, pending = combined, ""
        if head:
            yield _TTS_PUNCT_RUN_RE.sub("!!!?", head)
    if pending:
        yield _TTS_PUNCT_RUN_RE.sub("!!!?", pending)


async def resume_owner_turn(session: AgentSession, decision: dict[str, str]) -> None:
    """Resume a call after ask_owner has returned, outside the completed tool turn."""
    try:
        await session.interrupt(force=True)
    except Exception as exc:
        logger.warning("OWNER_QUESTION continuation interrupt failed type=%s", type(exc).__name__)
    await asyncio.sleep(0)
    session.generate_reply(
        user_input=(
            "Армен только что ответил на уточнение. "
            f"Вопрос: {decision['question']}\n"
            f"Ответ Армена: {decision['response']}\n"
            f"Правило: {decision['instruction']}"
        ),
        instructions=(
            "Немедленно продолжи текущий телефонный разговор, применив ответ Армена. "
            "Не жди новой реплики собеседника."
        ),
        allow_interruptions=False,
    )


async def apply_live_instruction(session: AgentSession, instruction: str) -> None:
    """Inject a fresh owner instruction into the same active conversation."""
    try:
        await session.interrupt(force=True)
    except Exception as exc:
        logger.warning("OWNER_INSTRUCTION interrupt failed type=%s", type(exc).__name__)
    await asyncio.sleep(0)
    session.generate_reply(
        user_input=(
            "Новое поручение владельца во время текущего звонка: "
            f"{instruction}\nПродолжи этот же разговор и сразу выполни поручение. "
            "Не упоминай интерфейс или сообщение владельца."
        ),
        instructions=(
            "Задай собеседнику сам вопрос сразу, естественно и без предисловия. "
            "Не говори, что ты уточнил, узнал или получил новое поручение. "
            "Не проси подождать и не упоминай Армена, владельца, интерфейс или сообщение. "
            "Не жди новой реплики собеседника."
        ),
        allow_interruptions=False,
    )


# LLM may use a dedicated OpenAI-compatible provider.  Audio continues to use
# GEN2B_BASE/GEN2B_KEY when its endpoints are routed through the Gen2B gateway.
GEN2B_BASE = os.environ.get("GEN2B_BASE", "https://ai-kz.gen2b.ai/v1")
GEN2B_KEY = os.environ.get("GEN2B_KEY", "")
GEN2B_LLM_BASE = os.environ.get("GEN2B_LLM_BASE", GEN2B_BASE)
GEN2B_LLM_KEY = os.environ.get("GEN2B_LLM_KEY", GEN2B_KEY)
GEN2B_LLM_MODEL = os.environ.get("GEN2B_LLM_MODEL", "gemini/gemini-2.5-flash")
# Voice replies prioritise first-audio latency; Terra must not spend tokens on
# hidden reasoning before emitting the streamed answer.
GEN2B_LLM_REASONING_EFFORT = os.environ.get("GEN2B_LLM_REASONING_EFFORT", "none")

# STT/TTS now hit Gen2B's own STT/TTS boxes directly over NetBird VPN
# (100.104.x.x), bypassing the LiteLLM proxy.
GEN2B_STT_BASE = os.environ.get("GEN2B_STT_BASE", "http://100.104.4.1:8093/v1")
_GEN2B_DIRECT_STT_KEY = os.environ.get("GEN2B_STT_KEY", "")
_GEN2B_GATEWAY_KEY = os.environ.get("GEN2B_KEY", "")
GEN2B_STT_KEY = (
    _GEN2B_GATEWAY_KEY or _GEN2B_DIRECT_STT_KEY
    if "ai-kz.gen2b.ai" in GEN2B_STT_BASE
    else _GEN2B_DIRECT_STT_KEY
)
GEN2B_STT_MODEL = os.environ.get("GEN2B_STT_MODEL", "gen2asr")
GROQ_STT_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_STT_BASE = os.environ.get("GROQ_STT_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")

GEN2B_TTS_BASE = os.environ.get("GEN2B_TTS_BASE", "http://100.104.4.1:8001/v1")
# Durable clone profiles live on Gen2TTS itself, not behind the gateway.
GEN2B_DIRECT_TTS_BASE = os.environ.get("GEN2B_DIRECT_TTS_BASE", "http://100.104.4.1:8001/v1")
_GEN2B_DIRECT_TTS_KEY = os.environ.get("GEN2B_TTS_KEY", "")
GEN2B_TTS_KEY = (
    _GEN2B_GATEWAY_KEY or _GEN2B_DIRECT_TTS_KEY
    if "ai-kz.gen2b.ai" in GEN2B_TTS_BASE
    else _GEN2B_DIRECT_TTS_KEY
)
GEN2B_TTS_MODEL = os.environ.get("GEN2B_TTS_MODEL", "gen2tts")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
CALL_RESULTS_DIR = Path(os.environ.get("CALL_RESULTS_DIR", "/app/call-results"))
CALL_RECORDINGS_DIR = Path(os.environ.get("CALL_RECORDINGS_DIR", "/app/call-recordings"))
PERSONAS_PATH = Path(os.environ.get("PERSONAS_PATH", Path(__file__).parent / "personas.json"))
GLOBAL_TONE_PATH = Path(
    os.environ.get("GLOBAL_TONE_PATH", Path(__file__).parent / "global_tone.md")
)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
LIVE_CALLBACK_DIR = Path(os.environ.get("LIVE_CALLBACK_DIR", "/home/ubuntu/livekit-stack/live-callbacks"))
LIVE_CALLBACK_TIMEOUT = float(os.environ.get("LIVE_CALLBACK_TIMEOUT", "180"))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


KRISP_ENABLED = _env_bool("KRISP_ENABLED", True)
KRISP_MODELS_DIR = Path(
    os.environ.get("KRISP_MODELS_DIR", Path(__file__).parent / "models" / "krisp")
)
KRISP_SUPPRESSION_LEVEL = int(os.environ.get("KRISP_SUPPRESSION_LEVEL", "100"))

# Telephony audio format. Krisp's tel models are trained for 8 kHz/10 ms frames,
# and the SIP leg is 8 kHz anyway, so this avoids a pointless resample.
KRISP_SAMPLE_RATE = int(os.environ.get("KRISP_SAMPLE_RATE", "8000"))
KRISP_FRAME_DURATION_MS = int(os.environ.get("KRISP_FRAME_DURATION_MS", "10"))

# The plugin resolves each model from its own env var. Default them to the
# local models dir so the service works without extra .env plumbing.
KRISP_MODEL_ENV_DEFAULTS = {
    "KRISP_VIVA_FILTER_MODEL_PATH": "krisp-viva-vi-tel-v2.kef",
    "KRISP_VIVA_VAD_MODEL_PATH": "krisp-viva-vad-v2.kef",
    "KRISP_VIVA_EOU_MODEL_PATH": "krisp-viva-tp-v3.kef",
    "KRISP_VIVA_INTERRUPT_MODEL_PATH": "krisp-viva-ip-v1.kef",
}
for _env_name, _model_file in KRISP_MODEL_ENV_DEFAULTS.items():
    os.environ.setdefault(_env_name, str(KRISP_MODELS_DIR / _model_file))

# --- VAD (speech vs silence only; turn-taking is a separate model now) ---
KRISP_ACTIVATION_THRESHOLD = float(os.environ.get("KRISP_ACTIVATION_THRESHOLD", "0.50"))
KRISP_MIN_SILENCE_DURATION = float(os.environ.get("KRISP_MIN_SILENCE_DURATION", "0.50"))
KRISP_MIN_SPEECH_DURATION = float(os.environ.get("KRISP_MIN_SPEECH_DURATION", "0.10"))
KRISP_PREFIX_PADDING_DURATION = float(
    os.environ.get("KRISP_PREFIX_PADDING_DURATION", "0.30")
)

# --- Turn detection (tp-v3) ---
# predict_end_of_turn() result is compared against this threshold: below it the
# caller is considered done (commit after min_delay), at or above it the caller
# is still mid-thought (wait up to max_delay). Unlike the old implementation,
# an uncertain TP now makes the agent wait LONGER rather than cut in sooner.
KRISP_UNLIKELY_EOT_THRESHOLD = float(
    os.environ.get("KRISP_UNLIKELY_EOT_THRESHOLD", "0.3")
)
KRISP_ENDPOINTING_MIN_DELAY = float(os.environ.get("KRISP_ENDPOINTING_MIN_DELAY", "0.1"))
KRISP_ENDPOINTING_MAX_DELAY = float(os.environ.get("KRISP_ENDPOINTING_MAX_DELAY", "8.0"))

# --- Interruption detection (ip-v1) ---
# Native VAD/word-count interruption is disabled via unreachable thresholds so
# that ONLY the Krisp IP model can cut the agent off mid-sentence. That is what
# keeps loud background noise/music from being mistaken for the caller talking.
KRISP_INTERRUPT_THRESHOLD = float(os.environ.get("KRISP_INTERRUPT_THRESHOLD", "0.5"))
KRISP_AEC_WARMUP_DURATION = float(os.environ.get("KRISP_AEC_WARMUP_DURATION", "0.5"))
KRISP_FALSE_INTERRUPTION_TIMEOUT = float(
    os.environ.get("KRISP_FALSE_INTERRUPTION_TIMEOUT", "2.0")
)
# Keep every caller utterance: turns spoken over a protected reply are stored
# (and answered afterwards) instead of being dropped.
KEEP_ALL_USER_SPEECH = _env_bool("KEEP_ALL_USER_SPEECH", True)
# Debug: when set, the noise-cancellation processor dumps before/after audio
# per call as do_<ts>.ogg / posle_<ts>.ogg so the filtering can be verified by
# ear. Requires the `soundfile` package. Leave unset in normal operation.
KRISP_DEBUG_AUDIO_DIR = os.environ.get("KRISP_DEBUG_AUDIO_DIR", "").strip()


def format_call_outcome(outcome: dict[str, Any]) -> str:
    """Create the concise Telegram result delivered after a SIP call is closed."""
    lines = [
        "📞 Итог звонка",
        f"Статус: {outcome['outcome']}",
        f"Итог: {outcome['summary']}",
    ]
    if outcome.get("agreed_price_gbp") is not None:
        lines.append(f"Цена: £{outcome['agreed_price_gbp']:g}")
    elif outcome.get("agreed_price_kzt") is not None:
        lines.append(f"Цена: {outcome['agreed_price_kzt']} ₸/м³")
    if outcome.get("volume_m3") is not None:
        lines.append(f"Объём: {outcome['volume_m3']} м³")
    if outcome.get("next_step"):
        lines.append(f"Следующий шаг: {outcome['next_step']}")
    return "\n".join(lines)


def _wait_for_recording(path: Path, timeout: float = 12.0) -> None:
    """Wait until the egress file exists, is non-empty, and stops growing."""
    deadline = time.monotonic() + timeout
    previous_size = -1
    stable_checks = 0
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = 0
        if size > 0 and size == previous_size:
            stable_checks += 1
            if stable_checks >= 2:
                return
        else:
            stable_checks = 0
        previous_size = size
        time.sleep(0.4)
    raise TimeoutError(f"recording did not become ready: {path}")


async def publish_web_outcome(session_id: str, room_name: str, outcome: dict[str, Any]) -> None:
    """Publish the actionable result immediately; recording readiness follows separately."""
    emit_demo_event(session_id, room_name, "call.outcome", outcome)


def _multipart_audio_body(
    *, chat_id: str, caption: str, recording: Path
) -> tuple[bytes, str]:
    """Build a Telegram sendAudio multipart body without extra dependencies.

    sendAudio (not sendDocument) so the recording renders as an inline audio
    player in the chat instead of a grey file attachment.
    """
    boundary = f"HermesVoiceAgent{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    def field(name: str, value: str) -> None:
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"),
            b"\r\n",
        ])

    field("chat_id", chat_id)
    field("title", "Запись звонка")
    # Telegram limits audio captions to 1024 characters.
    field("caption", caption if len(caption) <= 1024 else caption[:1020] + "…")
    chunks.extend([
        f"--{boundary}\r\n".encode(),
        (
            'Content-Disposition: form-data; name="audio"; '
            f'filename="{recording.name}"\r\n'
        ).encode(),
        b"Content-Type: audio/ogg\r\n\r\n",
        recording.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return b"".join(chunks), boundary


def deliver_outcome_to_telegram(
    outcome: dict[str, Any], recording_path: str | None = None
) -> None:
    """Push outcome and recording directly to Telegram after call teardown.

    Raises on any failure so the caller can retain the durable cron queue as a
    fallback instead of silently losing either the summary or audio file.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not configured")

    caption = format_call_outcome(outcome)
    if recording_path:
        recording = Path(recording_path)
        _wait_for_recording(recording)
        body, boundary = _multipart_audio_body(
            chat_id=TELEGRAM_CHAT_ID,
            caption=caption,
            recording=recording,
        )
        endpoint = "sendAudio"
        content_type = f"multipart/form-data; boundary={boundary}"
    else:
        body = urllib.parse.urlencode({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": caption,
        }).encode()
        endpoint = "sendMessage"
        content_type = "application/x-www-form-urlencoded"

    request = urllib.request.Request(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{endpoint}",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Telegram {endpoint} returned HTTP {response.status}")


def send_live_callback_to_telegram(request_id: str, question: str) -> int:
    """Ask the owner for an in-call decision using Telegram buttons."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not configured")
    reply_markup = json.dumps({
        "inline_keyboard": [[
            {"text": "✅ Да", "callback_data": f"vc:{request_id}:yes"},
            {"text": "❌ Отказаться", "callback_data": f"vc:{request_id}:no"},
            {"text": "✏️ Другое", "callback_data": f"vc:{request_id}:other"},
        ]]
    }, ensure_ascii=False)
    body = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"📞 Уточнение во время звонка\n\n{question}",
        "reply_markup": reply_markup,
    }).encode()
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    # Callback delivery is part of a live call. Fail quickly instead of holding
    # the restaurant conversation for the generic 30-second Telegram timeout.
    with urllib.request.urlopen(request, timeout=10) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Telegram sendMessage returned HTTP {response.status}")
        payload = json.loads(response.read())
        if not payload.get("ok") or not payload.get("result", {}).get("message_id"):
            raise RuntimeError("Telegram did not confirm message delivery")
        return int(payload["result"]["message_id"])


DEFAULT_PERSONA_KEY = "angry_grandpa"
DEFAULT_TARGET_NAME = "собеседник"
DEFAULT_TARGET_PRONOUN_NOM = "он"
DEFAULT_TARGET_PRONOUN_ACC = "его"


def load_personas() -> dict[str, dict[str, Any]]:
    with open(PERSONAS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_global_tone() -> str:
    """Load the shared spoken-style layer applied to every persona."""
    try:
        return GLOBAL_TONE_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        logger.exception("could not load global tone instructions from %s", GLOBAL_TONE_PATH)
        return ""


def resolve_call_config(metadata_raw: str) -> dict[str, Any]:
    """Parse dispatch metadata JSON and merge with the selected persona.

    Expected metadata shape (all optional, sensible defaults applied):
    {
      "persona": "angry_grandpa",
      "target_name": "Армен",
      "target_pronoun_nom": "он",
      "target_pronoun_acc": "его",
      "target_identity": "armen",
      "task": "забронировать стол на двоих завтра в 20:00",
      "task_details": "Бронь на имя Армен Атаян"
    }
    """
    personas = load_personas()

    parsed: dict[str, Any] = {}
    if metadata_raw:
        try:
            parsed = json.loads(metadata_raw)
        except json.JSONDecodeError:
            logger.warning("could not parse job metadata as JSON: %r", metadata_raw)

    persona_key = parsed.get("persona", DEFAULT_PERSONA_KEY)
    persona = personas.get(persona_key)
    if persona is None:
        logger.warning("unknown persona %r, falling back to %r", persona_key, DEFAULT_PERSONA_KEY)
        persona_key = DEFAULT_PERSONA_KEY
        persona = personas[DEFAULT_PERSONA_KEY]

    target_name = parsed.get("target_name", DEFAULT_TARGET_NAME)
    target_pronoun_nom = parsed.get("target_pronoun_nom", DEFAULT_TARGET_PRONOUN_NOM)
    target_pronoun_acc = parsed.get("target_pronoun_acc", DEFAULT_TARGET_PRONOUN_ACC)
    target_identity = parsed.get("target_identity", "callee")
    task = str(parsed.get("task", "")).strip()
    task_details = str(parsed.get("task_details", "")).strip()

    fmt_kwargs = {
        "target_name": target_name,
        "target_pronoun_nom": target_pronoun_nom,
        "target_pronoun_acc": target_pronoun_acc,
        "task": task,
        "task_details": task_details or "Дополнительных условий нет.",
    }

    prompt_path = persona.get("system_prompt_path")
    if prompt_path:
        persona_prompt = Path(prompt_path).read_text(encoding="utf-8").strip().format(**fmt_kwargs)
    else:
        persona_prompt = persona["system_prompt"].format(**fmt_kwargs)

    if persona.get("exact_system_prompt", False):
        system_prompt = persona_prompt
    else:
        global_tone = load_global_tone()
        system_prompt = (
            f"{persona_prompt}\n\n{global_tone}" if global_tone else persona_prompt
        )
        system_prompt += f"\n\n{OWNER_CLARIFICATION_POLICY}"
    greeting_instructions = persona["greeting_instructions"].format(**fmt_kwargs)

    return {
        "persona_key": persona_key,
        "persona_name": persona["name"],
        "voice": persona["voice"],
        "llm_model": persona.get("llm_model", GEN2B_LLM_MODEL),
        "llm_route": persona.get("llm_route", "default"),
        "stt_language": persona.get("stt_language", "kk_ru_iso"),
        "tts_intonation": bool(persona.get("tts_intonation", True)),
        "wait_for_user_first": bool(persona.get("wait_for_user_first", False)),
        "booking_confirmation_required": persona_key == "armen_personal_assistant" and bool(
            re.search(r"брон|резерв|запис|запиш|назнач", task, re.I)
        ),
        "system_prompt": system_prompt,
        "greeting_instructions": greeting_instructions,
        "target_identity": target_identity,
        "target_name": target_name,
        "task": task,
        "task_details": task_details,
        "origin": str(parsed.get("origin", "telegram")),
        "owner_channel": str(parsed.get("owner_channel", "telegram")),
        "result_channel": str(parsed.get("result_channel", "telegram")),
        "demo_session_id": str(parsed.get("demo_session_id", "")),
    }


class DemoTranscriptEmitter:
    """Translate LiveKit transcript events into the public demo event contract."""

    def __init__(self, session_id: str, room_name: str):
        self.session_id = session_id
        self.room_name = room_name
        self._fallback_utterance_id = ""

    def on_user_transcript(self, event: object) -> None:
        text = str(getattr(event, "transcript", "")).strip()
        if not text:
            return
        item_id = str(getattr(event, "item_id", "") or "")
        if item_id:
            utterance_id = item_id
        else:
            if not self._fallback_utterance_id:
                self._fallback_utterance_id = uuid.uuid4().hex[:12]
            utterance_id = self._fallback_utterance_id
        is_final = bool(getattr(event, "is_final", False))
        event_type = "transcript.caller.final" if is_final else "transcript.caller.partial"
        emit_demo_event(self.session_id, self.room_name, event_type, {
            "utterance_id": utterance_id,
            "text": text,
        })
        if is_final:
            self._fallback_utterance_id = ""


def infer_owner_options(question: str, context: str = "") -> list[str]:
    """Extract 2–4 explicit alternatives without domain-specific option lists."""
    sentences = re.split(r"[.!?]\s*", question)
    colon_match = re.search(r":\s*([^:?.!]+?)(?:[?.!]|$)", question)
    if colon_match:
        listed = [part.strip() for part in re.split(r"\s*,\s*|\s+или\s+", colon_match.group(1), flags=re.IGNORECASE) if part.strip()]
        if 2 <= len(listed) <= 4 and all(1 <= len(part.split()) <= 4 for part in listed):
            return [part[0].upper() + part[1:] for part in listed]
    patterns = (
        r"(?:выбрать|выбираете|предпочитаете)\s*:?\s*([^,:;?]+?)\s+или\s+([^,:;?]+)$",
        r"(?:есть|доступны|предлага(?:ет|ют))\s+(?:только\s+)?([^,:;?]+?)\s+(?:или|и)\s+([^,:;?]+)$",
    )
    for sentence in reversed(sentences):
        value = sentence.strip()
        candidates: list[list[str] | tuple[str, str]] = []
        if ":" in value:
            colon_tail = value.rsplit(":", 1)[-1].strip()
            listed = [part.strip() for part in re.split(r"\s*,\s*|\s+или\s+", colon_tail, flags=re.IGNORECASE) if part.strip()]
            if 2 <= len(listed) <= 4:
                candidates.append(listed)
            simple = re.fullmatch(r"([^,:;?]+?)\s+или\s+([^,:;?]+)", colon_tail, re.IGNORECASE)
            if simple:
                candidates.append(simple.groups())
        for pattern in patterns:
            match = re.search(pattern, value, re.IGNORECASE)
            if match:
                candidates.append(match.groups())
        for groups in candidates:
            options = []
            for raw in groups:
                option = re.sub(r"^(?:вам\s+|для вас\s+)", "", raw.strip(), flags=re.IGNORECASE)
                option = option.strip(" —-,:;")
                if 1 <= len(option.split()) <= 4:
                    options.append(option[0].upper() + option[1:])
            if len(options) == 2 and options[0].lower() != options[1].lower():
                return options
    return []


class Gen2BAssistant(Agent):
    def __init__(
        self,
        room_name: str,
        participant_identity: str,
        on_outcome_saved: Callable[[], None],
        system_prompt: str,
        recording_path_getter: Callable[[], str | None],
        stop_recording: Callable[[], "asyncio.Future"] | None = None,
        personal_assistant: bool = False,
        booking_confirmation_required: bool = False,
        owner_channel: str = "telegram",
        result_channel: str = "telegram",
        demo_session_id: str = "",
        transcript_emitter: DemoTranscriptEmitter | None = None,
        tts_intonation: bool = True,
    ):
        self._room_name = room_name
        self._participant_identity = participant_identity
        self._on_outcome_saved = on_outcome_saved
        self._recording_path_getter = recording_path_getter
        self._stop_recording = stop_recording
        self._personal_assistant = personal_assistant
        self._booking_confirmation_required = booking_confirmation_required
        self._owner_channel = owner_channel
        self._result_channel = result_channel
        self._demo_session_id = demo_session_id
        self._transcript_emitter = transcript_emitter
        self._tts_intonation = tts_intonation
        self._latest_user_text: str | None = None
        self._pending_outcome: dict[str, Any] | None = None
        self._web_outcome_published = False
        self._web_ended_published = False
        self._hangup_started = False
        self._owner_continuation_task: asyncio.Task[None] | None = None
        super().__init__(instructions=system_prompt)

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: agents.ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Normalize streamed ?/! runs and preserve raw generation evidence."""
        raw_chunks: list[str] = []
        tts_chunks: list[str] = []
        utterance_id = uuid.uuid4().hex[:12]

        async def capture_effective_text() -> AsyncIterable[str]:
            effective_text: AsyncIterable[str] = text
            if self._personal_assistant:
                effective_text = normalize_personal_assistant_text(effective_text)
            async for chunk in effective_text:
                raw_chunks.append(chunk)
                if self._transcript_emitter and chunk:
                    emit_demo_event(
                        self._transcript_emitter.session_id,
                        self._room_name,
                        "transcript.assistant.delta",
                        {"utterance_id": utterance_id, "text": chunk},
                    )
                yield chunk

        async def capture_tts_text() -> AsyncIterable[str]:
            effective_text: AsyncIterable[str] = capture_effective_text()
            if getattr(self, "_tts_intonation", True):
                effective_text = adjust_tts_question_punctuation(effective_text)
            async for chunk in effective_text:
                tts_chunks.append(chunk)
                logger.info("RAW_TTS_CHUNK room=%s text=%r", self._room_name, chunk)
                yield chunk

        try:
            async for frame in Agent.default.tts_node(self, capture_tts_text(), model_settings):
                yield frame
        finally:
            if raw_chunks:
                logger.info("RAW_LLM_TO_TTS room=%s text=%r", self._room_name, "".join(raw_chunks))
            if tts_chunks:
                logger.info("RAW_TTS_INPUT room=%s text=%r", self._room_name, "".join(tts_chunks))
            if self._transcript_emitter and raw_chunks:
                emit_demo_event(
                    self._transcript_emitter.session_id,
                    self._room_name,
                    "transcript.assistant.final",
                    {"utterance_id": utterance_id, "text": "".join(raw_chunks)},
                )

    def _booking_close_block(self, outcome: str) -> str | None:
        if not getattr(self, "_booking_confirmation_required", False):
            return None
        text = self._latest_user_text
        if text is None:
            text = next((item.text_content for item in reversed(self.chat_ctx.items)
                         if getattr(item, "role", None) == "user"), "")
        return booking_close_reason(text or "", outcome)

    @function_tool(flags=ToolFlag.CANCELLABLE)
    async def ask_owner(
        self,
        ctx: RunContext,
        question: str,
        context: str = "",
        options: list[str] | None = None,
    ) -> str:
        """Универсальное уточнение у Армена в Telegram в текущем звонке.

        Используй для любого решения или недостающих сведений Армена: дата,
        адрес, выбор, условия, разрешение, изменение поручения, а не только бронь.
        Сам произнесёт фразу ожидания и отправит кнопки; не обещай вместо вызова.
        Не завершай звонок и не соглашайся сам, пока ждёшь. Вернёт JSON с решением.

        Args:
            question: Один конкретный вопрос, организация и точный вариант для согласия.
            context: Исходное поручение, известные условия и причина уточнения, без секретов.
            options: 2–4 коротких конкретных варианта ответа для текущего вопроса.
                Например для выбора зала: ["Обычный зал", "VIP-зал"]. Не добавляй
                сюда «Другое» — интерфейс добавляет эту кнопку автоматически.
        """
        question, context = question.strip(), context.strip()
        clean_options: list[str] = []
        for option in options or infer_owner_options(question, context):
            value = str(option).strip()
            if value and len(value) <= 80 and value not in clean_options:
                clean_options.append(value)
        clean_options = clean_options[:4]
        if not question or len(question) > 1400 or len(context) > 1600:
            return decision_result(question[:1400], context[:1600], "delivery_failed")
        if getattr(self, "_owner_question_task", None) is not None:
            return decision_result(question, context, "busy")
        self._owner_question_task = asyncio.current_task()
        request_id = ""
        logger.info("OWNER_QUESTION entered room=%s", self._room_name)
        try:
            # Scheduling audio must not gate Telegram delivery: a stuck playout
            # would otherwise prevent the send path from being reached.
            try:
                ctx.session.say("Секундочку, сейчас уточню.", allow_interruptions=False)
            except Exception as exc:
                logger.warning("OWNER_QUESTION speech failed type=%s", type(exc).__name__)
            display = question + (f"\n\nКонтекст: {context}" if context else "")
            request_id = create_request(
                LIVE_CALLBACK_DIR, room_name=self._room_name,
                chat_id=(
                    getattr(self, "_demo_session_id", "")
                    if getattr(self, "_owner_channel", "telegram") == "web"
                    else TELEGRAM_CHAT_ID
                ),
                question=display,
            )
            if getattr(self, "_owner_channel", "telegram") == "web":
                emit_demo_event(getattr(self, "_demo_session_id", ""), self._room_name, "owner.question", {
                    "request_id": request_id, "question": question, "context": context,
                    "options": clean_options,
                })
                logger.info("OWNER_QUESTION published to web room=%s request=%s", self._room_name, request_id)
            else:
                message_id = await asyncio.to_thread(send_live_callback_to_telegram, request_id, display)
                logger.info("OWNER_QUESTION sent room=%s request=%s message_id=%s",
                            self._room_name, request_id, message_id)
            answer = await wait_for_response(
                LIVE_CALLBACK_DIR, request_id, timeout=LIVE_CALLBACK_TIMEOUT,
                poll_interval=min(0.05, LIVE_CALLBACK_TIMEOUT / 2),
            )
            result = decision_result(question, context, answer, request_id)
            decision = json.loads(result)
            logger.info("OWNER_QUESTION resolved room=%s request=%s action=%s",
                        self._room_name, request_id, decision["action"])
            if getattr(self, "_owner_channel", "telegram") == "web":
                emit_demo_event(getattr(self, "_demo_session_id", ""), self._room_name, "owner.answer", {
                    "request_id": request_id,
                    "answer": answer,
                    "action": decision["action"],
                })
            # Run after this function-tool returns. Calling generate_reply inside
            # the tool left the reply queued behind the tool's uninterruptible
            # waiting speech, so the restaurant had to speak again to revive it.
            task = asyncio.create_task(resume_owner_turn(ctx.session, decision))
            self._owner_continuation_task = task
            logger.info("OWNER_QUESTION continuation queued room=%s request=%s",
                        self._room_name, request_id)
            return result
        except asyncio.CancelledError:
            logger.info("OWNER_QUESTION cancelled room=%s request=%s", self._room_name, request_id)
            raise
        except Exception as exc:
            logger.error("OWNER_QUESTION delivery failed room=%s request=%s type=%s",
                         self._room_name, request_id, type(exc).__name__)
            return decision_result(question, context, "delivery_failed", request_id)
        finally:
            if request_id:
                (LIVE_CALLBACK_DIR / "pending" / f"{request_id}.json").unlink(missing_ok=True)
                (LIVE_CALLBACK_DIR / "responses" / f"{request_id}.txt").unlink(missing_ok=True)
            self._owner_question_task = None

    @function_tool
    async def finalize_call(
        self,
        ctx: RunContext,
        outcome: str,
        summary: str,
        agreed_price_kzt: int | None = None,
        agreed_price_gbp: float | None = None,
        volume_m3: int | None = None,
        next_step: str = "",
    ) -> str:
        """Зафиксировать итог звонка перед завершением. Доставлен в Telegram будет
        только после фактического завершения звонка (после end_call)."""
        reason = self._booking_close_block(outcome)
        if reason:
            logger.warning("FINALIZE_BLOCKED room=%s reason=%s", self._room_name, reason)
            return reason
        self._pending_outcome = {
            "outcome": outcome,
            "summary": summary,
            "agreed_price_kzt": agreed_price_kzt,
            "agreed_price_gbp": agreed_price_gbp,
            "volume_m3": volume_m3,
            "next_step": next_step,
        }
        logger.info("staged call outcome for room=%s (will save after hangup)", self._room_name)
        return "Итог зафиксирован, будет доставлен Армену после завершения звонка."

    @function_tool
    async def end_call(self, ctx: RunContext) -> str:
        """Завершить телефонный звонок после финального итога или по просьбе собеседника."""
        if getattr(self, "_hangup_started", False):
            return "Звонок уже завершается."
        latest_user_text = getattr(self, "_latest_user_text", None)
        if latest_user_text is None:
            chat_ctx = getattr(self, "_chat_ctx", None)
            latest_user_text = next((item.text_content for item in reversed(chat_ctx.items)
                                     if getattr(item, "role", None) == "user"), "") if chat_ctx else ""
        explicit_hangup = callee_requested_hangup(latest_user_text or "")
        if explicit_hangup and self._pending_outcome is None:
            self._pending_outcome = {
                "outcome": "incomplete",
                "summary": "Собеседник завершил разговор до фиксации итоговой договорённости.",
                "agreed_price_kzt": None,
                "agreed_price_gbp": None,
                "volume_m3": None,
                "next_step": "Проверить детали разговора и при необходимости перезвонить.",
            }
        if getattr(self, "_booking_confirmation_required", False):
            reason = self._booking_close_block((self._pending_outcome or {}).get("outcome", ""))
            if not explicit_hangup and (reason or self._pending_outcome is None):
                reason = reason or "Не завершай звонок: сначала нужен допустимый итог finalize_call."
                logger.warning("HANGUP_BLOCKED room=%s reason=%s", self._room_name, reason)
                return reason
        self._hangup_started = True
        # Finish the current tool-owning turn first, then play a deterministic
        # farewell. This prevents a tool-only turn from hanging up in silence.
        if not explicit_hangup:
            await ctx.wait_for_playout()
            farewell = ctx.session.say("Спасибо, до свидания.", allow_interruptions=False)
            await farewell.wait_for_playout()
            await asyncio.sleep(3)
        if (
            self._pending_outcome is not None
            and self._result_channel == "web"
            and self._demo_session_id
        ):
            await publish_web_outcome(self._demo_session_id, self._room_name, self._pending_outcome)
            self._web_outcome_published = True
        if self._stop_recording:
            await self._stop_recording()
        await hang_up_sip_participant(self._room_name, self._participant_identity)
        if self._result_channel == "web" and self._demo_session_id:
            emit_demo_event(self._demo_session_id, self._room_name, "call.ended", {"reason": "completed"})
            self._web_ended_published = True
        ctx.session.shutdown(drain=False)
        return "Звонок завершён."


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    logger.info(f"connected to room {ctx.room.name}")

    call_config = resolve_call_config(ctx.job.metadata)
    demo_session_id = str(call_config.get("demo_session_id", ""))
    demo_emitter = (
        DemoTranscriptEmitter(demo_session_id, ctx.room.name)
        if call_config.get("origin") == "web_demo" and demo_session_id
        else None
    )
    if demo_emitter:
        emit_demo_event(demo_session_id, ctx.room.name, "call.connected", {
            "contact": call_config["target_name"],
        })
    logger.info(
        "using persona=%s (%s) for target=%s",
        call_config["persona_key"],
        call_config["persona_name"],
        call_config["target_name"],
    )

    egress_id: str | None = await start_call_recording(ctx.room.name, CALL_RECORDINGS_DIR)
    recording_stopped = False
    if egress_id:
        logger.info("started call recording egress_id=%s for room=%s", egress_id, ctx.room.name)
    else:
        logger.warning("failed to start call recording for room=%s", ctx.room.name)

    def recording_path() -> str | None:
        if not egress_id:
            return None
        host_path = CALL_RECORDINGS_DIR / f"{ctx.room.name.replace('/', '_')}.ogg"
        return str(host_path)

    async def stop_recording_once() -> None:
        nonlocal recording_stopped
        if egress_id and not recording_stopped:
            recording_stopped = True
            await stop_call_recording(egress_id)
            logger.info("stopped call recording egress_id=%s", egress_id)

    transcript: list[str] = []
    outcome_saved = False

    def mark_outcome_saved() -> None:
        nonlocal outcome_saved
        outcome_saved = True

    def record_transcript(event: object) -> None:
        is_final = bool(getattr(event, "is_final", False))
        text = str(getattr(event, "transcript", "")).strip()
        logger.info("[DEBUG-uchqun-stt] final=%s chars=%d text=%r", is_final, len(text), text)
        if is_final:
            transcript.append(text)
            agent._latest_user_text = text
        if demo_emitter:
            demo_emitter.on_user_transcript(event)

    def save_fallback_outcome(*_args: object) -> None:
        if outcome_saved:
            return
        text = " ".join(item for item in transcript if item)
        write_call_outcome(
            results_dir=CALL_RESULTS_DIR,
            room_name=ctx.room.name,
            outcome="incomplete",
            summary="Звонок завершился без структурированного итога.",
            agreed_price_kzt=None,
            agreed_price_gbp=None,
            volume_m3=None,
            next_step=(f"Проверить запись звонка. Распознанная речь: {text}" if text else "Перезвонить и уточнить итог."),
            recording_path=recording_path(),
        )
        logger.warning("saved fallback call outcome for %s", ctx.room.name)
    # --- Krisp VIVA pipeline (ML-team plugin architecture) ---
    # Four independent models with clearly separated jobs, instead of the old
    # home-grown VAD that also tried to do turn-taking by shrinking its own
    # silence window.
    krisp_active = False
    noise_cancellation = None
    turn_detector = None
    interrupt_detector = None
    session_vad: Any = None

    if KRISP_ENABLED:
        try:
            session_vad = FanoutVAD(
                KrispVivaVAD.load(
                    activation_threshold=KRISP_ACTIVATION_THRESHOLD,
                    prefix_padding_duration=KRISP_PREFIX_PADDING_DURATION,
                    min_silence_duration=KRISP_MIN_SILENCE_DURATION,
                    min_speech_duration=KRISP_MIN_SPEECH_DURATION,
                )
            )
            turn_detector = KrispVivaTurnDetector.load(
                unlikely_eot_threshold=KRISP_UNLIKELY_EOT_THRESHOLD,
            )
            interrupt_detector = KrispVivaInterruptDetector.load(
                interrupt_threshold=KRISP_INTERRUPT_THRESHOLD,
            )
            noise_cancellation = KrispVivaFilterFrameProcessor(
                noise_suppression_level=KRISP_SUPPRESSION_LEVEL,
                frame_duration_ms=KRISP_FRAME_DURATION_MS,
                sample_rate=KRISP_SAMPLE_RATE,
                debug_audio_dir=KRISP_DEBUG_AUDIO_DIR or None,
            )
            krisp_active = True
            logger.info(
                "Krisp pipeline: NC=on VAD=fanout(vad-v2) TP=tp-v3(eot<%.2f) IP=ip-v1(thr=%.2f)",
                KRISP_UNLIKELY_EOT_THRESHOLD,
                KRISP_INTERRUPT_THRESHOLD,
            )
        except Exception:
            logger.exception("Krisp initialization failed; falling back to Silero VAD")
            krisp_active = False
            noise_cancellation = None
            turn_detector = None
            interrupt_detector = None
            session_vad = None

    if session_vad is None:
        from livekit.plugins import silero

        session_vad = silero.VAD.load()
        logger.warning("using Silero VAD fallback (Krisp unavailable)")

    pipeline_latency_ms: dict[str, list[float]] = {}

    def add_latency(name: str, seconds: float) -> None:
        if seconds >= 0:
            pipeline_latency_ms.setdefault(name, []).append(seconds * 1000)

    def record_pipeline_metrics(event: object) -> None:
        metric = getattr(event, "metrics", None)
        metric_type = getattr(metric, "type", "")
        if metric_type == "stt_metrics":
            add_latency("stt_request", metric.duration)
        elif metric_type == "llm_metrics":
            add_latency("llm_ttft", metric.ttft)
            add_latency("llm_total", metric.duration)
        elif metric_type == "tts_metrics":
            add_latency("tts_ttfb", metric.ttfb)
            add_latency("tts_total", metric.duration)
        elif metric_type == "eou_metrics":
            add_latency("eou_delay", metric.end_of_utterance_delay)
            add_latency("transcription_delay", metric.transcription_delay)

    stt_language = str(call_config["stt_language"])
    if stt_language.lower().startswith("en"):
        if not GROQ_STT_KEY:
            raise RuntimeError("GROQ_API_KEY is required for English voice-agent STT")
        session_stt = openai.STT(
            model=GROQ_STT_MODEL,
            base_url=GROQ_STT_BASE,
            api_key=GROQ_STT_KEY,
            language="en",
        )
        logger.info("STT provider=groq model=%s language=en", GROQ_STT_MODEL)
    else:
        if not GEN2B_STT_KEY:
            raise RuntimeError("GEN2B_STT_KEY is required for non-English voice-agent STT")
        session_stt = openai.STT(
            model=GEN2B_STT_MODEL,
            base_url=GEN2B_STT_BASE,
            api_key=GEN2B_STT_KEY,
            # Callers can speak Russian while the Uchqun persona replies in
            # Uzbek. Sending a fixed `uz` hint made Gen2 ASR return an empty
            # final transcript for the captured Russian SIP audio.
            detect_language=True,
            language=stt_language,
        )
        logger.info("STT provider=gen2 model=%s language=%s", GEN2B_STT_MODEL, stt_language)

    session_kwargs: dict[str, Any] = {}
    if krisp_active:
        session_kwargs["aec_warmup_duration"] = KRISP_AEC_WARMUP_DURATION
        session_kwargs["turn_handling"] = TurnHandlingOptions(
            # Krisp VAD already sends the caller audio to STT, but the TP
            # end-of-turn handoff was not committing those final transcripts.
            # Commit on the proven VAD speech boundary; Krisp NC and IP remain
            # active in the audio path.
            turn_detection="vad",
            endpointing=EndpointingOptions(
                mode="fixed",
                min_delay=KRISP_ENDPOINTING_MIN_DELAY,
                max_delay=KRISP_ENDPOINTING_MAX_DELAY,
            ),
            interruption=InterruptionOptions(
                # enabled must stay True: it also sets the default
                # allow_interruptions for every reply, and disabling it makes
                # livekit silently discard user turns committed mid-reply.
                enabled=True,
                # Unreachable thresholds neuter the framework's own VAD- and
                # word-count-based barge-in, leaving Krisp IP as the only thing
                # that can interrupt the agent.
                min_duration=9999.0,
                min_words=2,
                discard_audio_if_uninterruptible=False,
                resume_false_interruption=True,
                false_interruption_timeout=KRISP_FALSE_INTERRUPTION_TIMEOUT,
            ),
        )

    # Gateway TTS currently labels raw clone PCM as audio/mpeg, which makes the
    # LiveKit OpenAI decoder emit no frames.  Use the durable-profile service
    # directly for every clone voice; non-clone voices keep the current route.
    tts_base = GEN2B_TTS_BASE
    tts_key = GEN2B_TTS_KEY
    tts_model = GEN2B_TTS_MODEL
    if str(call_config["voice"]).startswith("clone:"):
        if not _GEN2B_DIRECT_TTS_KEY:
            raise RuntimeError("GEN2B_TTS_KEY is required for clone voices")
        tts_base = GEN2B_DIRECT_TTS_BASE
        tts_key = _GEN2B_DIRECT_TTS_KEY
        tts_model = "gen2tts"
        logger.info("TTS provider=gen2-direct model=%s voice=%s", tts_model, call_config["voice"])

    llm_route = call_config.get("llm_route", "default")
    if llm_route == "gateway":
        llm_base = GEN2B_BASE
        llm_key = GEN2B_KEY
    else:
        llm_base = GEN2B_LLM_BASE
        llm_key = GEN2B_LLM_KEY
    logger.info(
        "LLM route=%s model=%s",
        llm_route,
        call_config["llm_model"],
    )

    session = AgentSession(
        vad=session_vad,
        stt=session_stt,
        llm=openai.LLM(
            model=call_config["llm_model"],
            reasoning_effort=GEN2B_LLM_REASONING_EFFORT,
            base_url=llm_base,
            api_key=llm_key,
        ),
        tts=openai.TTS(
            model=tts_model,
            voice=call_config["voice"],
            speed=float(call_config.get("tts_speed", 1.0)),
            base_url=tts_base,
            api_key=tts_key,
            response_format="pcm",
        ),
        **session_kwargs,
    )

    if krisp_active:
        set_keep_all_user_speech(session, KEEP_ALL_USER_SPEECH)

    def log_latency_summary(module: str, values: list[float] | Any) -> None:
        samples = sorted(float(value) for value in values)
        if not samples:
            return
        p95 = samples[max(0, min(len(samples) - 1, int(len(samples) * 0.95) - 1))]
        logger.info(
            "LATENCY_SUMMARY module=%s count=%d mean_ms=%.3f p95_ms=%.3f max_ms=%.3f",
            module,
            len(samples),
            sum(samples) / len(samples),
            p95,
            samples[-1],
        )

    instruction_task: asyncio.Task[None] | None = None
    farewell_controller: FarewellHangupController | None = None

    async def automatic_hangup(reason: str) -> None:
        if agent._hangup_started:
            return
        agent._hangup_started = True
        if agent._pending_outcome is None:
            agent._pending_outcome = {
                "outcome": "incomplete",
                "summary": "Разговор завершён после прощания без структурированного итога.",
                "agreed_price_kzt": None,
                "agreed_price_gbp": None,
                "volume_m3": None,
                "next_step": "Проверить детали разговора и при необходимости перезвонить.",
            }
        if call_config.get("result_channel") == "web" and demo_emitter and not agent._web_outcome_published:
            await publish_web_outcome(demo_session_id, ctx.room.name, agent._pending_outcome)
            agent._web_outcome_published = True
        await stop_recording_once()
        await hang_up_sip_participant(ctx.room.name, call_config["target_identity"])
        if call_config.get("result_channel") == "web" and demo_emitter and not agent._web_ended_published:
            emit_demo_event(demo_session_id, ctx.room.name, "call.ended", {"reason": reason})
            agent._web_ended_published = True
        session.shutdown(drain=False)

    farewell_controller = FarewellHangupController(automatic_hangup)

    async def on_close(*_args: object) -> None:
        for task in (
            getattr(agent, "_owner_question_task", None),
            getattr(agent, "_owner_continuation_task", None),
            instruction_task,
        ):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if farewell_controller is not None:
            await farewell_controller.close()
        await stop_recording_once()
        for module, values in pipeline_latency_ms.items():
            log_latency_summary(module, values)

        if agent._pending_outcome is not None and not outcome_saved:
            outcome = agent._pending_outcome
            if call_config.get("result_channel") == "web" and demo_emitter:
                if not agent._web_outcome_published:
                    await publish_web_outcome(demo_session_id, ctx.room.name, outcome)
                path = recording_path()
                if path:
                    await asyncio.to_thread(_wait_for_recording, Path(path))
                    emit_demo_event(demo_session_id, ctx.room.name, "recording.ready", {
                        "url": f"/api/calls/{demo_session_id}/recording",
                    })
                if not agent._web_ended_published:
                    emit_demo_event(demo_session_id, ctx.room.name, "call.ended", {"reason": "completed"})
                mark_outcome_saved()
                logger.info("published staged call outcome to web after call ended")
                return
            try:
                await asyncio.to_thread(
                    deliver_outcome_to_telegram, outcome, recording_path()
                )
                mark_outcome_saved()
                logger.info("pushed staged call outcome directly to Telegram after call ended")
            except Exception:
                # A direct push must not lose the outcome. The existing cron queue
                # is retained only as a failure fallback, never as the normal path.
                logger.exception("direct Telegram outcome push failed; queuing fallback")
                path = write_call_outcome(
                    results_dir=CALL_RESULTS_DIR,
                    room_name=ctx.room.name,
                    outcome=outcome["outcome"],
                    summary=outcome["summary"],
                    agreed_price_kzt=outcome["agreed_price_kzt"],
                    agreed_price_gbp=outcome["agreed_price_gbp"],
                    volume_m3=outcome["volume_m3"],
                    next_step=outcome["next_step"],
                    recording_path=recording_path(),
                )
                mark_outcome_saved()
                logger.warning("queued staged call outcome fallback at %s", path)
        else:
            text = " ".join(item for item in transcript if item)
            fallback = {
                "outcome": "incomplete",
                "summary": "Звонок завершился без структурированного итога.",
                "agreed_price_kzt": None,
                "agreed_price_gbp": None,
                "volume_m3": None,
                "next_step": (
                    f"Проверить запись звонка. Распознанная речь: {text}"
                    if text else "Перезвонить и уточнить итог."
                ),
            }
            if call_config.get("result_channel") == "web" and demo_emitter:
                emit_demo_event(demo_session_id, ctx.room.name, "call.outcome", fallback)
                path = recording_path()
                if path:
                    await asyncio.to_thread(_wait_for_recording, Path(path))
                    emit_demo_event(demo_session_id, ctx.room.name, "recording.ready", {
                        "url": f"/api/calls/{demo_session_id}/recording",
                    })
                emit_demo_event(demo_session_id, ctx.room.name, "call.ended", {"reason": "completed"})
                mark_outcome_saved()
                logger.info("published fallback call outcome to web after call ended")
                return
            try:
                await asyncio.to_thread(
                    deliver_outcome_to_telegram, fallback, recording_path()
                )
                mark_outcome_saved()
                logger.info("pushed fallback call outcome directly to Telegram after call ended")
            except Exception:
                logger.exception("direct Telegram fallback push failed; queuing fallback")
                save_fallback_outcome()

    def record_agent_state(event: object) -> None:
        # Krisp IP arms/disarms itself from the session's own agent-state
        # events now, so no manual bot_speaking bookkeeping is needed.
        if demo_emitter:
            state = str(getattr(event, "new_state", ""))
            if state:
                emit_demo_event(demo_session_id, ctx.room.name, "call.state", {"state": state})
        if farewell_controller is not None:
            farewell_controller.on_agent_state(str(getattr(event, "new_state", "")))

    def record_user_state(event: object) -> None:
        if farewell_controller is not None:
            farewell_controller.on_user_state(str(getattr(event, "new_state", "")))

    def record_conversation_for_hangup(event: object) -> None:
        item = getattr(event, "item", None)
        text = str(getattr(item, "text_content", "") or "").strip()
        if not text or farewell_controller is None:
            return
        role = getattr(item, "role", None)
        if role == "assistant":
            farewell_controller.on_assistant_final(text)
        elif role == "user":
            farewell_controller.on_user_final(text)

    def record_conversation_item(event: object) -> None:
        if not demo_emitter:
            return
        item = getattr(event, "item", None)
        if getattr(item, "role", None) != "assistant":
            return
        text = str(getattr(item, "text_content", "") or "").strip()
        if text:
            emit_demo_event(demo_session_id, ctx.room.name, "transcript.assistant.final", {
                "utterance_id": str(getattr(item, "id", "") or "active"),
                "text": text,
            })

    session.on("user_input_transcribed", record_transcript)
    session.on("agent_state_changed", record_agent_state)
    session.on("user_state_changed", record_user_state)
    session.on("conversation_item_added", record_conversation_for_hangup)

    session.on("metrics_collected", record_pipeline_metrics)
    session.on("close", lambda *a: asyncio.create_task(on_close(*a)))

    agent = Gen2BAssistant(
        room_name=ctx.room.name,
        participant_identity=call_config["target_identity"],
        on_outcome_saved=mark_outcome_saved,
        system_prompt=call_config["system_prompt"],
        recording_path_getter=recording_path,
        stop_recording=stop_recording_once,
        personal_assistant=call_config["persona_key"] == "armen_personal_assistant",
        booking_confirmation_required=call_config["booking_confirmation_required"],
        owner_channel=call_config["owner_channel"],
        result_channel=call_config["result_channel"],
        demo_session_id=demo_session_id,
        transcript_emitter=demo_emitter,
        tts_intonation=call_config["tts_intonation"],
    )

    async def relay_live_instructions() -> None:
        if not demo_session_id or call_config.get("owner_channel") != "web":
            return
        while True:
            item = await asyncio.to_thread(
                pop_instruction,
                LIVE_CALLBACK_DIR,
                session_id=demo_session_id,
                room_name=ctx.room.name,
            )
            if item is None:
                await asyncio.sleep(0.2)
                continue
            instruction_id = item["instruction_id"]
            instruction = item["text"]
            logger.info("OWNER_INSTRUCTION received room=%s instruction_id=%s", ctx.room.name, instruction_id)
            await apply_live_instruction(session, instruction)
            logger.info("OWNER_INSTRUCTION dispatched room=%s instruction_id=%s", ctx.room.name, instruction_id)
            emit_demo_event(
                demo_session_id,
                ctx.room.name,
                "owner.instruction.accepted",
                {"instruction_id": instruction_id},
            )

    if krisp_active:
        await session.start(
            room=ctx.room,
            agent=agent,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    sample_rate=KRISP_SAMPLE_RATE,
                    num_channels=1,
                    frame_size_ms=KRISP_FRAME_DURATION_MS,
                    # Noise cancellation runs on the room audio input, i.e.
                    # BEFORE the audio reaches VAD/STT/TP/IP.
                    noise_cancellation=noise_cancellation,
                ),
                audio_output=True,
            ),
        )
        # attach() must happen after start(): it wraps session.input.audio so
        # both detectors receive every frame.
        turn_detector.attach(session)
        interrupt_detector.attach(session)
        # Backchannel gate: a turn committed over the agent's speech that Krisp
        # IP deliberately did NOT treat as an interruption is a "uh-huh", not a
        # real turn — don't let it hijack the conversation.
        attach_backchannel_gate(session, interrupt_detector)

        @interrupt_detector.on_interrupt
        def _on_krisp_interrupt(probability: float) -> None:
            logger.info("Krisp IP interruption p=%.3f", probability)
            asyncio.ensure_future(session.interrupt(force=True))
    else:
        await session.start(room=ctx.room, agent=agent)

    instruction_task = asyncio.create_task(relay_live_instructions())

    # Most personas speak first.  A persona with wait_for_user_first instead
    # stays silent and lets AgentSession generate its first reply only after a
    # real caller turn has been transcribed and committed.
    if not call_config["wait_for_user_first"]:
        await wait_for_sip_call_active(ctx.room, call_config["target_identity"])
        greeting = session.generate_reply(
            instructions=call_config["greeting_instructions"],
            allow_interruptions=False,
        )
        if krisp_active:
            mark_no_rescue(greeting)
            interrupt_detector.protect_speech(greeting)
        await greeting
    else:
        logger.info("waiting for caller's first utterance before agent speaks")


AGENT_NAME = os.environ.get("GEN2B_AGENT_NAME", "gen2b-agent")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=AGENT_NAME))
