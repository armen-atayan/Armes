"""Krisp VIVA Interrupt Detection for LiveKit Agents.

This module provides an interrupt detector backed by the Krisp VIVA IP model
(``krisp-viva-ip-v1.kef``).  It runs alongside agent speech and answers
whether the user is trying to interrupt the agent.

The ``IpInt16``/``IpFloat`` session processes audio per-frame during agent
playback and emits a probability that the user is interrupting.  A callback
(``on_interrupt``) is fired when the smoothed probability crosses the
configured threshold so the agent can decide whether to stop speaking.

Usage::

    from krisp.viva_interrupt_detector import KrispVivaInterruptDetector

    detector = KrispVivaInterruptDetector.load()

    session = AgentSession(...)
    await session.start(room=ctx.room, agent=agent, ...)
    detector.attach(session)

    @detector.on_interrupt
    def _handle_interrupt(probability: float) -> None:
        logger.info("User is interrupting! p=%.3f", probability)

How Krisp IP model works
------------------------
``ip_instance.process(frame, vad_flag)`` — returns a float probability [0, 1]
that the user utterance in *frame* is an interrupt of the currently-playing
agent audio.

- ``vad_flag`` — boolean indicating whether VAD detects speech in *frame*.
  An optional auxiliary VAD session (same model as used by the turn detector)
  provides this flag.  When no VAD model path is configured, ``vad_flag``
  defaults to ``False`` (slightly lower accuracy but still functional).
- Results are smoothed with an exponential moving average before comparison
  to the threshold.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.io import AudioInput

from krisp.krisp_instance import (
    KRISP_FRAME_DURATIONS,
    KrispSDKManager,
    int_to_krisp_frame_duration,
    int_to_krisp_sample_rate,
)
from krisp.turn_rescue import mark_no_rescue

if TYPE_CHECKING:
    from livekit.agents.voice.agent_session import AgentSession

logger = logging.getLogger("livekit.plugins.krisp.interrupt_detector")

try:
    import krisp_audio  # type: ignore[import-not-found]

    KRISP_AUDIO_AVAILABLE = True
except ModuleNotFoundError:
    KRISP_AUDIO_AVAILABLE = False
    logger.warning(
        "krisp-audio package not found. "
        "Install it to use Krisp Interrupt Detector: pip install krisp-audio"
    )


@dataclass
class _InterruptDetectorOptions:
    """Configuration options for KrispVivaInterruptDetector."""

    model_path: str
    vad_model_path: str | None
    frame_duration_ms: int
    sample_rate: int
    smoothing_alpha: float
    interrupt_threshold: float
    # Minimum continuous duration (seconds) above threshold before firing callback
    min_interrupt_duration: float
    # Cooldown (seconds) between successive interrupt callbacks
    cooldown_duration: float


class _InterruptAudioInput(AudioInput):
    """Thin passthrough that feeds every incoming frame into the IP detector.

    Audio data is not modified; the only side-effect is calling
    ``detector.push_frame(frame)`` on each frame.
    """

    def __init__(self, source: AudioInput, detector: "KrispVivaInterruptDetector") -> None:
        super().__init__(label="krisp_ip", source=source)
        self._detector = detector

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await super().__anext__()
        try:
            self._detector.push_frame(frame)
        except Exception:
            logger.exception("KrispVivaInterruptDetector.push_frame failed")
        return frame


class KrispVivaInterruptDetector:
    """Audio-driven interrupt detector using the Krisp VIVA IP model.

    Listens to the user's audio stream continuously and fires registered
    callbacks when the model decides the user is interrupting the agent.

    Interrupt callbacks are only active while the agent is speaking — the
    detector automatically arms/disarms itself by subscribing to
    ``agent_state_changed`` events inside :meth:`attach`.

    Example::

        detector = KrispVivaInterruptDetector.load()

        session = AgentSession(...)
        await session.start(room=ctx.room, agent=agent, ...)
        detector.attach(session)

        @detector.on_interrupt
        def _handle(probability: float) -> None:
            logger.warning("Interrupt detected: p=%.3f", probability)
    """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        *,
        model_path: str | None = None,
        vad_model_path: str | None = None,
        frame_duration_ms: int = 10,
        sample_rate: int = 8000,
        smoothing_alpha: float = 0.35,
        interrupt_threshold: float = 0.5,
        min_interrupt_duration: float = 0.2,
        cooldown_duration: float = 1.0,
    ) -> "KrispVivaInterruptDetector":
        """Load and initialise the Krisp VIVA Interrupt Prediction model.

        Args:
            model_path: Path to ``krisp-viva-ip-v1.kef``.
                Falls back to ``KRISP_VIVA_INTERRUPT_MODEL_PATH`` env var.
            vad_model_path: Optional path to ``krisp-viva-vad-v2.kef`` for
                per-frame VAD flags required by the IP model.  Falls back to
                ``KRISP_VIVA_VAD_MODEL_PATH`` env var.  When absent, VAD flags
                default to ``False``.
            frame_duration_ms: Frame size in ms. One of ``{10, 15, 20, 30, 32}``.
            sample_rate: Model input sample rate in Hz.
                One of ``{8000, 16000, 24000, 32000, 44100, 48000}``. Default 8000.
            smoothing_alpha: EMA smoothing factor for per-frame IP output (0 < α ≤ 1).
                Lower values → smoother but more latent response.
            interrupt_threshold: Smoothed probability above which an interrupt
                is detected. Default ``0.5``.
            min_interrupt_duration: Minimum continuous duration (seconds) above
                ``interrupt_threshold`` before the callback fires. Prevents
                spurious single-frame detections. Default ``0.2``.
            cooldown_duration: Minimum gap (seconds) between successive
                interrupt callbacks. Default ``1.0``.

        Returns:
            A ready-to-use :class:`KrispVivaInterruptDetector`.

        Raises:
            RuntimeError: If ``krisp-audio`` is not installed.
            ValueError: If model path is missing or parameters are invalid.
            FileNotFoundError: If the model file does not exist.
        """
        if not KRISP_AUDIO_AVAILABLE:
            raise RuntimeError(
                "krisp-audio package is not installed. Run: pip install krisp-audio"
            )

        resolved_model_path = model_path or os.getenv("KRISP_VIVA_INTERRUPT_MODEL_PATH")
        if not resolved_model_path:
            raise ValueError(
                "Krisp Interrupt Detector model path is not provided and "
                "KRISP_VIVA_INTERRUPT_MODEL_PATH environment variable is not set."
            )
        if not resolved_model_path.endswith(".kef"):
            raise ValueError("Krisp model file must have a .kef extension.")
        if not Path(resolved_model_path).is_file():
            raise FileNotFoundError(
                f"Krisp Interrupt Detector model file not found: {resolved_model_path}"
            )
        if frame_duration_ms not in KRISP_FRAME_DURATIONS:
            raise ValueError(
                f"Unsupported frame_duration_ms: {frame_duration_ms}. "
                f"Supported: {sorted(KRISP_FRAME_DURATIONS.keys())}"
            )

        resolved_vad_path = vad_model_path or os.getenv("KRISP_VIVA_VAD_MODEL_PATH")
        if resolved_vad_path and not Path(resolved_vad_path).is_file():
            logger.warning(
                "Krisp VAD model for IP not found at '%s'. Falling back to vad_flag=False.",
                resolved_vad_path,
            )
            resolved_vad_path = None

        opts = _InterruptDetectorOptions(
            model_path=resolved_model_path,
            vad_model_path=resolved_vad_path,
            frame_duration_ms=frame_duration_ms,
            sample_rate=sample_rate,
            smoothing_alpha=smoothing_alpha,
            interrupt_threshold=interrupt_threshold,
            min_interrupt_duration=min_interrupt_duration,
            cooldown_duration=cooldown_duration,
        )
        return cls(opts=opts)

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, *, opts: _InterruptDetectorOptions) -> None:
        self._opts = opts
        self._sdk_acquired = False

        try:
            KrispSDKManager.acquire()
            self._sdk_acquired = True
        except Exception as e:
            raise RuntimeError(f"Failed to acquire Krisp SDK: {e}") from e

        # Krisp sessions — created lazily on first frame
        self._ip_session: Any | None = None
        self._vad_session: Any | None = None
        self._session_sample_rate: int | None = None
        self._resampler: rtc.AudioResampler | None = None
        self._window_size: int | None = None

        # Smoothed interrupt probability (float writes are atomic in CPython)
        self._interrupt_prob: float = 0.0
        self._exp_filter = utils.ExpFilter(alpha=opts.smoothing_alpha)
        self._pending_samples: np.ndarray = np.empty(0, dtype=np.int16)

        # State: only emit callbacks when agent is speaking
        self._agent_speaking: bool = False

        # Attached session (set in attach()) and speeches explicitly created
        # with allow_interruptions=False — never interrupted by this detector.
        self._session: Any | None = None
        self._protected_speeches: "weakref.WeakSet[Any]" = weakref.WeakSet()

        # Debouncing: track how long we've been above threshold
        self._above_threshold_duration: float = 0.0
        self._last_callback_time: float = 0.0  # monotonic seconds

        # Registered interrupt callbacks
        self._interrupt_callbacks: list[Callable[[float], None]] = []

        logger.info(
            "KrispVivaInterruptDetector loaded "
            "(model=%s, vad=%s, sr=%dHz, frame=%dms, threshold=%.2f)",
            Path(opts.model_path).name,
            "yes" if opts.vad_model_path else "no",
            opts.sample_rate,
            opts.frame_duration_ms,
            opts.interrupt_threshold,
        )

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_interrupt(self, callback: Callable[[float], None]) -> Callable[[float], None]:
        """Register a callback to be called when an interrupt is detected.

        Can be used as a decorator::

            @detector.on_interrupt
            def _handle(probability: float) -> None:
                logger.warning("Interrupt! p=%.3f", probability)

        Or directly::

            detector.on_interrupt(my_handler)

        Args:
            callback: Callable that receives the smoothed interrupt probability.

        Returns:
            The callback unchanged (for use as a decorator).
        """
        self._interrupt_callbacks.append(callback)
        return callback

    def remove_interrupt_callback(self, callback: Callable[[float], None]) -> None:
        """Remove a previously registered interrupt callback.

        Args:
            callback: The callback to remove.
        """
        try:
            self._interrupt_callbacks.remove(callback)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Session wiring
    # ------------------------------------------------------------------

    def attach(self, session: "AgentSession") -> None:
        """Wire the detector to an already-started ``AgentSession``.

        Call this **once** after ``await session.start(...)``::

            detector.attach(session)

        This method:
        - Wraps ``session.input.audio`` with a passthrough that calls
          :meth:`push_frame` for every incoming frame.
        - Subscribes to ``agent_state_changed`` to arm/disarm interrupt
          detection depending on whether the agent is currently speaking.
        - Wraps ``session.say`` / ``session.generate_reply`` so that speeches
          created with an explicit ``allow_interruptions=False`` are
          automatically protected from this detector (see
          :meth:`protect_speech`).

        Args:
            session: The running ``AgentSession`` to attach to.
        """
        self._session = session
        self._wrap_speech_factories(session)

        audio = session.input.audio
        if audio is None:
            logger.warning(
                "KrispVivaInterruptDetector.attach: session.input.audio is None, "
                "audio wiring skipped"
            )
        else:
            session.input.audio = _InterruptAudioInput(source=audio, detector=self)
            logger.info("KrispVivaInterruptDetector wired to session audio input")

        @session.on("agent_state_changed")
        def _on_agent_state_changed(ev: Any) -> None:  # noqa: ANN401
            self._handle_agent_state_changed(ev)

    def protect_speech(self, handle: Any) -> Any:
        """Mark a ``SpeechHandle`` as protected from Krisp interruptions.

        While a protected speech is the session's current speech, interrupt
        callbacks are NOT fired even if the IP model detects an interruption.

        Speeches created via ``session.say(...)`` / ``session.generate_reply(...)``
        with an explicit ``allow_interruptions=False`` are protected
        automatically after :meth:`attach`.  Use this method for handles
        created *before* ``attach()`` (e.g. a greeting in ``on_enter``)::

            handle = session.generate_reply(..., allow_interruptions=False)
            detector.protect_speech(handle)
            await handle

        Args:
            handle: The ``SpeechHandle`` to protect.

        Returns:
            The same handle, for chaining.
        """
        self._protected_speeches.add(handle)
        # Защищённую реплику нельзя «спасать» turn-rescue-логикой: ход юзера,
        # пришедший во время неё, — эхо/фон (IP подавлен), а не ответ.
        mark_no_rescue(handle)
        return handle

    def _wrap_speech_factories(self, session: "AgentSession") -> None:
        """Wrap ``say``/``generate_reply`` to auto-protect uninterruptible speech.

        Only speeches with an *explicit* ``allow_interruptions=False`` keyword
        are registered — the session-level default cannot be used here because
        with ``InterruptionOptions(enabled=False)`` every speech defaults to
        ``allow_interruptions=False`` and the detector would never fire.
        """
        for name in ("say", "generate_reply"):
            original = getattr(session, name, None)
            if original is None or getattr(original, "_krisp_ip_wrapped", False):
                continue

            @functools.wraps(original)
            def _wrapper(*args: Any, _orig: Any = original, **kwargs: Any) -> Any:
                handle = _orig(*args, **kwargs)
                if kwargs.get("allow_interruptions") is False:
                    self.protect_speech(handle)
                return handle

            _wrapper._krisp_ip_wrapped = True  # type: ignore[attr-defined]
            setattr(session, name, _wrapper)

    def _is_current_speech_protected(self) -> bool:
        """Whether the session's current speech must not be interrupted."""
        if self._session is None:
            return False
        speech = getattr(self._session, "current_speech", None)
        return speech is not None and speech in self._protected_speeches

    def _handle_agent_state_changed(self, ev: Any) -> None:
        new_state = ev.new_state if hasattr(ev, "new_state") else str(ev)
        was_speaking = self._agent_speaking
        self._agent_speaking = new_state == "speaking"
        if was_speaking and not self._agent_speaking:
            # Agent stopped speaking — reset debounce state
            self._reset_debounce()
            logger.debug("KrispVivaInterruptDetector: agent stopped — debounce reset")
        elif not was_speaking and self._agent_speaking:
            logger.debug("KrispVivaInterruptDetector: agent started speaking — armed")

    # ------------------------------------------------------------------
    # Audio ingestion
    # ------------------------------------------------------------------

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Feed a raw audio frame into the IP model.

        Typically called automatically after :meth:`attach`. Can also be
        called manually when managing the audio pipeline directly.

        Interrupt callbacks are only fired while the agent is speaking
        (``_agent_speaking`` is True).

        Args:
            frame: Raw ``AudioFrame`` from the user's audio track.
        """
        if not self._ensure_sessions(frame.sample_rate):
            return

        assert self._window_size is not None

        if self._resampler is not None:
            resampled = self._resampler.push(frame)
            if not resampled:
                return
            combined = utils.combine_frames(resampled)
        else:
            combined = frame

        incoming = np.frombuffer(combined.data, dtype=np.int16)
        self._pending_samples = np.concatenate([self._pending_samples, incoming])

        while len(self._pending_samples) >= self._window_size:
            window = self._pending_samples[: self._window_size]
            self._pending_samples = self._pending_samples[self._window_size :]
            self._process_window(window)

    def reset(self) -> None:
        """Reset interrupt probability and frame buffer."""
        self._interrupt_prob = 0.0
        self._exp_filter = utils.ExpFilter(alpha=self._opts.smoothing_alpha)
        self._pending_samples = np.empty(0, dtype=np.int16)
        self._reset_debounce()
        logger.debug("KrispVivaInterruptDetector reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_debounce(self) -> None:
        """Reset the above-threshold duration counter."""
        self._above_threshold_duration = 0.0

    def _ensure_sessions(self, sample_rate: int) -> bool:
        """Create Krisp IP (and optional VAD) sessions for *sample_rate*.

        Args:
            sample_rate: Sample rate of the incoming audio.

        Returns:
            ``True`` if sessions are ready, ``False`` on error.
        """
        if self._ip_session is not None and self._session_sample_rate == sample_rate:
            return True

        try:
            krisp_sr = int_to_krisp_sample_rate(self._opts.sample_rate)
            krisp_fd = int_to_krisp_frame_duration(self._opts.frame_duration_ms)

            model_info = krisp_audio.ModelInfo()
            model_info.path = self._opts.model_path

            ip_cfg = krisp_audio.IpSessionConfig()
            ip_cfg.inputSampleRate = krisp_sr
            ip_cfg.inputFrameDuration = krisp_fd
            ip_cfg.modelInfo = model_info
            self._ip_session = krisp_audio.IpInt16.create(ip_cfg)

            self._vad_session = None
            if self._opts.vad_model_path:
                vad_model_info = krisp_audio.ModelInfo()
                vad_model_info.path = self._opts.vad_model_path

                vad_cfg = krisp_audio.VadSessionConfig()
                vad_cfg.inputSampleRate = krisp_sr
                vad_cfg.inputFrameDuration = krisp_fd
                vad_cfg.modelInfo = vad_model_info
                self._vad_session = krisp_audio.VadInt16.create(vad_cfg)

            self._resampler = None
            if sample_rate != self._opts.sample_rate:
                self._resampler = rtc.AudioResampler(
                    input_rate=sample_rate,
                    output_rate=self._opts.sample_rate,
                    quality=rtc.AudioResamplerQuality.QUICK,
                )

            self._session_sample_rate = sample_rate
            self._window_size = int(
                self._opts.sample_rate * self._opts.frame_duration_ms / 1000
            )
            logger.info(
                "Krisp IP sessions created: %dHz input → %dHz model / %dms frames",
                sample_rate,
                self._opts.sample_rate,
                self._opts.frame_duration_ms,
            )
            return True

        except Exception:
            logger.exception("Failed to create Krisp IP session for %dHz", sample_rate)
            self._ip_session = None
            self._vad_session = None
            return False

    def _process_window(self, window: np.ndarray) -> None:
        """Run one Krisp IP inference window and update the smoothed probability.

        Fires interrupt callbacks if the smoothed probability exceeds the
        configured threshold for at least ``min_interrupt_duration`` seconds,
        subject to ``cooldown_duration`` between successive callbacks.

        Args:
            window: PCM-16 samples array of exactly ``_window_size`` elements.
        """
        assert self._ip_session is not None

        vad_flag = False
        if self._vad_session is not None:
            try:
                vad_prob: float = self._vad_session.process(window)
                vad_flag = vad_prob >= 0.5
            except Exception:
                logger.exception("Krisp aux-VAD inference failed")

        try:
            raw_prob: float = self._ip_session.process(window, vad_flag)
        except Exception:
            logger.exception("Krisp IP inference failed")
            return

        self._interrupt_prob = self._exp_filter.apply(exp=1.0, sample=float(raw_prob))

        frame_duration_s = self._opts.frame_duration_ms / 1000.0
        logger.debug(
            "Krisp IP frame: vad=%d raw=%.4f smoothed=%.4f agent_speaking=%s",
            int(vad_flag),
            raw_prob,
            self._interrupt_prob,
            self._agent_speaking,
        )

        if not self._agent_speaking:
            # No point accumulating above-threshold time when agent is silent
            self._above_threshold_duration = 0.0
            return

        if self._is_current_speech_protected():
            # Current speech was created with allow_interruptions=False —
            # never fire interrupt callbacks for it.
            self._above_threshold_duration = 0.0
            return

        if self._interrupt_prob >= self._opts.interrupt_threshold:
            self._above_threshold_duration += frame_duration_s
        else:
            self._above_threshold_duration = 0.0

        if self._above_threshold_duration < self._opts.min_interrupt_duration:
            return

        # Check cooldown using event loop's monotonic clock
        try:
            loop = asyncio.get_running_loop()
            now = loop.time()
        except RuntimeError:
            import time
            now = time.monotonic()

        if now - self._last_callback_time < self._opts.cooldown_duration:
            return

        self._last_callback_time = now
        self._above_threshold_duration = 0.0

        logger.warning(
            "🔕 Krisp IP: interruption detected (p=%.3f) — stopping agent (threshold=%.2f)",
            self._interrupt_prob,
            self._opts.interrupt_threshold,
        )

        for cb in list(self._interrupt_callbacks):
            try:
                cb(self._interrupt_prob)
            except Exception:
                logger.exception("Interrupt callback raised an exception")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def interrupt_probability(self) -> float:
        """Latest smoothed interrupt probability [0, 1]."""
        return self._interrupt_prob

    @property
    def agent_speaking(self) -> bool:
        """Whether the agent is currently in the speaking state."""
        return self._agent_speaking

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Release the Krisp SDK reference on garbage collection."""
        if KrispSDKManager is None:
            return
        if getattr(self, "_sdk_acquired", False):
            try:
                KrispSDKManager.release()
                self._sdk_acquired = False
            except Exception:
                pass
