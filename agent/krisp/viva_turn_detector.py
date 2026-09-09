"""Krisp VIVA End-of-Turn (Turn Prediction) detector for LiveKit Agents.

This module provides a turn detector backed by the Krisp VIVA TP model
(``krisp-viva-tp-v3.kef``).  It implements the ``_TurnDetector`` protocol
defined in ``livekit.agents.voice.turn`` so it can be passed directly to
``TurnHandlingOptions(turn_detection=...)`` in ``AgentSession``.

Unlike language-model-based turn detectors, this implementation works
**purely on audio** — no LLM call, no text transcript required.  Under the
hood it runs ``krisp_audio.TtInt16`` per audio frame during the user's
utterance and answers ``predict_end_of_turn`` using the last smoothed
probability accumulated while the user was speaking.

Usage::

    from krisp.viva_turn_detector import KrispVivaTurnDetector

    detector = KrispVivaTurnDetector.load()

    session = AgentSession(
        turn_handling=TurnHandlingOptions(
            turn_detection=detector,
        ),
        ...
    )
    await session.start(room=ctx.room, agent=agent, ...)
    detector.attach(session)  # wire audio — one extra line

How the LiveKit turn detection pipeline calls this
--------------------------------------------------
1. VAD fires ``END_OF_SPEECH`` — at this point the user has gone quiet.
2. LiveKit calls ``predict_end_of_turn(chat_ctx)`` once.
3. The returned float is compared to ``unlikely_threshold()``:
   - result >= threshold  →  user will probably *continue* speaking,
     so LiveKit waits up to ``max_delay`` before committing the turn.
   - result < threshold   →  user is done, LiveKit commits after ``min_delay``.

Audio wiring via ``attach``
---------------------------
Because the Krisp TP model works on raw audio (not text), the detector
must receive audio frames continuously while the user speaks.  Call
``detector.attach(session)`` **once** after ``session.start()``.

``attach`` subscribes to:
- ``user_state_changed`` — resets the internal EoT probability at the
  start of every new user utterance.
- ``session.input.audio`` — wraps the audio stream with a thin passthrough
  that calls ``push_frame`` for each incoming frame without touching the
  audio data.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from livekit.agents.voice.agent_session import AgentSession

logger = logging.getLogger("livekit.plugins.krisp.turn_detector")
logger.setLevel(logging.WARNING)

try:
    import krisp_audio  # type: ignore[import-not-found]

    KRISP_AUDIO_AVAILABLE = True
except ModuleNotFoundError:
    KRISP_AUDIO_AVAILABLE = False
    logger.warning(
        "krisp-audio package not found. "
        "Install it to use Krisp Turn Detector: pip install krisp-audio"
    )


@dataclass
class _TurnDetectorOptions:
    """Configuration options for KrispVivaTurnDetector."""

    model_path: str
    vad_model_path: str | None
    frame_duration_ms: int
    sample_rate: int
    smoothing_alpha: float
    unlikely_eot_threshold: float


class _TurnDetectorAudioInput(AudioInput):
    """Thin passthrough that feeds every incoming frame into the TP detector.

    Audio data is not modified; the only side-effect is calling
    ``detector.push_frame(frame)`` on each frame.
    """

    def __init__(self, source: AudioInput, detector: "KrispVivaTurnDetector") -> None:
        super().__init__(label="krisp_tp", source=source)
        self._detector = detector

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await super().__anext__()
        try:
            self._detector.push_frame(frame)
        except Exception:
            logger.exception("KrispVivaTurnDetector.push_frame failed")
        return frame


class KrispVivaTurnDetector:
    """Audio-driven End-of-Turn detector using the Krisp VIVA TP model.

    Implements the ``_TurnDetector`` protocol expected by LiveKit
    ``TurnHandlingOptions``.

    Example::

        detector = KrispVivaTurnDetector.load()

        session = AgentSession(
            turn_handling=TurnHandlingOptions(turn_detection=detector, ...),
            ...
        )
        await session.start(room=ctx.room, agent=agent, ...)
        detector.attach(session)  # one extra line — wires audio automatically
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
        smoothing_alpha: float = 0.4,
        unlikely_eot_threshold: float = 0.5,
    ) -> "KrispVivaTurnDetector":
        """Load and initialise the Krisp VIVA Turn Prediction model.

        Args:
            model_path: Path to ``krisp-viva-tp-v3.kef``.
                Falls back to ``KRISP_VIVA_EOU_MODEL_PATH`` env var.
            vad_model_path: Optional path to ``krisp-viva-vad-v2.kef`` for the
                per-frame VAD flag required by the TP model.  Falls back to
                ``KRISP_VIVA_VAD_MODEL_PATH`` env var.  When absent, VAD flags
                default to ``False`` (slightly lower accuracy).
            frame_duration_ms: Frame size in ms. One of ``{10,15,20,30,32}``.
            sample_rate: Model input sample rate in Hz.
                One of ``{8000,16000,24000,32000,44100,48000}``. Default 8000.
            smoothing_alpha: EMA factor for per-frame TP output (0 < α ≤ 1).
            unlikely_eot_threshold: ``eot_prob`` threshold. If Krisp's
                end-of-turn probability is **below** this value, LiveKit
                waits up to ``max_delay`` before committing the turn
                (user is likely still speaking). Default ``0.5``.

        Returns:
            A ready-to-use :class:`KrispVivaTurnDetector`.

        Raises:
            RuntimeError: If ``krisp-audio`` is not installed.
            ValueError: If model path is missing or parameters are invalid.
            FileNotFoundError: If the model file does not exist.
        """
        if not KRISP_AUDIO_AVAILABLE:
            raise RuntimeError(
                "krisp-audio package is not installed. Run: pip install krisp-audio"
            )

        resolved_model_path = model_path or os.getenv("KRISP_VIVA_EOU_MODEL_PATH")
        if not resolved_model_path:
            raise ValueError(
                "Krisp Turn Detector model path is not provided and "
                "KRISP_VIVA_EOU_MODEL_PATH environment variable is not set."
            )
        if not resolved_model_path.endswith(".kef"):
            raise ValueError("Krisp model file must have a .kef extension.")
        if not Path(resolved_model_path).is_file():
            raise FileNotFoundError(
                f"Krisp Turn Detector model file not found: {resolved_model_path}"
            )
        if frame_duration_ms not in KRISP_FRAME_DURATIONS:
            raise ValueError(
                f"Unsupported frame_duration_ms: {frame_duration_ms}. "
                f"Supported: {sorted(KRISP_FRAME_DURATIONS.keys())}"
            )

        resolved_vad_path = vad_model_path or os.getenv("KRISP_VIVA_VAD_MODEL_PATH")
        if resolved_vad_path and not Path(resolved_vad_path).is_file():
            logger.warning(
                "Krisp VAD model for TP not found at '%s'. "
                "Falling back to vad_flag=False.",
                resolved_vad_path,
            )
            resolved_vad_path = None

        opts = _TurnDetectorOptions(
            model_path=resolved_model_path,
            vad_model_path=resolved_vad_path,
            frame_duration_ms=frame_duration_ms,
            sample_rate=sample_rate,
            smoothing_alpha=smoothing_alpha,
            unlikely_eot_threshold=unlikely_eot_threshold,
        )
        return cls(opts=opts)

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, *, opts: _TurnDetectorOptions) -> None:
        self._opts = opts
        self._sdk_acquired = False

        try:
            KrispSDKManager.acquire()
            self._sdk_acquired = True
        except Exception as e:
            raise RuntimeError(f"Failed to acquire Krisp SDK: {e}") from e

        # Krisp sessions — created lazily on first frame
        self._tt_session: Any | None = None
        self._vad_session: Any | None = None
        self._session_sample_rate: int | None = None
        self._resampler: rtc.AudioResampler | None = None
        self._window_size: int | None = None

        # EoT probability state (float writes are atomic in CPython)
        self._eot_prob: float = 0.0
        self._exp_filter = utils.ExpFilter(alpha=opts.smoothing_alpha)
        self._pending_samples: np.ndarray = np.empty(0, dtype=np.int16)

        logger.info(
            "KrispVivaTurnDetector loaded "
            "(model=%s, vad=%s, sr=%dHz, frame=%dms)",
            Path(opts.model_path).name,
            "yes" if opts.vad_model_path else "no",
            opts.sample_rate,
            opts.frame_duration_ms,
        )

    # ------------------------------------------------------------------
    # _TurnDetector protocol
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Identifier of the underlying Krisp model."""
        return "krisp-viva-tp"

    @property
    def provider(self) -> str:
        """Name of the inference provider."""
        return "krisp"

    async def unlikely_threshold(self, language: Any = None) -> float | None:
        """Threshold below which LiveKit considers the turn NOT yet finished.

        LiveKit semantics: if ``predict_end_of_turn() < unlikely_threshold``
        the pipeline waits up to ``max_delay`` before committing the turn.
        We return ``eot_prob`` (probability that the turn IS over), so a low
        threshold (e.g. 0.3) means "if Krisp is less than 30% sure the turn
        is over — wait longer".

        Args:
            language: Language code (unused — model is language-agnostic).

        Returns:
            The configured ``unlikely_eot_threshold``.
        """
        return self._opts.unlikely_eot_threshold

    async def supports_language(self, language: Any = None) -> bool:
        """Always ``True`` — the Krisp TP model is language-agnostic."""
        return True

    async def predict_end_of_turn(
        self,
        chat_ctx: Any,
        *,
        timeout: float | None = None,
    ) -> float:
        """Return the probability that the user's turn IS over.

        LiveKit semantics:
        - if result >= unlikely_threshold → turn likely complete → min_delay
        - if result <  unlikely_threshold → user likely still speaking → max_delay

        The Krisp TP model outputs ``eot_prob`` — probability that the turn
        has ended — which matches this contract directly.

        Args:
            chat_ctx: Conversation context (not used by this model).
            timeout: Ignored — inference happens synchronously per-frame.

        Returns:
            Float in [0, 1]: probability that the turn IS over.
        """
        eot = self._eot_prob
        threshold = self._opts.unlikely_eot_threshold
        verdict = "COMPLETE ✅" if eot >= threshold else "INCOMPLETE ⏳"
        logger.warning(
            "Krisp EoT: %s  eot=%.3f  threshold=%.2f",
            verdict,
            eot,
            threshold,
        )
        return eot

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
        - Subscribes to ``user_state_changed`` to :meth:`reset` the EoT
          probability at the start of each new user utterance.

        Args:
            session: The running ``AgentSession`` to attach to.
        """
        audio = session.input.audio
        if audio is None:
            logger.warning(
                "KrispVivaTurnDetector.attach: session.input.audio is None, "
                "audio wiring skipped"
            )
        else:
            session.input.audio = _TurnDetectorAudioInput(source=audio, detector=self)
            logger.warning("KrispVivaTurnDetector wired to session audio input")

        @session.on("user_state_changed")
        def _on_user_state_changed(ev: Any) -> None:
            if ev.new_state == "speaking":
                self.reset()

    # ------------------------------------------------------------------
    # Audio ingestion
    # ------------------------------------------------------------------

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Feed a raw audio frame into the TP model.

        Typically called automatically after :meth:`attach`. Can also be
        called manually if you manage the audio pipeline yourself.

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
        """Reset EoT probability and frame buffer for a new utterance."""
        self._eot_prob = 0.0
        self._exp_filter = utils.ExpFilter(alpha=self._opts.smoothing_alpha)
        self._pending_samples = np.empty(0, dtype=np.int16)
        # logger.warning("KrispVivaTurnDetector reset (new utterance)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_sessions(self, sample_rate: int) -> bool:
        """Create Krisp TP (and optional VAD) sessions for *sample_rate*.

        Args:
            sample_rate: Sample rate of the incoming audio.

        Returns:
            ``True`` if sessions are ready, ``False`` on error.
        """
        if self._tt_session is not None and self._session_sample_rate == sample_rate:
            return True

        try:
            krisp_sr = int_to_krisp_sample_rate(self._opts.sample_rate)
            krisp_fd = int_to_krisp_frame_duration(self._opts.frame_duration_ms)

            model_info = krisp_audio.ModelInfo()
            model_info.path = self._opts.model_path

            tt_cfg = krisp_audio.TtSessionConfig()
            tt_cfg.inputSampleRate = krisp_sr
            tt_cfg.inputFrameDuration = krisp_fd
            tt_cfg.modelInfo = model_info
            self._tt_session = krisp_audio.TtInt16.create(tt_cfg)

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
                "Krisp TP sessions created: %dHz input → %dHz model / %dms frames",
                sample_rate,
                self._opts.sample_rate,
                self._opts.frame_duration_ms,
            )
            return True

        except Exception:
            logger.exception("Failed to create Krisp TP session for %dHz", sample_rate)
            self._tt_session = None
            self._vad_session = None
            return False

    def _process_window(self, window: np.ndarray) -> None:
        """Run one Krisp TP inference window and update the smoothed probability.

        Args:
            window: PCM-16 samples array of exactly ``_window_size`` elements.
        """
        assert self._tt_session is not None

        vad_flag = False
        if self._vad_session is not None:
            try:
                vad_prob: float = self._vad_session.process(window)
                vad_flag = vad_prob >= 0.5
            except Exception:
                logger.exception("Krisp aux-VAD inference failed")

        try:
            # Third argument: force_process=False — matches example usage
            raw_prob: float = self._tt_session.process(window, vad_flag, False)
        except Exception:
            logger.exception("Krisp TP inference failed")
            return

        self._eot_prob = self._exp_filter.apply(exp=1.0, sample=float(raw_prob))
        logger.debug(
            "Krisp TP frame: vad=%d raw=%.4f smoothed=%.4f",
            int(vad_flag),
            raw_prob,
            self._eot_prob,
        )

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
