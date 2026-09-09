"""Krisp VIVA adapters for LiveKit Agents.

The audio path is deliberately fixed to mono PCM16, 16 kHz, 20 ms frames:
LiveKit track -> Krisp Voice Isolation FrameProcessor -> Krisp VAD stream -> STT.
TP/IP probabilities are computed alongside VAD and exposed through KrispSignals.
They are observational by default until thresholds are validated on live calls.
"""

from __future__ import annotations

import atexit
import asyncio
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import krisp_audio
import numpy as np
from livekit import agents, rtc

logger = logging.getLogger("gen2b-agent.krisp")

SAMPLE_RATE = 16_000
FRAME_DURATION_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_DURATION_MS // 1000


@dataclass
class KrispSignals:
    vad_probability: float = 0.0
    turn_probability: float = 0.0
    interruption_probability: float = 0.0
    speaking: bool = False
    bot_speaking: bool = False
    processed_frames: int = 0
    vi_latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=10_000))
    vad_latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=10_000))
    tp_latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=10_000))
    ip_latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=10_000))


@dataclass
class KrispVADOptions:
    min_speech_duration: float = 0.06
    min_silence_duration: float = 0.55
    prefix_padding_duration: float = 0.30
    max_buffered_speech: float = 60.0
    activation_threshold: float = 0.50
    deactivation_threshold: float = 0.35
    tp_endpointing_enabled: bool = False
    tp_threshold: float = 0.70
    tp_min_silence_duration: float = 0.18


class KrispRuntime:
    """Own one SDK global context and persistent ModelInfo objects per process."""

    def __init__(self, *, license_key: str, models_dir: Path | str) -> None:
        if not license_key:
            raise ValueError("KRISP_VIVA_SDK_LICENSE_KEY is empty")
        self.models_dir = Path(models_dir).resolve()
        self.signals = KrispSignals()
        self.license_errors: list[tuple[str, str]] = []
        self._destroyed = False
        self._model_infos: dict[str, Any] = {}

        required = {
            "vad": "krisp-viva-vad-v2.kef",
            "vi": "krisp-viva-vi-tel-v2.kef",
            "tp": "krisp-viva-tp-v3.kef",
            "ip": "krisp-viva-ip-v1.kef",
        }
        for key, filename in required.items():
            path = self.models_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            info = krisp_audio.ModelInfo()
            info.path = str(path)
            # Critical: the binding stores a native pointer. Retain ModelInfo.
            self._model_infos[key] = info

        krisp_audio.globalInit(
            "",
            license_key,
            self._licensing_error,
            self._sdk_log,
            krisp_audio.LogLevel.Off,
        )
        atexit.register(self.destroy)
        logger.info("Krisp VIVA SDK initialized with all four models")

    def _licensing_error(self, code: object, message: str) -> None:
        self.license_errors.append((str(code), str(message)))
        logger.error("Krisp licensing error code=%s: %s", code, message)

    @staticmethod
    def _sdk_log(message: str, level: object) -> None:
        logger.debug("Krisp SDK [%s] %s", level, message)

    def create_voice_isolation(self):
        cfg = krisp_audio.NcSessionConfig()
        cfg.inputSampleRate = krisp_audio.SamplingRate.Sr16000Hz
        cfg.inputFrameDuration = krisp_audio.FrameDuration.Fd20ms
        cfg.outputSampleRate = krisp_audio.SamplingRate.Sr16000Hz
        cfg.modelInfo = self._model_infos["vi"]
        return krisp_audio.NcInt16.create(cfg)

    def create_vad(self):
        cfg = krisp_audio.VadSessionConfig()
        cfg.inputSampleRate = krisp_audio.SamplingRate.Sr16000Hz
        cfg.inputFrameDuration = krisp_audio.FrameDuration.Fd20ms
        cfg.modelInfo = self._model_infos["vad"]
        return krisp_audio.VadInt16.create(cfg)

    def create_turn_prediction(self):
        cfg = krisp_audio.TtSessionConfig()
        cfg.inputSampleRate = krisp_audio.SamplingRate.Sr16000Hz
        cfg.inputFrameDuration = krisp_audio.FrameDuration.Fd20ms
        cfg.modelInfo = self._model_infos["tp"]
        return krisp_audio.TtInt16.create(cfg)

    def create_interruption_prediction(self):
        cfg = krisp_audio.IpSessionConfig()
        cfg.inputSampleRate = krisp_audio.SamplingRate.Sr16000Hz
        cfg.inputFrameDuration = krisp_audio.FrameDuration.Fd20ms
        cfg.modelInfo = self._model_infos["ip"]
        return krisp_audio.IpInt16.create(cfg)

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        try:
            krisp_audio.globalDestroy()
        except Exception:
            logger.exception("Krisp globalDestroy failed")


class KrispVoiceIsolationProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    def __init__(self, runtime: KrispRuntime, *, suppression_level: int = 100) -> None:
        self._runtime = runtime
        self._suppression_level = max(0, min(100, int(suppression_level)))
        self._enabled = True
        self._lock = threading.RLock()
        self._session = runtime.create_voice_isolation()
        logger.info("Krisp Voice Isolation enabled suppression=%d", self._suppression_level)

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = bool(value)

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        with self._lock:
            if not self._enabled or self._session is None:
                return frame
            if frame.sample_rate != SAMPLE_RATE or frame.num_channels != 1:
                raise ValueError(
                    f"Krisp VI requires mono {SAMPLE_RATE}Hz; got "
                    f"{frame.num_channels}ch {frame.sample_rate}Hz"
                )
            if frame.samples_per_channel != SAMPLES_PER_FRAME:
                raise ValueError(
                    f"Krisp VI requires {FRAME_DURATION_MS}ms/{SAMPLES_PER_FRAME} samples; "
                    f"got {frame.samples_per_channel}"
                )
            samples = np.frombuffer(frame.data, dtype=np.int16)
            started = time.perf_counter()
            clean = np.asarray(
                self._session.process(samples, self._suppression_level), dtype=np.int16
            )
            self._runtime.signals.vi_latency_ms.append(
                (time.perf_counter() - started) * 1000
            )
            if clean.size != samples.size:
                raise RuntimeError(
                    f"Krisp VI returned {clean.size} samples for {samples.size}"
                )
            return rtc.AudioFrame(
                data=clean.tobytes(),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=clean.size,
                userdata=frame.userdata,
            )

    def _close(self) -> None:
        with self._lock:
            self._enabled = False
            self._session = None


class KrispVAD(agents.vad.VAD):
    def __init__(self, runtime: KrispRuntime, options: KrispVADOptions | None = None) -> None:
        super().__init__(
            capabilities=agents.vad.VADCapabilities(
                update_interval=FRAME_DURATION_MS / 1000
            )
        )
        self.runtime = runtime
        self.options = options or KrispVADOptions()
        self._stream_count = 0

    @property
    def model(self) -> str:
        return "krisp-viva-vad-v2"

    @property
    def provider(self) -> str:
        return "Krisp"

    @property
    def min_silence_duration(self) -> float:
        return self.options.min_silence_duration

    def stream(self) -> "KrispVADStream":
        # AgentSession creates one stream for endpointing and a second one for
        # the non-streaming STT adapter. Only the first needs TP/IP; running
        # those heavier predictors twice wastes CPU without adding signals.
        include_predictors = self._stream_count == 0
        self._stream_count += 1
        return KrispVADStream(
            self,
            self.runtime,
            self.options,
            include_predictors=include_predictors,
        )


class KrispVADStream(agents.vad.VADStream):
    def __init__(
        self,
        vad: KrispVAD,
        runtime: KrispRuntime,
        options: KrispVADOptions,
        *,
        include_predictors: bool,
    ) -> None:
        self._runtime = runtime
        self._options = options
        self._include_predictors = include_predictors
        self._vad_session = None
        self._tp_session = None
        self._ip_session = None
        self._recreate_native_sessions()
        super().__init__(vad)
        logger.info(
            "Krisp VAD stream created predictors=%s", include_predictors
        )

    def _recreate_native_sessions(self) -> None:
        self._vad_session = None
        self._tp_session = None
        self._ip_session = None
        self._vad_session = self._runtime.create_vad()
        if self._include_predictors:
            self._tp_session = self._runtime.create_turn_prediction()
            self._ip_session = self._runtime.create_interruption_prediction()

    async def aclose(self) -> None:
        await super().aclose()
        self._vad_session = None
        self._tp_session = None
        self._ip_session = None

    @staticmethod
    def _audio_frame(data: bytes | bytearray) -> rtc.AudioFrame:
        payload = bytes(data)
        return rtc.AudioFrame(
            data=payload,
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=len(payload) // 2,
        )

    async def _main_task(self) -> None:
        opts = self._options
        frame_duration = FRAME_DURATION_MS / 1000
        prefix_frames = max(1, math.ceil(opts.prefix_padding_duration / frame_duration))
        max_speech_frames = max(1, math.ceil(opts.max_buffered_speech / frame_duration))
        prefix: deque[bytes] = deque(maxlen=prefix_frames)
        speech_frames: deque[bytes] = deque(maxlen=max_speech_frames + prefix_frames)
        pending = bytearray()

        speaking = False
        speech_threshold_duration = 0.0
        silence_threshold_duration = 0.0
        speech_duration = 0.0
        silence_duration = 0.0
        samples_index = 0
        timestamp = 0.0

        def reset_state() -> None:
            nonlocal speaking, speech_threshold_duration, silence_threshold_duration
            nonlocal speech_duration, silence_duration, samples_index, timestamp
            pending.clear()
            prefix.clear()
            speech_frames.clear()
            speaking = False
            speech_threshold_duration = 0.0
            silence_threshold_duration = 0.0
            speech_duration = 0.0
            silence_duration = 0.0
            samples_index = 0
            timestamp = 0.0
            self._runtime.signals.speaking = False

        async for input_frame in self._input_ch:
            if isinstance(input_frame, self._FlushSentinel):
                reset_state()
                # Krisp sessions are stateful and expose no reset method. A
                # plain flush is a hard boundary; end_input() also sends a
                # sentinel but closes the channel, so do not rebuild on exit.
                if not self._input_ch.closed:
                    self._recreate_native_sessions()
                continue
            if not isinstance(input_frame, rtc.AudioFrame):
                continue
            if input_frame.sample_rate != SAMPLE_RATE or input_frame.num_channels != 1:
                raise ValueError(
                    f"Krisp VAD requires mono {SAMPLE_RATE}Hz; got "
                    f"{input_frame.num_channels}ch {input_frame.sample_rate}Hz"
                )
            pending.extend(input_frame.data.cast("B"))
            bytes_per_frame = SAMPLES_PER_FRAME * 2

            while len(pending) >= bytes_per_frame:
                raw = bytes(pending[:bytes_per_frame])
                del pending[:bytes_per_frame]
                samples = np.frombuffer(raw, dtype=np.int16)
                vad_started = time.perf_counter()
                probability = float(self._vad_session.process(samples))
                vad_duration = time.perf_counter() - vad_started
                self._runtime.signals.vad_latency_ms.append(vad_duration * 1000)
                voice_flag = probability >= opts.activation_threshold
                if self._tp_session is not None:
                    tp_started = time.perf_counter()
                    turn_probability = float(
                        self._tp_session.process(
                            samples,
                            voice_flag,
                            self._runtime.signals.bot_speaking,
                        )
                    )
                    self._runtime.signals.tp_latency_ms.append(
                        (time.perf_counter() - tp_started) * 1000
                    )
                else:
                    turn_probability = self._runtime.signals.turn_probability
                if self._ip_session is not None:
                    ip_started = time.perf_counter()
                    interruption_probability = float(
                        self._ip_session.process(samples, voice_flag)
                    )
                    self._runtime.signals.ip_latency_ms.append(
                        (time.perf_counter() - ip_started) * 1000
                    )
                else:
                    interruption_probability = (
                        self._runtime.signals.interruption_probability
                    )

                signals = self._runtime.signals
                signals.vad_probability = probability
                if self._tp_session is not None:
                    signals.turn_probability = turn_probability
                if self._ip_session is not None:
                    signals.interruption_probability = interruption_probability
                signals.processed_frames += 1

                samples_index += SAMPLES_PER_FRAME
                timestamp += frame_duration
                frame = self._audio_frame(raw)
                prefix.append(raw)

                if speaking:
                    speech_duration += frame_duration
                    speech_frames.append(raw)
                else:
                    silence_duration += frame_duration

                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.INFERENCE_DONE,
                        samples_index=samples_index,
                        timestamp=timestamp,
                        speech_duration=speech_duration,
                        silence_duration=silence_duration,
                        frames=[frame],
                        probability=probability,
                        inference_duration=vad_duration,
                        speaking=speaking,
                        raw_accumulated_speech=speech_threshold_duration,
                        raw_accumulated_silence=silence_threshold_duration,
                    )
                )

                active = probability >= opts.activation_threshold or (
                    speaking and probability > opts.deactivation_threshold
                )
                if active:
                    speech_threshold_duration += frame_duration
                    silence_threshold_duration = 0.0
                    if not speaking and speech_threshold_duration >= opts.min_speech_duration:
                        speaking = True
                        signals.speaking = True
                        silence_duration = 0.0
                        speech_duration = speech_threshold_duration
                        speech_frames.clear()
                        speech_frames.extend(prefix)
                        self._event_ch.send_nowait(
                            agents.vad.VADEvent(
                                type=agents.vad.VADEventType.START_OF_SPEECH,
                                samples_index=samples_index,
                                timestamp=timestamp,
                                speech_duration=speech_duration,
                                silence_duration=0.0,
                                frames=[self._audio_frame(b"".join(speech_frames))],
                                speaking=True,
                            )
                        )
                else:
                    silence_threshold_duration += frame_duration
                    speech_threshold_duration = 0.0
                    endpoint_delay = opts.min_silence_duration
                    if (
                        opts.tp_endpointing_enabled
                        and self._tp_session is not None
                        and turn_probability >= opts.tp_threshold
                    ):
                        endpoint_delay = min(endpoint_delay, opts.tp_min_silence_duration)
                    if speaking and silence_threshold_duration >= endpoint_delay:
                        speaking = False
                        signals.speaking = False
                        silence_duration = silence_threshold_duration
                        self._event_ch.send_nowait(
                            agents.vad.VADEvent(
                                type=agents.vad.VADEventType.END_OF_SPEECH,
                                samples_index=samples_index,
                                timestamp=timestamp,
                                speech_duration=max(
                                    0.0, speech_duration - silence_threshold_duration
                                ),
                                silence_duration=silence_duration,
                                frames=[self._audio_frame(b"".join(speech_frames))],
                                speaking=False,
                            )
                        )
                        speech_duration = 0.0
                        speech_frames.clear()
