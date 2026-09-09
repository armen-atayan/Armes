# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Krisp VIVA VAD (Voice Activity Detection) for LiveKit Agents.

This module provides a VAD implementation backed by the Krisp VIVA SDK
(``krisp-viva-vad-v2.kef`` model). It implements the ``livekit.agents.vad.VAD``
interface and can be used as a drop-in replacement for Silero VAD.

Usage::

    from krisp.viva_vad import KrispVivaVAD

    vad = KrispVivaVAD.load(
        activation_threshold=0.5,
        min_silence_duration=0.6,
        prefix_padding_duration=0.3,
    )

    session = AgentSession(
        vad=vad,
        ...
    )
"""

from __future__ import annotations

import asyncio
import os
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from livekit import agents, rtc
from livekit.agents import utils

from krisp.krisp_instance import (
    KRISP_FRAME_DURATIONS,
    KrispSDKManager,
    int_to_krisp_frame_duration,
    int_to_krisp_sample_rate,
)

import logging

logger = logging.getLogger("livekit.plugins.krisp")

try:
    import krisp_audio  # type: ignore[import-not-found]

    KRISP_AUDIO_AVAILABLE = True
except ModuleNotFoundError:
    KRISP_AUDIO_AVAILABLE = False
    logger.warning(
        "krisp-audio package not found. "
        "Install it to use Krisp VAD: pip install krisp-audio"
    )


@dataclass
class _VADOptions:
    """Configuration options for Krisp VIVA VAD."""

    model_path: str
    frame_duration_ms: int
    sample_rate: int
    activation_threshold: float
    deactivation_threshold: float
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float


class KrispVivaVAD(agents.vad.VAD):
    """Voice Activity Detection using the Krisp VIVA SDK.

    Implements the ``livekit.agents.vad.VAD`` interface so it can be passed
    directly to ``AgentSession`` or ``VoicePipelineAgent`` in place of Silero.

    The underlying Krisp model processes PCM-16 audio in fixed-size frames
    and returns a speech probability per frame.  This class accumulates frames
    and applies hysteresis (activation / deactivation thresholds) to emit
    ``START_OF_SPEECH`` and ``END_OF_SPEECH`` events that match the standard
    LiveKit VAD contract.

    Example::

        vad = KrispVivaVAD.load(
            activation_threshold=0.5,
            min_silence_duration=0.6,
        )
    """

    @classmethod
    def load(
        cls,
        *,
        model_path: str | None = None,
        frame_duration_ms: int = 10,
        sample_rate: int = 8000,
        activation_threshold: float = 0.5,
        deactivation_threshold: float | None = None,
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.55,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
    ) -> KrispVivaVAD:
        """Load and initialise the Krisp VIVA VAD model.

        This call is **blocking** — the model is loaded into memory here.
        Run it inside a ``prewarm`` function or before starting the agent loop.

        Args:
            model_path: Path to the ``.kef`` VAD model file.
                Falls back to the ``KRISP_VIVA_VAD_MODEL_PATH`` environment
                variable when *None*.
            frame_duration_ms: Frame size passed to the Krisp session.
                Must be one of ``{10, 15, 20, 30, 32}``. Default: ``10``.
            sample_rate: Hint for the Krisp session sample rate. In practice the
                session is created from the actual ``input_frame.sample_rate``
                on the first received frame, so this value is only used when
                ``load()`` needs to validate the model before any audio arrives.
                Must be one of ``{8000, 16000, 24000, 32000, 44100, 48000}``.
                Default: ``8000`` (telephony).
            activation_threshold: Probability above which a frame is considered
                speech (0–1). Default: ``0.5``.
            deactivation_threshold: Probability below which the detector exits
                the *speaking* state. Defaults to
                ``max(activation_threshold - 0.15, 0.01)``.
            min_speech_duration: Minimum continuous speech duration (seconds)
                before a ``START_OF_SPEECH`` event is emitted. Default: ``0.05``.
            min_silence_duration: Minimum continuous silence duration (seconds)
                before an ``END_OF_SPEECH`` event is emitted. Default: ``0.55``.
            prefix_padding_duration: Audio padding (seconds) prepended to each
                speech segment. Default: ``0.5``.
            max_buffered_speech: Maximum speech duration (seconds) kept in the
                ring buffer. Default: ``60.0``.

        Returns:
            A ready-to-use :class:`KrispVivaVAD` instance.

        Raises:
            RuntimeError: If the ``krisp-audio`` package is not installed or
                the SDK cannot be initialised.
            ValueError: If *model_path* is not provided and the environment
                variable is unset, or if an unsupported *frame_duration_ms* /
                *sample_rate* is specified.
            FileNotFoundError: If the model file does not exist.
        """
        if not KRISP_AUDIO_AVAILABLE:
            raise RuntimeError(
                "krisp-audio package is not installed. "
                "Run: pip install krisp-audio"
            )

        resolved_model_path = model_path or os.getenv("KRISP_VIVA_VAD_MODEL_PATH")
        if not resolved_model_path:
            raise ValueError(
                "Krisp VAD model path is not provided and "
                "KRISP_VIVA_VAD_MODEL_PATH environment variable is not set."
            )

        if not resolved_model_path.endswith(".kef"):
            raise ValueError("Krisp model file must have a .kef extension.")

        if not Path(resolved_model_path).is_file():
            raise FileNotFoundError(
                f"Krisp VAD model file not found: {resolved_model_path}"
            )

        if frame_duration_ms not in KRISP_FRAME_DURATIONS:
            raise ValueError(
                f"Unsupported frame_duration_ms: {frame_duration_ms}. "
                f"Supported: {sorted(KRISP_FRAME_DURATIONS.keys())}"
            )

        resolved_deactivation = (
            deactivation_threshold
            if deactivation_threshold is not None
            else max(activation_threshold - 0.15, 0.01)
        )

        opts = _VADOptions(
            model_path=resolved_model_path,
            frame_duration_ms=frame_duration_ms,
            sample_rate=sample_rate,
            activation_threshold=activation_threshold,
            deactivation_threshold=resolved_deactivation,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
        )
        return cls(opts=opts)

    def __init__(self, *, opts: _VADOptions) -> None:
        """Initialise the VAD and acquire the shared Krisp SDK reference.

        Args:
            opts: Validated configuration options.

        Raises:
            RuntimeError: If the Krisp SDK cannot be acquired.
        """
        # update_interval controls how often VADMetrics are emitted
        super().__init__(
            capabilities=agents.vad.VADCapabilities(
                update_interval=opts.frame_duration_ms / 1000.0
            )
        )
        self._opts = opts
        self._streams: weakref.WeakSet[KrispVivaVADStream] = weakref.WeakSet()
        self._sdk_acquired = False

        try:
            KrispSDKManager.acquire()
            self._sdk_acquired = True
        except Exception as e:
            raise RuntimeError(f"Failed to acquire Krisp SDK: {e}") from e

    @property
    def model(self) -> str:
        """Identifier of the underlying model."""
        return "krisp-viva-vad"

    @property
    def provider(self) -> str:
        """Name of the inference provider."""
        return "krisp"

    def stream(self) -> KrispVivaVADStream:
        """Create a new :class:`KrispVivaVADStream`.

        Returns:
            A fresh stream ready to accept audio frames.
        """
        vad_stream = KrispVivaVADStream(self, self._opts)
        self._streams.add(vad_stream)
        return vad_stream

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


class KrispVivaVADStream(agents.vad.VADStream):
    """Async stream that processes audio frames through Krisp VIVA VAD.

    Audio frames are pushed via :meth:`push_frame` and the stream emits
    :class:`~livekit.agents.vad.VADEvent` objects (``INFERENCE_DONE``,
    ``START_OF_SPEECH``, ``END_OF_SPEECH``) as an async iterator.

    The stream accumulates raw audio in a ring buffer so that the full speech
    segment (including the configured prefix padding) is available in the
    ``frames`` field of ``START_OF_SPEECH`` and ``END_OF_SPEECH`` events.
    """

    def __init__(self, vad: KrispVivaVAD, opts: _VADOptions) -> None:
        super().__init__(vad)
        self._opts = opts
        self._loop = asyncio.get_event_loop()

        # Krisp session – created lazily on first frame so we know the actual
        # input sample rate (which may differ from opts.sample_rate).
        self._krisp_session: Any | None = None
        self._session_sample_rate: int | None = None

        # Per-stream state (mirrors Silero VAD stream)
        self._input_sample_rate: int = 0
        self._speech_buffer: np.ndarray | None = None
        self._speech_buffer_max_reached: bool = False
        self._prefix_padding_samples: int = 0

        self._exp_filter = utils.ExpFilter(alpha=0.35)

    # ------------------------------------------------------------------
    # Krisp session helpers
    # ------------------------------------------------------------------

    def _create_krisp_session(self, sample_rate: int) -> Any:
        """Create a Krisp VAD session for *sample_rate*.

        Args:
            sample_rate: The sample rate of the audio to be processed.

        Returns:
            A ``krisp_audio.VadInt16`` session instance.

        Raises:
            ValueError: If *sample_rate* is not supported by the SDK.
        """
        model_info = krisp_audio.ModelInfo()
        model_info.path = self._opts.model_path

        vad_cfg = krisp_audio.VadSessionConfig()
        vad_cfg.inputSampleRate = int_to_krisp_sample_rate(sample_rate)
        vad_cfg.inputFrameDuration = int_to_krisp_frame_duration(
            self._opts.frame_duration_ms
        )
        vad_cfg.modelInfo = model_info

        session = krisp_audio.VadInt16.create(vad_cfg)
        logger.info(
            f"Krisp VAD session created: {sample_rate}Hz / "
            f"{self._opts.frame_duration_ms}ms frames"
        )
        return session

    def _ensure_session(self, sample_rate: int) -> bool:
        """Ensure a Krisp VAD session exists for *sample_rate*.

        Creates a new session if none exists or if the sample rate changed.

        Args:
            sample_rate: Desired audio sample rate in Hz.

        Returns:
            ``True`` if the session is ready, ``False`` on error.
        """
        if self._krisp_session is not None and self._session_sample_rate == sample_rate:
            return True
        try:
            self._krisp_session = self._create_krisp_session(sample_rate)
            self._session_sample_rate = sample_rate
            return True
        except Exception:
            logger.exception(
                f"Failed to create Krisp VAD session for {sample_rate}Hz"
            )
            self._krisp_session = None
            return False

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    @agents.utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:  # noqa: C901 (complex but linear)
        """Async task that consumes frames and emits VAD events."""
        # Number of samples per Krisp inference window (at inference rate)
        window_size_samples: int | None = None

        # Pending input frames (at original input sample rate)
        input_frames: list[rtc.AudioFrame] = []
        # Pending frames resampled to the Krisp session rate
        inference_frames: list[rtc.AudioFrame] = []

        # Running time / sample counters
        pub_speaking = False
        pub_speech_duration = 0.0
        pub_silence_duration = 0.0
        pub_current_sample = 0
        pub_timestamp = 0.0

        speech_threshold_duration = 0.0
        silence_threshold_duration = 0.0

        speech_buffer_index = 0


        async for input_frame in self._input_ch:
            if not isinstance(input_frame, rtc.AudioFrame):
                # FlushSentinel — honour it but don't reset state
                continue

            # ---- first-frame initialisation ----
            if not self._input_sample_rate:
                self._input_sample_rate = input_frame.sample_rate

                self._prefix_padding_samples = int(
                    self._opts.prefix_padding_duration * self._input_sample_rate
                )
                self._speech_buffer = np.empty(
                    int(self._opts.max_buffered_speech * self._input_sample_rate)
                    + self._prefix_padding_samples,
                    dtype=np.int16,
                )

                # Create the Krisp session at the actual incoming sample rate so
                # there is no need for a resampler.  opts.sample_rate is only used
                # as a fallback hint when load() is called without a first frame yet.
                if not self._ensure_session(self._input_sample_rate):
                    logger.error(
                        "Cannot create Krisp VAD session – passing frames through silently"
                    )
                    continue

                window_size_samples = int(
                    self._input_sample_rate * self._opts.frame_duration_ms / 1000
                )

                logger.info(
                    f"Krisp VAD stream initialised: "
                    f"{self._input_sample_rate}Hz / {self._opts.frame_duration_ms}ms "
                    f"({window_size_samples} samples/window)"
                )

            elif self._input_sample_rate != input_frame.sample_rate:
                logger.error(
                    "Incoming sample rate changed mid-stream – frame ignored"
                )
                continue

            assert self._speech_buffer is not None
            assert window_size_samples is not None

            input_frames.append(input_frame)
            inference_frames.append(input_frame)

            # ---- process all complete windows available ----
            while True:
                available = sum(f.samples_per_channel for f in inference_frames)
                if available < window_size_samples:
                    break

                start_time = time.perf_counter()

                combined_input = utils.combine_frames(input_frames)
                combined_inference = utils.combine_frames(inference_frames)

                # Guard: inference buffer must contain a full window.
                # This should always hold given the while-loop condition above,
                # but we double-check to avoid passing a truncated array to Krisp.
                if combined_inference.samples_per_channel < window_size_samples:
                    break

                # Slice exactly one inference window (PCM-16 → bytes)
                window_bytes = combined_inference.data[: window_size_samples * 2]
                window_samples = np.frombuffer(window_bytes, dtype=np.int16).copy()

                if len(window_samples) != window_size_samples:
                    logger.debug(
                        f"Krisp VAD: inference window size mismatch "
                        f"(expected {window_size_samples}, got {len(window_samples)}) — skipping"
                    )
                    break

                # Run Krisp VAD inference in a thread pool to keep async loop free
                p_raw: float = await self._loop.run_in_executor(
                    None, self._krisp_session.process, window_samples
                )
                p = self._exp_filter.apply(exp=1.0, sample=float(p_raw))

                window_duration = window_size_samples / self._opts.sample_rate

                pub_current_sample += window_size_samples
                pub_timestamp += window_duration

                # No resampling: input and inference share the same sample rate.
                to_copy = min(window_size_samples, combined_input.samples_per_channel)

                # Accumulate into speech ring buffer
                available_space = len(self._speech_buffer) - speech_buffer_index
                to_copy_buf = min(to_copy, available_space)
                if to_copy_buf > 0:
                    src = np.frombuffer(
                        combined_input.data[: to_copy_buf * 2], dtype=np.int16
                    )
                    self._speech_buffer[
                        speech_buffer_index : speech_buffer_index + to_copy_buf
                    ] = src
                    speech_buffer_index += to_copy_buf
                elif not self._speech_buffer_max_reached:
                    self._speech_buffer_max_reached = True
                    logger.warning(
                        "Krisp VAD: max_buffered_speech reached, "
                        "ignoring further data for the current speech segment"
                    )

                inference_duration = time.perf_counter() - start_time

                if pub_speaking:
                    pub_speech_duration += window_duration
                else:
                    pub_silence_duration += window_duration

                # Emit INFERENCE_DONE for every processed window
                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.INFERENCE_DONE,
                        samples_index=pub_current_sample,
                        timestamp=pub_timestamp,
                        silence_duration=pub_silence_duration,
                        speech_duration=pub_speech_duration,
                        probability=p,
                        inference_duration=inference_duration,
                        frames=[
                            rtc.AudioFrame(
                                data=combined_input.data[: to_copy * 2],
                                sample_rate=self._input_sample_rate,
                                num_channels=1,
                                samples_per_channel=to_copy,
                            )
                        ],
                        speaking=pub_speaking,
                        raw_accumulated_silence=silence_threshold_duration,
                        raw_accumulated_speech=speech_threshold_duration,
                    )
                )

                # ---- hysteresis state machine ----
                is_speech = p >= self._opts.activation_threshold or (
                    pub_speaking and p > self._opts.deactivation_threshold
                )

                if is_speech:
                    speech_threshold_duration += window_duration
                    silence_threshold_duration = 0.0

                    if not pub_speaking:
                        if speech_threshold_duration >= self._opts.min_speech_duration:
                            pub_speaking = True
                            pub_silence_duration = 0.0
                            pub_speech_duration = speech_threshold_duration

                            self._event_ch.send_nowait(
                                agents.vad.VADEvent(
                                    type=agents.vad.VADEventType.START_OF_SPEECH,
                                    samples_index=pub_current_sample,
                                    timestamp=pub_timestamp,
                                    silence_duration=pub_silence_duration,
                                    speech_duration=pub_speech_duration,
                                    frames=[self._copy_speech_buffer(speech_buffer_index)],
                                    speaking=True,
                                )
                            )
                else:
                    silence_threshold_duration += window_duration
                    speech_threshold_duration = 0.0

                    if not pub_speaking:
                        speech_buffer_index = self._reset_write_cursor(speech_buffer_index)

                    if (
                        pub_speaking
                        and silence_threshold_duration >= self._opts.min_silence_duration
                    ):
                        pub_speaking = False
                        pub_silence_duration = silence_threshold_duration

                        self._event_ch.send_nowait(
                            agents.vad.VADEvent(
                                type=agents.vad.VADEventType.END_OF_SPEECH,
                                samples_index=pub_current_sample,
                                timestamp=pub_timestamp,
                                silence_duration=pub_silence_duration,
                                speech_duration=max(
                                    0.0,
                                    pub_speech_duration - silence_threshold_duration,
                                ),
                                frames=[self._copy_speech_buffer(speech_buffer_index)],
                                speaking=False,
                            )
                        )

                        pub_speech_duration = 0.0
                        speech_buffer_index = self._reset_write_cursor(speech_buffer_index)

                # ---- consume used frames, keep remainder ----
                input_frames = []
                inference_frames = []

                remaining_input_bytes = combined_input.data[to_copy * 2 :]
                if remaining_input_bytes:
                    input_frames.append(
                        rtc.AudioFrame(
                            data=remaining_input_bytes,
                            sample_rate=self._input_sample_rate,
                            num_channels=1,
                            samples_per_channel=len(remaining_input_bytes) // 2,
                        )
                    )

                remaining_inf_bytes = combined_inference.data[window_size_samples * 2 :]
                if remaining_inf_bytes:
                    inference_frames.append(
                        rtc.AudioFrame(
                            data=remaining_inf_bytes,
                            sample_rate=self._opts.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(remaining_inf_bytes) // 2,
                        )
                    )

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _reset_write_cursor(self, speech_buffer_index: int) -> int:
        """Slide the ring buffer so only prefix-padding bytes are retained.

        Args:
            speech_buffer_index: Current write position in the speech buffer.

        Returns:
            Updated write position after the reset.
        """
        assert self._speech_buffer is not None

        if speech_buffer_index <= self._prefix_padding_samples:
            return speech_buffer_index

        padding_data = self._speech_buffer[
            speech_buffer_index - self._prefix_padding_samples : speech_buffer_index
        ]
        self._speech_buffer_max_reached = False
        self._speech_buffer[: self._prefix_padding_samples] = padding_data
        return self._prefix_padding_samples

    def _copy_speech_buffer(self, speech_buffer_index: int) -> rtc.AudioFrame:
        """Copy the accumulated speech from the ring buffer into an AudioFrame.

        Args:
            speech_buffer_index: Current write position (exclusive end index).

        Returns:
            An :class:`~livekit.rtc.AudioFrame` containing the speech audio.
        """
        assert self._speech_buffer is not None

        speech_data = self._speech_buffer[:speech_buffer_index].tobytes()
        return rtc.AudioFrame(
            sample_rate=self._input_sample_rate,
            num_channels=1,
            samples_per_channel=speech_buffer_index,
            data=speech_data,
        )
