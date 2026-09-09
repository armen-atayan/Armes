"""Разделяемый (fan-out) VAD: один проход детекции на несколько потребителей.

Проблема. При не-стриминговом STT LiveKit оборачивает его в ``StreamAdapter``,
который создаёт СВОЙ ``vad.stream()`` для нарезки речи под ``recognize()``.
Параллельно ``AudioRecognition`` создаёт ВТОРОЙ ``vad.stream()`` того же VAD-плагина
для детекции конца хода. Получаются два независимых экземпляра одной и той же модели,
работающие на одном аудио, но с отдельным состоянием (гистерезис, EMA-фильтр). Они
могут разойтись на границах сегментов: один считает, что речь была, второй — нет.

Решение. ``FanoutVAD`` оборачивает реальный VAD и гоняет ровно ОДИН внутренний
поток. Первый созданный дочерний поток становится «первичным» — только он
проталкивает аудио во внутренний поток. События внутреннего потока (включая
``frames`` сегмента речи для ``recognize()``) транслируются ВСЕМ дочерним потокам.
Итог: одна инференс-нагрузка, идентичная сегментация у всех потребителей.

Использование::

    base_vad = KrispVivaVAD.load(...)
    vad = FanoutVAD(base_vad)
    super().__init__(vad=vad, ...)   # один объект и для turn-detection, и для StreamAdapter
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from livekit import agents
from livekit.agents.utils import aio


if TYPE_CHECKING:
    from livekit.agents import vad as vad_types


logger = logging.getLogger("livekit.plugins.krisp")


class FanoutVAD(agents.vad.VAD):
    """VAD-обёртка, разделяющая один внутренний ``VADStream`` между потребителями.

    Все вызовы :meth:`stream` получают события из одного и того же внутреннего
    потока. Аудио во внутренний поток проталкивает только первый созданный
    дочерний поток («первичный») — у него гарантированно «сырое» (не подменённое
    тишиной) аудио, так как ``AudioRecognition`` стартует свой VAD-поток до того,
    как ``StreamAdapter`` начнёт читать STT-пайплайн.
    """

    def __init__(self, inner: "vad_types.VAD") -> None:
        super().__init__(capabilities=inner.capabilities)
        self._inner = inner
        self._inner_stream: "vad_types.VADStream | None" = None
        self._pump_task: asyncio.Task | None = None
        self._subscribers: list["_FanoutVADStream"] = []

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def provider(self) -> str:
        return self._inner.provider

    def stream(self) -> "_FanoutVADStream":
        sub = _FanoutVADStream(self)
        is_first = self._inner_stream is None
        if is_first:
            self._inner_stream = self._inner.stream()
            sub._is_primary = True
            self._pump_task = asyncio.create_task(self._pump(), name="FanoutVAD._pump")
            logger.info("FanoutVAD: shared VAD stream created (single inference pass)")
        self._subscribers.append(sub)
        return sub

    async def _pump(self) -> None:
        """Читать события единственного внутреннего потока и транслировать всем."""
        assert self._inner_stream is not None
        try:
            async for ev in self._inner_stream:
                for sub in list(self._subscribers):
                    sub._feed(ev)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("FanoutVAD: inner VAD pump failed")

    def _detach(self, sub: "_FanoutVADStream") -> None:
        try:
            self._subscribers.remove(sub)
        except ValueError:
            pass
        # Когда закрыт первичный поток — гасим внутренний поток и насос.
        if sub._is_primary:
            if self._pump_task is not None:
                self._pump_task.cancel()
                self._pump_task = None
            if self._inner_stream is not None:
                stream = self._inner_stream
                self._inner_stream = None
                asyncio.create_task(stream.aclose(), name="FanoutVAD._inner_aclose")


class _FanoutVADStream(agents.vad.VADStream):
    """Дочерний поток: проксирует события из общего насоса в собственный канал."""

    def __init__(self, vad: FanoutVAD) -> None:
        self._fanout = vad
        self._is_primary = False
        self._relay_ch: "aio.Chan[vad_types.VADEvent]" = aio.Chan()
        super().__init__(vad)  # запускает _main_task

    async def _main_task(self) -> None:
        async for ev in self._relay_ch:
            self._event_ch.send_nowait(ev)

    async def _metrics_monitor_task(self, event_aiter) -> None:  # type: ignore[override]
        # Метрики собирает один внутренний поток у реального VAD; здесь только
        # дренируем tee, чтобы не копить буфер и не дублировать метрики.
        async for _ in event_aiter:
            pass

    def _feed(self, ev: "vad_types.VADEvent") -> None:
        if not self._relay_ch.closed:
            self._relay_ch.send_nowait(ev)

    # Аудио во внутренний поток проталкивает только первичный дочерний поток.
    def push_frame(self, frame) -> None:  # type: ignore[override]
        if self._is_primary and self._fanout._inner_stream is not None:
            self._fanout._inner_stream.push_frame(frame)

    def flush(self) -> None:
        if self._is_primary and self._fanout._inner_stream is not None:
            self._fanout._inner_stream.flush()

    def end_input(self) -> None:
        if self._is_primary and self._fanout._inner_stream is not None:
            self._fanout._inner_stream.end_input()

    async def aclose(self) -> None:
        self._relay_ch.close()
        self._fanout._detach(self)
        await super().aclose()
