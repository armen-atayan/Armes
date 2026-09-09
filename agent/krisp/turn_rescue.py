"""Спасение пользовательских ходов, завершившихся во время непрерываемой речи агента.

Проблема (наблюдалась в проде): если ход пользователя коммитится, пока агент
генерирует/озвучивает реплику с ``allow_interruptions=False`` (например,
STT-транскрипт пришёл позже коммита предыдущего хода),
``AgentActivity._user_turn_completed_task`` логирует ``"skipping reply to user
input, current speech generation cannot be interrupted"`` и выходит, НЕ добавляя
сообщение в chat_ctx — фраза пользователя уничтожается без ответа и без следа
в истории. При ``InterruptionOptions(enabled=False)`` непрерываемы ВСЕ реплики
и теряться может любой ход; при рекомендованном ``enabled=True`` — только ходы,
пришедшие во время явно защищённых реплик (приветствие, служебные ``say``).

``discard_audio_if_uninterruptible=False`` здесь не помогает: он управляет
только подменой аудио-фреймов тишиной для STT и не влияет на судьбу уже
закоммиченного хода.

Патч: перед выполнением оригинального ``_user_turn_completed_task`` дожидаемся
окончания текущей непрерываемой реплики (``SpeechHandle.wait_for_playout``).
К этому моменту ``_current_speech`` сброшен, и оригинальный код штатно
генерирует ответ на сохранённый транскрипт.

Политика модуля: НИ ОДИН непустой транскрипт не пропадает. Ход либо получает
ответ (сразу, после ожидания реплики или накопленным вместе с соседними
фразами), либо сохраняется в историю через ``skip_reply`` — тогда агент не
прерывается и не отвечает, но сказанное лежит в ``chat_ctx`` и модель увидит его
на следующем ходу. Решение «прерывать/отвечать» отделено от решения «сохранить»:
первое остаётся за ``allow_interruptions`` и Krisp IP, второе — да, но только при
включённом рубильнике ``set_keep_all_user_speech`` (флаг мерчанта
``keep_all_user_speech``, дефолт — ВЫКЛЮЧЕН). По умолчанию работает старое
поведение: ход, пришедший во время защищённой реплики, и поддакивание поверх
обычной — теряются.

Использование (один раз при старте, до создания AgentSession не обязательно —
патчится метод класса)::

    from krisp import patch_uninterruptible_turn_drop
    patch_uninterruptible_turn_drop()
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
import weakref
from collections import deque
from dataclasses import replace
from typing import Any

from livekit.agents import stt
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.events import AgentFalseInterruptionEvent
from livekit.agents.voice.speech_handle import SpeechHandle


logger = logging.getLogger("livekit.plugins.krisp")

_patch_applied: set[str] = set()

# Выставлен на время выполнения оригинального ``_user_turn_completed_task``.
# Внутри этого окна непрерываемая реплика, которую LiveKit пытается прервать
# как устаревшую (новый ход пользователя уже начался), форс-прерывается вместо
# падения с RuntimeError. См. _patch_speech_handle_interrupt.
_discarding_outdated_turn: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_krisp_discarding_outdated_turn", default=False
)

# session -> _BackchannelGate; патчированный _user_turn_completed_task ищет
# гейт по self._session
_gates: "weakref.WeakKeyDictionary[Any, _BackchannelGate]" = weakref.WeakKeyDictionary()

# session -> рубильник «сохранять всё сказанное клиентом» (см.
# set_keep_all_user_speech). Дефолт для незарегистрированной сессии — False:
# поведение до доработок, включается только явным вызовом сеттера.
_keep_all_speech: "weakref.WeakKeyDictionary[Any, bool]" = weakref.WeakKeyDictionary()


def set_keep_all_user_speech(session: Any, enabled: bool) -> None:
    """Включить/выключить сохранение всей речи клиента для сессии.

    Единственный рубильник на три поведения этого модуля:

    1. накопление речи, сказанной во время защищённой (no-rescue) реплики, и
       ответ на неё одним ходом после окончания реплики;
    2. сохранение хода без ответа, когда он мог быть эхом собственной озвучки
       агента (``_spoken_over_agent_audio``);
    3. сохранение хода без ответа, когда гейт счёл его поддакиванием
       (``_BackchannelGate.should_discard``).

    ``enabled=False`` (и дефолт) — поведение до этих доработок: ход, пришедший во
    время защищённой реплики, и поддакивание поверх обычной реплики просто
    отбрасываются, ни ответа, ни записи в историю. Включается точечно флагом
    мерчанта ``keep_all_user_speech`` в ``llm_config``: частота полезной речи
    поверх филлеров против частоты фонового шума на живых звонках пока не
    измерена, поэтому по умолчанию поведение не меняется ни у кого.

    Вызывать сразу после создания ``AgentSession`` — до ``on_enter``, иначе
    приветствие отработает с дефолтом.
    """
    _keep_all_speech[session] = bool(enabled)
    logger.info("turn-rescue: keep_all_user_speech=%s", bool(enabled))


def _keep_all_user_speech(activity: Any) -> bool:
    """Значение рубильника для сессии активности (по умолчанию — выключен)."""
    try:
        return _keep_all_speech.get(activity._session, False)
    except Exception:
        logger.exception("turn-rescue: не удалось прочитать keep_all_user_speech, считаем выключенным")
        return False


# Максимальное ожидание окончания реплики. Страховка от зависших SpeechHandle:
# по истечении отдаём управление оригинальному коду (в худшем случае он
# отработает как раньше — пропустит ответ).
_WAIT_PLAYOUT_TIMEOUT = 60.0

# Пауза между проверками, пока планировщик сбрасывает _current_speech после
# завершения playout (speech.done() уже True, но ссылка ещё не очищена).
_SCHEDULER_POLL_INTERVAL = 0.05

# Атрибут-маркер на SpeechHandle: реплику нельзя «спасать» rescue-логикой.
# Ставится на явно защищённые реплики (приветствие, служебные say вроде
# "Алло?"/"Вы ещё здесь?", филлеры перед долгими инструментами). Для них Krisp IP
# подавлен, поэтому отличить реальные слова пользователя от эха собственного
# голоса агента (AEC warmup, SIP-эхо) нельзя — а отвечать на услышанное поверх
# такой фразы продукт не должен. Ход, пришедший пока такая реплика ЗВУЧИТ,
# сохраняется в историю без ответа (``skip_reply``); ход, сказанный в тишину до
# начала озвучки, копится и получает полноценный ответ после неё
# (см. _spoken_over_agent_audio и _commit_without_reply).
_NO_RESCUE_ATTR = "_krisp_no_rescue"


def mark_no_rescue(handle: Any) -> Any:
    """Пометить ``SpeechHandle`` как защищённую от rescue-регенерации.

    Ход пользователя, закоммиченный пока такая реплика звучит, попадёт в историю
    без ответа (вероятное эхо); ход, сказанный в тишину до начала озвучки, —
    накоплен и озвучен ответом после её окончания. Ничего не теряется.
    Идемпотентно, безопасно для ``None``. Вызывается из ``protect_speech`` и
    вручную для приветствия в ``on_enter`` (где IP может быть выключен).
    """
    if handle is not None:
        try:
            setattr(handle, _NO_RESCUE_ATTR, True)
        except (AttributeError, TypeError):
            logger.debug("mark_no_rescue: не удалось пометить handle %r", handle)
    return handle


# Атрибут-маркер на SpeechHandle, оставленный для совместимости с флагом
# мерчанта ``save_parallel_speech``. На решение больше не влияет: речь, сказанную
# в тишину во время no-rescue реплики, копим всегда — терять сказанное клиентом
# нельзя ни при какой конфигурации.
_ACCUMULATE_ATTR = "_krisp_accumulate_parallel"


def mark_accumulate_parallel(handle: Any) -> Any:
    """Пометить no-rescue реплику как накапливающую параллельную речь.

    Исторический маркер: накопление теперь безусловное (см. ``_ACCUMULATE_ATTR``),
    и метка ни на что не влияет. Оставлена, чтобы не ломать вызовы по флагу
    мерчанта ``save_parallel_speech``. Идемпотентно, безопасно для ``None``.
    """
    if handle is not None:
        try:
            setattr(handle, _ACCUMULATE_ATTR, True)
        except (AttributeError, TypeError):
            logger.debug("mark_accumulate_parallel: не удалось пометить handle %r", handle)
    return handle


class _ParallelAccumulator:
    """Копит транскрипты ходов, пришедших во время накапливающей no-rescue реплики.

    Один экземпляр на сессию, пока реплика играет. ``flush_task`` — единственная
    фоновая корутина: дожидается окончания реплики и отдаёт склеенный транскрипт
    оригинальному обработчику одним ходом. Последующие параллельные ходы, пока
    ``flush_task`` жив, лишь дописываются в ``parts``.
    """

    __slots__ = ("parts", "flush_task")

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.flush_task: asyncio.Task | None = None


# session -> активный аккумулятор параллельной речи
_parallel_accumulators: "weakref.WeakKeyDictionary[Any, _ParallelAccumulator]" = weakref.WeakKeyDictionary()


# Доля речи пользователя, перекрытая озвучкой агента, начиная с которой ход,
# пришедший во время защищённой реплики, считается эхом собственного голоса
# агента. Порог низкий намеренно: различить эхо и реальное прерывание поверх
# такой реплики нечем (Krisp IP на ней подавлен), поэтому единственный надёжный
# признак «эха точно не было» — агент в это время не издавал звука вообще.
_ECHO_OVERLAP_THRESHOLD = 0.2


def _spoken_over_agent_audio(activity: Any, info: Any) -> bool:
    """Мог ли ход быть эхом: шла ли озвучка агента, пока пользователь говорил.

    ``False`` — пользователь говорил в тишину (или судить не по чему): эха
    физически быть не могло, ход терять нельзя.
    """
    gate = _gates.get(activity._session)
    if gate is None:
        return False  # гейт не подключён — таймингов нет, ход принимаем
    try:
        ratio = gate.agent_overlap_ratio(info)
    except Exception:
        logger.exception("turn-rescue: ошибка расчёта перекрытия речью агента, считаем что агент молчал")
        return False
    if ratio is None:
        logger.info("turn-rescue: тайминги речи неизвестны — считаем, что эха не было, ход принимаем")
        return False
    return ratio >= _ECHO_OVERLAP_THRESHOLD


async def _commit_without_reply(activity: Any, info: Any, original: Any) -> None:
    """Записать фразу в историю диалога, НЕ генерируя ответ на неё.

    Штатный механизм livekit ``skip_reply``: сообщение уходит в ``chat_ctx`` и в
    транскрипт сессии (``_conversation_item_added``), новая реплика агента не
    создаётся, текущая не прерывается (agent_activity, ветка ``if
    info.skip_reply``). Так решение «отвечать сейчас» отделено от решения
    «сохранить сказанное»: затранскрибированное не теряется никогда, а на
    следующем ходу модель увидит эту фразу в контексте и учтёт.
    """
    token = _discarding_outdated_turn.set(True)
    try:
        await original(activity, None, replace(info, skip_reply=True))
    except Exception:
        logger.exception(
            "turn-rescue: не удалось сохранить фразу в историю без ответа: %r",
            getattr(info, "new_transcript", None),
        )
    finally:
        _discarding_outdated_turn.reset(token)


def _accumulate_parallel_turn(activity: Any, info: Any, original: Any) -> None:
    """Накопить транскрипт хода и однократно завести фоновый сброс после реплики."""
    session = activity._session
    acc = _parallel_accumulators.get(session)
    if acc is None:
        acc = _ParallelAccumulator()
        _parallel_accumulators[session] = acc

    text = (getattr(info, "new_transcript", None) or "").strip()
    if text:
        acc.parts.append(text)
        logger.warning(
            "turn-rescue: реплика клиента накоплена во время защищённой реплики "
            "(keep_all_user_speech), ответим после её окончания: %r",
            text,
        )

    if acc.flush_task is None or acc.flush_task.done():
        # Через _create_speech_task (как штатный user-turn у livekit): выставляет
        # _AgentActivityContextVar/otel и регистрирует задачу для drain. Голый
        # ensure_future этого не делает, а _generate_reply на контекст опирается.
        acc.flush_task = activity._create_speech_task(
            _flush_parallel_after_speech(activity, info, original, acc),
            name="turn-rescue._flush_parallel_after_speech",
        )
        # Клиент добавил слова — всё, что старая цепочка (инструмент, начатый до
        # этой фразы) собиралась озвучить, устарело: ответ надо готовить заново
        # на всё сказанное. LiveKit прерывает генерацию, чей инициатор не равен
        # _user_turn_completed_atask, поэтому назначаем нашу задачу сразу, а не
        # после ожидания реплики — иначе успевший вернуться результат инструмента
        # озвучится ответом на одну первую фразу.
        activity._user_turn_completed_atask = acc.flush_task


def _pending_user_speech(activity: Any) -> str | None:
    """Висит ли в STT незакоммиченная речь клиента.

    ``None`` — не висит ничего: всё сказанное уже дошло до ``_user_turn_completed``
    (или внешнего STT нет вообще — realtime-модель). Иначе строка с содержимым
    буфера ``_audio_transcript``; она может быть и ПУСТОЙ, когда клиент говорит
    прямо сейчас или финал текущей фразы ещё в полёте, поэтому проверять результат
    надо через ``is not None``, а не на truthiness.

    Три признака (те же, по которым работает ``patch_stale_turn_commit``):

    - непустой ``_audio_transcript`` — фраза затранскрибирована, но EoT счёл мысль
      незаконченной (``INCOMPLETE``) и ход не закоммитил;
    - ``_speaking`` — клиент говорит в эту секунду;
    - ``_last_final_transcript_time < _last_speaking_time`` — финал в полёте.
    """
    try:
        ar = activity._audio_recognition
        if ar is None or activity.stt is None:
            return None
        buffered = ar._audio_transcript or ""
        if buffered.strip() or ar._speaking:
            return buffered
        last_speaking = ar._last_speaking_time
        last_final = ar._last_final_transcript_time
        if last_speaking is not None and (last_final is None or last_final < last_speaking):
            return buffered
        return None
    except Exception:
        logger.exception("turn-rescue: не удалось проверить буфер STT, считаем что клиент договорил")
        return None


async def _flush_parallel_after_speech(
    activity: Any, info_template: Any, original: Any, acc: _ParallelAccumulator
) -> None:
    """Дождаться окончания защищённой реплики и ответить на накопленное одним ходом.

    Исключение — клиент к этому моменту ещё не договорил (см.
    ``_pending_user_speech``): тогда накопленное уходит в историю без ответа, а
    ответ сгенерит штатный коммит недосказанной фразы. Иначе агент успевает
    ответить на огрызок («Рее.») раньше, чем клиент закончит мысль («Имя,
    фамилия и компания»), — а если на огрызок ещё и дёрнется инструмент с
    защищённым филлером, реальная фраза утонет под ним.
    """
    session = activity._session
    try:
        await _wait_until_interruptible(activity, "<накопленная параллельная речь>")
    except Exception:
        logger.exception("turn-rescue: ошибка ожидания окончания реплики при сбросе накопленного")
    finally:
        # Снимаем аккумулятор до генерации: параллельные ходы уже дописаны, а
        # новые (после реплики) должны начать свой цикл, а не долить сюда.
        if _parallel_accumulators.get(session) is acc:
            _parallel_accumulators.pop(session, None)

    combined = " ".join(p for p in acc.parts if p).strip()
    if not combined:
        return

    # Клиент не договорил: отвечать на накопленный огрызок нельзя — EoT ещё ждёт
    # продолжения мысли. Сохраняем сказанное в историю и уходим; ответ сгенерит
    # ход с недосказанной фразой, а накопленное модель увидит прямо перед ним.
    # Форсить коммит буфера здесь не пытаемся: это ровно то перебивание на
    # полуслове, против которого работает вердикт EoT. Если финал потеряется,
    # буфер добьёт _STALE_FINAL_GRACE в stale-guard, а полное молчание —
    # user_away_timeout.
    pending = _pending_user_speech(activity)
    if pending is not None:
        logger.warning(
            "turn-rescue: сброс накопленного отложен — клиент ещё не договорил "
            "(в буфере STT: %r), накопленное уходит в историю без ответа: %r",
            pending,
            combined,
        )
        await _commit_without_reply(activity, replace(info_template, new_transcript=combined), original)
        return

    # livekit прерывает свежесозданную генерацию, если задача-инициатор не равна
    # _user_turn_completed_atask (считает ход устаревшим, agent_activity ~2310).
    # Наш сброс — отдельная задача, поэтому назначаем её текущей user-turn прямо
    # перед генерацией: делаем это ПОСЛЕ ожидания (пока играло приветствие,
    # append-ходы перезаписывали atask на себя; теперь новых уже не будет).
    activity._user_turn_completed_atask = asyncio.current_task()

    # info_template.skip_reply здесь False (ход дошёл до этой ветки), поэтому
    # склонированный ход штатно сгенерирует ответ. metrics/confidence берём от
    # последнего исходного хода — для накопленного они всё равно приблизительны.
    synth = replace(info_template, new_transcript=combined)
    logger.info("turn-rescue: сброс накопленной за защищённую реплику речи одним ходом: %r", combined)
    try:
        await original(activity, None, synth)
    except Exception:
        logger.exception("turn-rescue: ошибка генерации ответа на накопленную речь")


async def _wait_until_interruptible(activity: Any, transcript: str) -> None:
    """Ждать, пока текущая непрерываемая реплика агента не закончится.

    Выходит сразу, если текущей реплики нет или она допускает прерывание.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + _WAIT_PLAYOUT_TIMEOUT
    waited = False

    while True:
        speech = activity._current_speech
        if speech is None or speech.allow_interruptions:
            break
        if loop.time() >= deadline:
            logger.warning(
                "turn-rescue: реплика агента не завершилась за %.0fs, передаём ход оригинальному обработчику как есть",
                _WAIT_PLAYOUT_TIMEOUT,
            )
            break
        if not waited:
            waited = True
            logger.warning(
                "turn-rescue: ход пользователя завершился во время непрерываемой "
                "реплики агента — ждём её окончания вместо сброса хода",
                extra={"user_input": transcript},
            )
        if speech.done():
            # playout закончен, ждём пока scheduling-цикл очистит _current_speech
            await asyncio.sleep(_SCHEDULER_POLL_INTERVAL)
            continue
        try:
            await asyncio.wait_for(speech.wait_for_playout(), timeout=max(0.0, deadline - loop.time()))
        except asyncio.TimeoutError:
            continue  # цикл сам обработает дедлайн

    if waited:
        logger.warning(
            "turn-rescue: реплика агента завершена, отвечаем на сохранённый ход",
            extra={"user_input": transcript},
        )


class _BackchannelGate:
    """Решает судьбу хода пользователя, закоммиченного во время речи агента.

    Логика (по спецификации продукта):

    - агент в момент речи пользователя молчал → ход ПРИНИМАЕТСЯ;
    - речь пользователя шла поверх озвучки агента и Krisp IP счёл её
      прерыванием (сработал callback) → ход ПРИНИМАЕТСЯ (агент уже прерван);
    - речь шла поверх озвучки агента, но Krisp IP промолчал → это
      поддакивание/фон → ОТВЕТ НЕ ГЕНЕРИТСЯ (агент не прерывается), но фраза
      сохраняется в историю через ``skip_reply`` — затранскрибированное не
      теряется, решает только судьбу ответа.

    Тайминги речи пользователя берутся из ``_EndOfTurnInfo``
    (``started_speaking_at``/``stopped_speaking_at``, wall-clock); если их нет —
    из собственного трекинга ``user_state_changed``. Интервалы озвучки агента
    трекаются по ``agent_state_changed``. При любой неопределённости ход
    принимается (безопасный дефолт — потерять поддакивание хуже, чем фразу).
    """

    def __init__(
        self,
        detector: Any,
        *,
        overlap_threshold: float = 0.5,
        ip_fire_margin: float = 0.5,
    ) -> None:
        self._detector_ref = weakref.ref(detector)
        self._overlap_threshold = overlap_threshold
        self._ip_fire_margin = ip_fire_margin

        # Закрытые интервалы озвучки агента (wall-clock) + текущий открытый
        self._agent_intervals: deque[tuple[float, float]] = deque(maxlen=20)
        self._agent_speaking_since: float | None = None

        # Fallback-тайминги речи пользователя
        self._user_speaking_since: float | None = None
        self._user_last_start: float | None = None
        self._user_last_stop: float | None = None

    # -- подписки на события сессии ------------------------------------

    def on_agent_state_changed(self, ev: Any) -> None:
        now = time.time()
        speaking = getattr(ev, "new_state", None) == "speaking"
        if speaking and self._agent_speaking_since is None:
            self._agent_speaking_since = now
        elif not speaking and self._agent_speaking_since is not None:
            self._agent_intervals.append((self._agent_speaking_since, now))
            self._agent_speaking_since = None

    def on_user_state_changed(self, ev: Any) -> None:
        now = time.time()
        state = getattr(ev, "new_state", None)
        if state == "speaking":
            self._user_speaking_since = now
            self._user_last_start = now
        elif self._user_speaking_since is not None:
            self._user_last_stop = now
            self._user_speaking_since = None

    # -- решение ---------------------------------------------------------

    def _agent_overlap(self, start: float, stop: float) -> float:
        """Сколько секунд из [start, stop] агент озвучивал аудио."""
        intervals = list(self._agent_intervals)
        if self._agent_speaking_since is not None:
            intervals.append((self._agent_speaking_since, time.time()))
        overlap = 0.0
        for a, b in intervals:
            overlap += max(0.0, min(stop, b) - max(start, a))
        return overlap

    def _ip_fired_since(self, t: float) -> bool:
        detector = self._detector_ref()
        if detector is None:
            return False
        fired_at = getattr(detector, "last_interrupt_at", None)
        return fired_at is not None and fired_at >= t

    def _turn_bounds(self, info: Any) -> tuple[float, float] | None:
        """Границы речи пользователя (wall-clock) или ``None``, если их не знаем.

        getattr с fallback: поле ``_EndOfTurnInfo`` есть не во всех версиях
        livekit-agents. Тайминги речи пользователя гейт трекает сам
        (``on_user_state_changed``) — это и есть запасной путь.
        """
        start = getattr(info, "started_speaking_at", None) or self._user_last_start
        stop = getattr(info, "stopped_speaking_at", None) or self._user_last_stop
        if start is None:
            return None
        if stop is None or stop < start:
            stop = time.time()
        return start, stop

    def agent_overlap_ratio(self, info: Any) -> float | None:
        """Доля речи пользователя, перекрытая озвучкой агента.

        ``0.0`` — пользователь говорил в полную тишину, ``None`` — тайминги
        неизвестны, судить не по чему.
        """
        bounds = self._turn_bounds(info)
        if bounds is None:
            return None
        start, stop = bounds
        return self._agent_overlap(start, stop) / max(stop - start, 1e-6)

    def should_discard(self, activity: Any, info: Any) -> bool:
        speech = activity._current_speech
        if speech is None or speech.done() or speech.interrupted:
            return False  # агент не говорит (или уже прерван) — принимаем
        if not speech.allow_interruptions:
            # защищённая реплика: гейт не судит, ход обработает wait-ветка
            return False

        bounds = self._turn_bounds(info)
        if bounds is None:
            return False  # тайминги неизвестны — принимаем
        start, stop = bounds

        overlap_ratio = self._agent_overlap(start, stop) / max(stop - start, 1e-6)
        if overlap_ratio < self._overlap_threshold:
            return False  # пользователь говорил в тишину — принимаем

        if self._ip_fired_since(start - self._ip_fire_margin):
            return False  # Krisp IP счёл это прерыванием — принимаем

        logger.warning(
            "krisp-gate: поддакивание — ответ не генерим, фразу сохраняем (overlap=%.0f%%, IP молчал): %r",
            overlap_ratio * 100,
            info.new_transcript,
        )
        return True


def attach_backchannel_gate(
    session: Any,
    interrupt_detector: Any,
    *,
    overlap_threshold: float = 0.5,
    ip_fire_margin: float = 0.5,
) -> _BackchannelGate:
    """Включить гейт поддакиваний для сессии.

    Вызывать после ``session.start(...)`` рядом с ``detector.attach(session)``::

        attach_backchannel_gate(self.agent_session, self._krisp_interrupt_detector)

    Args:
        session: Запущенная ``AgentSession``.
        interrupt_detector: ``KrispVivaInterruptDetector`` этой же сессии.
        overlap_threshold: Доля речи пользователя, перекрытая озвучкой агента,
            начиная с которой ход считается «сказанным поверх агента».
        ip_fire_margin: Запас (сек) до начала речи пользователя, в котором
            срабатывание Krisp IP ещё засчитывается этому ходу.
    """
    patch_uninterruptible_turn_drop()

    gate = _BackchannelGate(
        interrupt_detector,
        overlap_threshold=overlap_threshold,
        ip_fire_margin=ip_fire_margin,
    )
    session.on("agent_state_changed")(gate.on_agent_state_changed)
    session.on("user_state_changed")(gate.on_user_state_changed)
    _gates[session] = gate
    logger.info("krisp-gate: backchannel gate attached to session")
    return gate


def patch_uninterruptible_turn_drop() -> None:
    """Заменить ``AgentActivity._user_turn_completed_task`` версией с ожиданием.

    Идемпотентно: повторные вызовы ничего не делают.
    """
    if "turn_drop" in _patch_applied:
        return
    _patch_applied.add("turn_drop")

    original = AgentActivity._user_turn_completed_task

    async def _user_turn_completed_task_with_rescue(
        self: AgentActivity, old_task: asyncio.Task | None, info: Any
    ) -> None:
        # Диагностика «сдвига на шаг»: фиксируем, с каким транскриптом ход дошёл до
        # обработки и в каком состоянии была текущая реплика агента (прервётся ли она).
        _speech = self._current_speech
        logger.info(
            "turn-diag: ход принят: %r | current_speech=%s",
            getattr(info, "new_transcript", None),
            "none"
            if _speech is None
            else (
                f"done={_speech.done()} interrupted={_speech.interrupted} "
                f"allow_interruptions={_speech.allow_interruptions}"
            ),
        )
        if not info.skip_reply:
            # Пустой транскрипт: VAD засёк речь (шум/дыхание/эхо), но STT не
            # распознал ни слова (`STT empty result`). LiveKit всё равно
            # коммитит пустой user-turn, и LLM генерит ответ из контекста — без
            # реального ввода скатывается в пере-приветствие/общую фразу. Отвечать
            # не на что → выходим. Единственный случай, когда ход не сохраняется:
            # сохранять нечего, слов нет (livekit пустое в chat_ctx тоже не пишет).
            if not (getattr(info, "new_transcript", None) or "").strip():
                logger.warning("turn-rescue: ход отброшен — пустой транскрипт (STT empty), VAD-фантом")
                return
            # Ход пришёл, пока ещё не закончилась защищённая (no-rescue) реплика:
            # приветствие, служебный say, филлер перед долгим инструментом.
            # Отвечать сразу нельзя, если ход МОГ быть эхом собственного голоса
            # агента (пользователь говорил поверх звучащей озвучки, а Krisp IP на
            # защищённых репликах подавлен и рассудить не может) — такую фразу
            # только сохраняем в историю. Если агент в это время молчал — думал,
            # шёл в API, ещё не начал озвучку — эха быть не могло: это реальные
            # слова клиента. Копим их и после окончания реплики отвечаем на всё
            # сказанное одним ходом.
            speech = self._current_speech
            if (
                speech is not None
                and not speech.done()
                and not speech.interrupted
                and getattr(speech, _NO_RESCUE_ATTR, False)
            ):
                if not _keep_all_user_speech(self):
                    # Рубильник выключен: поведение до доработок — ход, пришедший
                    # во время защищённой реплики, отбрасывается целиком.
                    logger.warning(
                        "turn-rescue: ход отброшен — пришёл во время защищённой реплики "
                        "(keep_all_user_speech=False): %r",
                        info.new_transcript,
                    )
                    return
                if _spoken_over_agent_audio(self, info):
                    logger.warning(
                        "turn-rescue: ход сказан поверх звучащей защищённой реплики (вероятное эхо) — "
                        "не отвечаем, но сохраняем в историю: %r",
                        info.new_transcript,
                    )
                    await _commit_without_reply(self, info, original)
                    return
                _accumulate_parallel_turn(self, info, original)
                return

            discard_reply = False
            try:
                gate = _gates.get(self._session)
                discard_reply = gate is not None and gate.should_discard(self, info)
            except Exception:
                logger.exception("turn-rescue: ошибка backchannel-гейта, принимаем ход как есть")
            if discard_reply:
                # Поддакивание поверх речи агента: агента не прерываем и ответ не
                # генерим (иначе он будет останавливаться на каждое «ага»), но
                # сказанное сохраняем — на следующем ходу модель это увидит.
                # При выключенном рубильнике — старое поведение, ход отбрасывается.
                if _keep_all_user_speech(self):
                    await _commit_without_reply(self, info, original)
                return
            try:
                await _wait_until_interruptible(self, info.new_transcript)
            except Exception:
                logger.exception("turn-rescue: ошибка при ожидании окончания реплики, продолжаем без ожидания")
        # Пока работает оригинал, форсируем прерывание устаревшей непрерываемой
        # реплики (LiveKit делает это безусловным interrupt() на line 2111 и
        # падает с RuntimeError, когда allow_interruptions=False).
        token = _discarding_outdated_turn.set(True)
        try:
            await original(self, old_task, info)
        finally:
            _discarding_outdated_turn.reset(token)
            logger.info("turn-diag: ход обработан: %r", getattr(info, "new_transcript", None))

    _patch_speech_handle_interrupt()
    AgentActivity._user_turn_completed_task = _user_turn_completed_task_with_rescue
    logger.info(
        "turn-rescue patch applied: user turns completed during uninterruptible "
        "agent speech will wait for playout instead of being dropped"
    )


def patch_false_interruption_min_timeout() -> None:
    """Не давать false-interruption таймеру резюмить речь раньше конфигового таймаута.

    Проблема: когда пользователь начинает говорить, пока агент в ``thinking``,
    LiveKit ставит запланированную реплику на паузу с ``timeout=0``
    (``AgentActivity.on_start_of_speech``), рассчитывая, что
    ``_interrupt_by_audio_activity`` поднимет таймаут до
    ``false_interruption_timeout`` по interim-транскриптам. Наш STT interim'ы не
    шлёт, поэтому гейт ``min_words`` видит пустой ``current_transcript`` и не
    срабатывает → на VAD end-of-speech таймер стартует с 0 → мгновенный резюм →
    агент озвучивает устаревший ответ, а финал реплики пользователя приходит
    позже и повисает в буфере (начало «сдвига на шаг»).

    Патч клампит таймаут снизу конфиговым ``false_interruption_timeout``: пауза
    живёт достаточно, чтобы финал STT успел прийти, прервать паузу
    (``_cancel_speech_pause`` на финале) и перегенерировать ответ уже с учётом
    новой фразы. Идемпотентно.
    """
    if "fi_min_timeout" in _patch_applied:
        return
    _patch_applied.add("fi_min_timeout")

    original = AgentActivity._start_false_interruption_timer

    def _start_false_interruption_timer(self: AgentActivity, timeout: float) -> None:
        try:
            configured = self._session.options.interruption["false_interruption_timeout"] or 0.0
        except Exception:
            configured = 0.0
        if timeout < configured:
            logger.info(
                "fi-timeout: клампим false-interruption таймаут %.2fs -> %.2fs (ждём финал STT)",
                timeout,
                configured,
            )
            timeout = configured
        # Пауза могла быть захвачена в состоянии "thinking" (юзер заговорил до
        # старта плейаута — ``on_start_of_speech``), а плейаут после этого успел
        # начаться. Резюм восстанавливает захваченное состояние, и протухший
        # "thinking" ломает всё дальше: следующий VAD-бурст снова мгновенно
        # паузит (ветка ``agent_state != "speaking"``), а диагностика видит
        # фантомный silent turn. Освежаем захват по фактическому состоянию.
        try:
            if self._paused_speech is not None and self._session.agent_state == "speaking":
                self._paused_speech.agent_state = "speaking"
        except Exception:
            logger.exception("fi-timeout: не удалось освежить agent_state паузы")
        return original(self, timeout)

    AgentActivity._start_false_interruption_timer = _start_false_interruption_timer
    logger.info("fi-timeout patch applied: false interruption resume clamped to configured timeout")


def _resume_paused_speech(activity: AgentActivity) -> bool:
    """Немедленно снять паузу ложного прерывания (зеркало ``_on_false_interruption``).

    Тело повторяет closure ``_on_false_interruption`` из
    ``AgentActivity._start_false_interruption_timer``: восстановить состояние
    агента, возобновить аудиовыход, сэмитить ``agent_false_interruption`` и
    почистить ``_paused_speech`` вместе с таймером.
    """
    paused = activity._paused_speech
    if paused is None:
        return False
    if activity._current_speech is not None and activity._current_speech is not paused.handle:
        # уже запланирована новая реплика — паузу просто забываем (паритет с оригиналом)
        activity._paused_speech = None
        return False

    resumed = False
    interruption_options = activity._session.options.interruption
    audio_output = activity._session.output.audio
    if (
        interruption_options["resume_false_interruption"]
        and audio_output is not None
        and audio_output.can_pause
        and not paused.handle.done()
    ):
        state = paused.agent_state
        # Захваченный "thinking" мог протухнуть, если плейаут стартовал уже
        # после постановки на паузу (см. patch_false_interruption_min_timeout).
        if activity._session.agent_state == "speaking":
            state = "speaking"
        activity._session._update_agent_state(state, otel_context=paused.handle._agent_turn_context)
        if activity._audio_recognition and state == "speaking":
            activity._audio_recognition._on_start_of_agent_speech(started_at=time.time())
        if activity.interruption_enabled:
            activity._disable_vad_interruption_soon()
        audio_output.resume()
        resumed = True

    activity._session.emit("agent_false_interruption", AgentFalseInterruptionEvent(resumed=resumed))
    activity._paused_speech = None
    if activity._false_interruption_timer is not None:
        activity._false_interruption_timer.cancel()
        activity._false_interruption_timer = None
    return resumed


def patch_resume_on_empty_stt_final() -> None:
    """Пустой финал STT при стоящей на паузе речи = подтверждённое ложное прерывание.

    Проблема (наблюдалась в проде): агент говорит, клиент шумит/бубнит поверх,
    VAD паузит речь, а STT раз за разом возвращает пустой текст. Пустые финалы
    LiveKit молча глотает (``audio_recognition: if not transcript: return``) —
    до ``on_final_transcript`` и его ``_cancel_speech_pause`` они не доходят,
    а false-interruption таймер каждый новый VAD-бурст отменяет, ОСТАВЛЯЯ паузу
    (``on_start_of_speech``). Итог — дедлок: агент молчит посреди фразы, пока
    STT наконец не распознает реальный текст, который эту фразу добивает
    ``interrupt()`` уже насовсем. Клиент слышит обрыв на полуслове и мёртвую
    тишину на 10-20 секунд.

    Патч: пустой финал — это доказательство, что прерывание было ложным
    (полезной речи от клиента не пришло), поэтому резюмим речь сразу, не дожидаясь
    таймера. Если клиент в этот момент реально заговорит, штатное прерывание
    отработает по непустому транскрипту. Идемпотентно.
    """
    if "resume_on_empty_final" in _patch_applied:
        return
    _patch_applied.add("resume_on_empty_final")

    original = AudioRecognition._on_stt_event

    async def _on_stt_event_with_empty_final_resume(self: AudioRecognition, ev: stt.SpeechEvent) -> None:
        await original(self, ev)
        try:
            if ev.type != stt.SpeechEventType.FINAL_TRANSCRIPT or not ev.alternatives:
                return
            if (ev.alternatives[0].text or "").strip():
                return
            activity = self._hooks  # RecognitionHooks == AgentActivity
            if getattr(activity, "_paused_speech", None) is None:
                return
            if _resume_paused_speech(activity):
                logger.info("empty-final: пустой финал STT — резюмим речь агента, не дожидаясь fi-таймера")
        except Exception:
            logger.exception("empty-final: ошибка резюма по пустому финалу STT")

    AudioRecognition._on_stt_event = _on_stt_event_with_empty_final_resume
    logger.info("empty-final patch applied: paused speech resumes immediately on empty STT finals")


# Сколько ждать свежий финал STT после отложенного коммита, прежде чем
# закоммитить буфер как есть (страховка от пустого/потерянного финала).
_STALE_FINAL_GRACE = 2.0


def patch_stale_turn_commit() -> None:
    """Не коммитить ход, пока не пришёл финал STT текущей реплики пользователя.

    Проблема («ответы на шаг назад»): внешний не-стриминговый STT отдаёт финал
    через 0.3–1.2s после VAD end-of-speech. Если в ``_audio_transcript`` уже
    лежит несмытый транскрипт ПРЕДЫДУЩЕЙ реплики, EoT по VAD (endpointing
    ``min_delay=0.1``) коммитит ход раньше прихода свежего финала — агент
    отвечает на прошлую фразу, свежая уезжает в следующий ход, и сдвиг
    самоподдерживается до конца звонка.

    Детект: ``_last_final_transcript_time < _last_speaking_time`` — пользователь
    закончил говорить позже, чем пришёл последний финал ⇒ финал текущей реплики
    ещё в полёте. Возвращаем ``False`` (LiveKit сохраняет буфер, как в штатной
    ветке min_words) — свежий финал сам перезапустит EoT и закоммитит обе фразы
    одним ходом. Страховка: если финал не придёт за ``_STALE_FINAL_GRACE``
    (пустой результат STT, потеря), форсим EoT с тем, что есть.

    Дефер максимум один раз на каждое окончание речи (метка по
    ``_last_speaking_time``), чтобы не зациклиться. Идемпотентно.
    """
    if "stale_commit" in _patch_applied:
        return
    _patch_applied.add("stale_commit")

    original = AgentActivity.on_end_of_turn

    def _schedule_stale_retry(activity: AgentActivity, last_speaking: float) -> None:
        ar = activity._audio_recognition

        def _retry() -> None:
            try:
                if ar is None or ar._audio_transcript == "":
                    return  # буфер уже закоммичен/очищен
                last_final = ar._last_final_transcript_time
                if last_final is not None and last_final >= last_speaking:
                    return  # финал пришёл — EoT перезапущен штатно
                if ar._speaking:
                    return  # пользователь снова говорит — коммит придёт от его EOS
                logger.warning(
                    "stale-guard: свежий финал не пришёл за %.1fs, коммитим буфер как есть: %r",
                    _STALE_FINAL_GRACE,
                    ar._audio_transcript,
                )
                ar._run_eou_detection(ar._hooks.retrieve_chat_ctx().copy(), trigger="stt")
            except Exception:
                logger.exception("stale-guard: ошибка форс-коммита отложенного хода")

        asyncio.get_running_loop().call_later(_STALE_FINAL_GRACE, _retry)

    def on_end_of_turn(self: AgentActivity, info: Any) -> bool:
        try:
            ar = self._audio_recognition
            if ar is not None and self.stt is not None:
                last_speaking = ar._last_speaking_time
                last_final = ar._last_final_transcript_time
                if (
                    last_speaking is not None
                    and (last_final is None or last_final < last_speaking)
                    and getattr(self, "_krisp_stale_deferred_for", None) != last_speaking
                ):
                    self._krisp_stale_deferred_for = last_speaking
                    logger.warning(
                        "stale-guard: коммит отложен — финал STT текущей реплики ещё не пришёл "
                        "(буфер: %r, last_final=%s, last_speaking=%.3f)",
                        getattr(info, "new_transcript", None),
                        f"{last_final:.3f}" if last_final is not None else "none",
                        last_speaking,
                    )
                    _schedule_stale_retry(self, last_speaking)
                    return False
        except Exception:
            logger.exception("stale-guard: ошибка проверки свежести буфера, коммитим как есть")
        return original(self, info)

    AgentActivity.on_end_of_turn = on_end_of_turn
    logger.info("stale-guard patch applied: turn commit waits for the current utterance STT final")


def _patch_speech_handle_interrupt() -> None:
    """Сделать прерывание устаревшей непрерываемой реплики безопасным.

    LiveKit в ``_user_turn_completed_task`` отбрасывает реплику, ставшую
    неактуальной (за время её генерации пришёл новый ход пользователя),
    безусловным ``await speech_handle.interrupt()``. Для агента с
    ``allow_interruptions=False`` это бросает
    ``RuntimeError("This generation handle does not allow interruptions")`` и
    рушит всю цепочку ``await old_task`` (наблюдалось в конце звонка).

    Это внутренний сброс мусора, а не пользовательское прерывание, поэтому в
    окне выполнения ``_user_turn_completed_task`` (флаг
    ``_discarding_outdated_turn``) подменяем такой бросок на ``force=True`` —
    ровно то, что и подразумевал фреймворк. Вне этого окна и для прерываемых
    реплик поведение ``interrupt`` не меняется.
    """
    original_interrupt = SpeechHandle.interrupt

    def interrupt(self: SpeechHandle, *, force: bool = False) -> SpeechHandle:
        if not force and not self._allow_interruptions and _discarding_outdated_turn.get():
            logger.warning(
                "turn-rescue: форс-прерывание устаревшей непрерываемой реплики "
                "(новый ход пользователя уже начался) вместо падения с RuntimeError"
            )
            return original_interrupt(self, force=True)
        return original_interrupt(self, force=force)

    SpeechHandle.interrupt = interrupt
