import asyncio

import gen2b_agent as g


async def _chunks(*parts: str):
    for part in parts:
        yield part


async def _collect(*parts: str) -> list[str]:
    return [part async for part in g.adjust_tts_question_punctuation(_chunks(*parts))]


async def _collect_personal(*parts: str) -> list[str]:
    from spoken_ack import normalize_personal_assistant_text
    return [part async for part in g.adjust_tts_question_punctuation(normalize_personal_assistant_text(_chunks(*parts)))]


def test_question_mark_is_accented_for_tts():
    assert "".join(asyncio.run(_collect("Столик на четыре человека подойдёт?"))) == (
        "Столик на четыре человека подойдёт!!!?"
    )


def test_punctuation_run_crossing_stream_chunks_is_normalized_once():
    assert "".join(asyncio.run(_collect("Подтвердите?", "! Тогда бронирую."))) == (
        "Подтвердите!!!? Тогда бронирую."
    )


def test_plain_text_is_streamed_without_buffering():
    assert asyncio.run(_collect("Здравствуйте, ", "я ассистент.")) == [
        "Здравствуйте, ", "я ассистент."
    ]


def test_personal_assistant_mozhno_sentence_ends_with_period_not_question_intonation():
    assert "".join(asyncio.run(_collect_personal("Можно ", "забронировать пять столов?"))) == (
        "Можно забронировать пять столов."
    )


def test_personal_assistant_regular_question_keeps_question_intonation():
    assert "".join(asyncio.run(_collect_personal("Получится забронировать пять столов?"))) == (
        "Получится забронировать пять столов!!!?"
    )
