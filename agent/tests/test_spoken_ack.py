"""Offline tests for the low-latency spoken acknowledgement filter."""

import asyncio
import importlib.util

import pytest


def normalize_personal(chunks):
    assert importlib.util.find_spec("spoken_ack") is not None, "normalizer is missing"
    from spoken_ack import normalize_personal_assistant_text

    async def source():
        for chunk in chunks:
            yield chunk

    async def collect():
        return "".join([chunk async for chunk in normalize_personal_assistant_text(source())])

    return asyncio.run(collect())


def normalize(chunks):
    assert importlib.util.find_spec("spoken_ack") is not None, "normalizer is missing"
    from spoken_ack import normalize_acknowledgements

    async def source():
        for chunk in chunks:
            yield chunk

    async def collect():
        return "".join([chunk async for chunk in normalize_acknowledgements(source())])

    return asyncio.run(collect())


def test_initial_acknowledgement():
    assert normalize(["Понял."]) == "Хорошо."


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Смотрите... А можно у вас со своим алкоголем?", "Подскажите, пожалуйста, можно у вас со своим алкоголем?"),
        ("Смотрите, стол у окна будет?", "Подскажите, пожалуйста, стол у окна будет?"),
        ("Хорошо, поняла.", "Хорошо, понял."),
        ("Я уточнила условия.", "Я уточнил условия."),
        ("А я уточнил: можно со своим алкоголем?", "А я уточнил: можно со своим алкоголем?"),
        ("Это возможно?", "Получится?"),
        ("Это возможно сделать?", "Получится сделать?"),
        ("Это допустимо?", "Можно?"),
    ],
)
def test_personal_assistant_male_wording_and_no_smotrite(source, expected):
    assert normalize_personal([source]) == expected
    assert normalize_personal(list(source)) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Понял.", "Хорошо."),
        ("Понятно, спасибо.", "Хорошо, спасибо."),
        ("понял", "хорошо"),
        ("понятно", "хорошо"),
        ("ПОНЯЛ!", "Хорошо!"),
        ("ПОНЯТНО.", "Хорошо."),
        ("пОнЯл, записываю", "хорошо, записываю"),
        ("Понял вас", "Хорошо вас"),
    ],
)
@pytest.mark.parametrize("chunking", ["whole", "characters", "every_split"])
def test_target_tokens_across_chunk_boundaries(source, expected, chunking):
    if chunking == "whole":
        assert normalize([source]) == expected
    elif chunking == "characters":
        assert normalize(list(source)) == expected
    else:
        for split in range(len(source) + 1):
            assert normalize([source[:split], "", source[split:]]) == expected


@pytest.mark.parametrize(
    "source",
    [
        "непонятно", "я понял адрес", "Я Понял.", "Всё понятно?",
        "Понятное дело", "Поняла.", "Поняли?", "Понялось", "Понял2",
        "Понял_адрес", "Понятно-таки", "Понял\u0301.", "Понятно?",
        "понятно?!", "ПОНЯТНО?", "Понял?", "П", "По", "Поня", "Понят",
        "Понятн", "Понялся. Понятность важна.", "Да, понял. Да; понятно.",
        "", " \t\n", "Просто текст без подтверждения",
    ],
)
def test_preserves_non_acknowledgements(source):
    assert normalize([source]) == source
    assert normalize(list(source)) == source
    for split in range(len(source) + 1):
        assert normalize([source[:split], source[split:]]) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (" \tПонял,  спасибо!", " \tХорошо,  спасибо!"),
        ("«Понятно».", "«Хорошо»."),
        ('"Понял!"', '"Хорошо!"'),
        ("'понятно'", "'хорошо'"),
        ("“Понял.”", "“Хорошо.”"),
        ("„Понятно“", "„Хорошо“"),
        ("«Понятно?»", "«Понятно?»"),
        ("Понял. Понятно! понял… понятно", "Хорошо. Хорошо! хорошо… хорошо"),
        ("Это адрес.\t«Понятно, спасибо».", "Это адрес.\t«Хорошо, спасибо»."),
        ("Готовы? Понял. Да! понятно.", "Готовы? Хорошо. Да! хорошо."),
        ("я понял адрес\nпонятно\r\nПонял", "я понял адрес\nхорошо\r\nХорошо"),
        ("— Понял.\n— Понятно.", "— Хорошо.\n— Хорошо."),
        ("(Понятно.) Понял.", "(Хорошо.) Хорошо."),
        ("Он сказал «Понял».", "Он сказал «Понял»."),
        ("Понятно, понял.", "Хорошо, понял."),
    ],
)
def test_sentence_boundaries_and_leading_quotes(source, expected):
    assert normalize([source]) == expected
    assert normalize(list(source)) == expected
    for split in range(len(source) + 1):
        assert normalize([source[:split], source[split:]]) == expected


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["П", "он", "ял", ","], "Хорошо,"),
        (["Понятно", " "], "Хорошо "),
        (["П", "онятно", "?"], "Понятно?"),
        (["По", "е"], "Пое"),
        (["Понятно", "е"], "Понятное"),
        ([" \t" * 1000], " \t" * 1000),
        (["«"], "«"),
        (["я уже записываю " * 1000], "я уже записываю " * 1000),
        (["Понял, ", "я уже записываю " * 1000], "Хорошо, " + "я уже записываю " * 1000),
    ],
    ids=["ack-comma", "ack-space", "question", "prefix-mismatch", "word-suffix",
         "whitespace", "quote", "non-ack-long", "ack-long-unfinished-sentence"],
)
def test_emits_available_text_while_upstream_remains_open(chunks, expected):
    from spoken_ack import normalize_acknowledgements

    async def check():
        release = asyncio.Event()
        upstream_closed = False

        async def source():
            nonlocal upstream_closed
            try:
                for chunk in chunks:
                    yield chunk
                await release.wait()
                yield " продолжение"
            finally:
                upstream_closed = True

        incoming = source()
        outgoing = normalize_acknowledgements(incoming)
        received = ""

        async def read_available():
            nonlocal received
            while len(received) < len(expected):
                received += await anext(outgoing)

        try:
            # A sentence/turn-buffering implementation would deadlock here.
            await asyncio.wait_for(read_available(), timeout=1)
            assert received == expected
            assert not release.is_set()
            assert not upstream_closed
            release.set()
            assert "".join([part async for part in outgoing]) == " продолжение"
            assert upstream_closed
        finally:
            release.set()
            await outgoing.aclose()
            await incoming.aclose()

    asyncio.run(check())


@pytest.mark.parametrize("token", ["Понял", "Понятно", "понятно"])
def test_waits_for_next_character_to_preserve_question(token):
    from spoken_ack import normalize_acknowledgements

    async def check():
        waiting_for_delimiter = asyncio.Event()
        release = asyncio.Event()

        async def source():
            for char in token:
                yield char
            waiting_for_delimiter.set()
            await release.wait()
            yield "?"

        incoming = source()
        outgoing = normalize_acknowledgements(incoming)
        first = asyncio.create_task(anext(outgoing))
        try:
            await asyncio.wait_for(waiting_for_delimiter.wait(), timeout=1)
            assert not first.done(), "target must not escape before its delimiter"
            release.set()
            assert await asyncio.wait_for(first, timeout=1) == token + "?"
            assert [part async for part in outgoing] == []
        finally:
            release.set()
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await outgoing.aclose()
            await incoming.aclose()

    asyncio.run(check())
