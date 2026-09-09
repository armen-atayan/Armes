from types import SimpleNamespace

import pytest

import gen2b_agent as g


@pytest.mark.asyncio
async def test_tts_node_streams_one_unique_demo_turn(monkeypatch):
    emitted = []
    monkeypatch.setattr(g, "emit_demo_event", lambda sid, room, kind, payload: emitted.append((kind, payload)))

    async def source():
        yield "Добрый "
        yield "день."

    async def fake_default(_self, text, _settings):
        async for chunk in text:
            yield chunk.encode()

    monkeypatch.setattr(g.Agent.default, "tts_node", fake_default)
    agent = object.__new__(g.Gen2BAssistant)
    agent._room_name = "room-web"
    agent._transcript_emitter = g.DemoTranscriptEmitter("demo_web", "room-web")
    agent._personal_assistant = False

    frames = [frame async for frame in g.Gen2BAssistant.tts_node(agent, source(), SimpleNamespace())]

    assert frames
    assert [kind for kind, _ in emitted] == [
        "transcript.assistant.delta",
        "transcript.assistant.delta",
        "transcript.assistant.final",
    ]
    ids = {payload["utterance_id"] for _, payload in emitted}
    assert len(ids) == 1
    assert emitted[-1][1]["text"] == "Добрый день."


@pytest.mark.asyncio
async def test_separate_tts_turns_get_separate_demo_ids(monkeypatch):
    emitted = []
    monkeypatch.setattr(g, "emit_demo_event", lambda sid, room, kind, payload: emitted.append((kind, payload)))

    async def fake_default(_self, text, _settings):
        async for chunk in text:
            yield chunk.encode()

    async def source(value):
        yield value

    monkeypatch.setattr(g.Agent.default, "tts_node", fake_default)
    agent = object.__new__(g.Gen2BAssistant)
    agent._room_name = "room-web"
    agent._transcript_emitter = g.DemoTranscriptEmitter("demo_web", "room-web")
    agent._personal_assistant = False

    for value in ("Первый ответ.", "Второй ответ."):
        _ = [frame async for frame in g.Gen2BAssistant.tts_node(agent, source(value), SimpleNamespace())]

    finals = [payload for kind, payload in emitted if kind == "transcript.assistant.final"]
    assert [item["text"] for item in finals] == ["Первый ответ.", "Второй ответ."]
    assert finals[0]["utterance_id"] != finals[1]["utterance_id"]


def test_demo_uses_tts_stream_as_the_only_assistant_transcript_source():
    import inspect
    source = inspect.getsource(g.entrypoint)
    assert 'session.on("conversation_item_added", record_conversation_item)' not in source


@pytest.mark.asyncio
async def test_personal_assistant_tts_uses_male_wording_and_replaces_smotrite(monkeypatch):
    spoken = []

    async def source():
        yield "Смотрите... А я уточнила: можно со своим алкоголем?"

    async def fake_default(_self, text, _settings):
        async for chunk in text:
            spoken.append(chunk)
            yield chunk.encode()

    monkeypatch.setattr(g.Agent.default, "tts_node", fake_default)
    agent = object.__new__(g.Gen2BAssistant)
    agent._room_name = "room-web"
    agent._transcript_emitter = None
    agent._personal_assistant = True

    _ = [frame async for frame in g.Gen2BAssistant.tts_node(agent, source(), SimpleNamespace())]

    emitted_text = "".join(spoken)
    assert emitted_text == "Подскажите, пожалуйста, я уточнил: можно со своим алкоголем!!!?"
    assert "Смотрите" not in emitted_text
    assert "уточнила" not in emitted_text


@pytest.mark.asyncio
async def test_personal_assistant_tts_does_not_add_question_intonation_to_mozhno(monkeypatch):
    spoken = []

    async def source():
        yield "Можно забронировать пять столов?"

    async def fake_default(_self, text, _settings):
        async for chunk in text:
            spoken.append(chunk)
            yield chunk.encode()

    monkeypatch.setattr(g.Agent.default, "tts_node", fake_default)
    agent = object.__new__(g.Gen2BAssistant)
    agent._room_name = "room-web"
    agent._transcript_emitter = None
    agent._personal_assistant = True

    _ = [frame async for frame in g.Gen2BAssistant.tts_node(agent, source(), SimpleNamespace())]

    assert "".join(spoken) == "Можно забронировать пять столов."


@pytest.mark.asyncio
async def test_tts_intonation_can_be_disabled_per_agent(monkeypatch):
    spoken = []

    async def source():
        yield "Подскажите?"

    async def fake_default(_self, text, _settings):
        async for chunk in text:
            spoken.append(chunk)
            yield chunk.encode()

    monkeypatch.setattr(g.Agent.default, "tts_node", fake_default)
    agent = object.__new__(g.Gen2BAssistant)
    agent._room_name = "room-web"
    agent._transcript_emitter = None
    agent._personal_assistant = False
    agent._tts_intonation = False

    _ = [frame async for frame in g.Gen2BAssistant.tts_node(agent, source(), SimpleNamespace())]

    assert "".join(spoken) == "Подскажите?"
