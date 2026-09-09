"""Exercise the installed SDK's automatic response with a deterministic LLM node."""
import asyncio
import json
from unittest.mock import Mock
from types import SimpleNamespace

import pytest
from livekit.agents import Agent, AgentSession, llm
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo, _EndOfTurnMetrics
from livekit.plugins import openai
import gen2b_agent as g


class OfflineLLM(openai.LLM):
    async def _prewarm_impl(self):
        pass


class ScriptedAgent(Agent):
    def __init__(self, first_output):
        super().__init__(instructions="test", llm=OfflineLLM(api_key="unused"))
        self.first_output = first_output
        self.requests = []
        self.silent = False
        self.settings = []

    async def llm_node(self, chat_ctx, tools, model_settings):
        self.requests.append(chat_ctx.copy())
        self.settings.append(model_settings)
        if self.silent:
            return
        if len(self.requests) == 1:
            if self.first_output:
                yield self.first_output
        else:
            yield "Здравствуйте, подскажите, пожалуйста, можно стол на 10 человек завтра в 20:00?"


@pytest.mark.asyncio
@pytest.mark.parametrize("first_output,expected", [("", 2), ("Здравствуйте, можно стол?", 1)])
async def test_first_committed_turn_empty_fallback_or_normal_no_duplicate(first_output, expected):
    session = AgentSession()
    agent = ScriptedAgent(first_output)
    config = g.resolve_call_config(json.dumps({
        "persona": "armen_personal_assistant",
        "task": "Забронируй стол на 10 человек завтра в 20:00",
    }))
    protect = Mock()
    g.attach_callee_first_greeting(session, config["greeting_instructions"], protect_speech=protect)
    handles = []
    session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
    await session.start(agent)
    try:
        await asyncio.sleep(0)
        assert not agent.requests  # never greet before the callee
        session._activity._user_turn_completed_atask = asyncio.current_task()
        await session._activity._user_turn_completed_task(None, _EndOfTurnInfo(
            skip_reply=False, new_transcript="Тоже", transcript_confidence=1.0,
            metrics=_EndOfTurnMetrics(None, None, None, None),
        ))
        await asyncio.wait_for(handles[0].wait_for_playout(), 2)
        assert handles[0].interrupted is False
        assert len(handles) == expected
        if not first_output:
            assert handles[0].chat_items == []  # exact no-speech/no-tool response
            await asyncio.wait_for(handles[1].wait_for_playout(), 2)
            assert any(config["greeting_instructions"] in (item.text_content or "")
                       for item in agent.requests[1].items if item.type == "message")
            assert handles[1].allow_interruptions is False
            assert handles[1]._krisp_no_rescue is True
            assert agent.settings[1].tool_choice == "none"
            assert "Здравствуйте" in handles[1].chat_items[0].text_content
            protect.assert_called_once_with(handles[1])
        else:
            protect.assert_not_called()
        assert len(agent.requests) == expected
        # Later silent replies must never restart the task greeting.
        agent.silent = True
        later = session.generate_reply()
        await asyncio.wait_for(later.wait_for_playout(), 2)
        assert len(handles) == expected + 1
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["no_caller", "blank_caller", "stt_only", "tool", "interrupted", "closed"])
async def test_does_not_greet_for_uncommitted_turn_tool_interruption_or_close(case):
    session = AgentSession()
    session.generate_reply = Mock()
    g.attach_callee_first_greeting(session, "task greeting")
    handle = SpeechHandle.create()
    session.emit("speech_created", SimpleNamespace(source="generate_reply", speech_handle=handle))
    if case == "stt_only":
        session.emit("user_input_transcribed", SimpleNamespace(transcript="Алло", is_final=True))
    elif case != "no_caller":
        session.emit("conversation_item_added", SimpleNamespace(item=llm.ChatMessage(
            role="user", content=[" " if case == "blank_caller" else "Алло"],
        )))
    if case == "tool":
        handle._item_added([llm.FunctionCall(call_id="owner", name="ask_owner", arguments="{}")])
    elif case == "interrupted":
        handle.interrupt()
    elif case == "closed":
        session.emit("close", SimpleNamespace())
    handle._mark_done()
    await asyncio.sleep(0)  # dispatch SpeechHandle's public done callbacks
    session.generate_reply.assert_not_called()
